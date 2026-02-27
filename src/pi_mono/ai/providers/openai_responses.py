"""OpenAI Responses API provider.

Implements :class:`~pi_mono.ai.providers.base.LLMProvider` using the
newer OpenAI Responses API (``/v1/responses``).  Handles streaming,
tool calls, reasoning items, and images.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

from openai import AsyncOpenAI

from pi_mono.ai.env_api_keys import get_env_api_key
from pi_mono.ai.models import calculate_cost, supports_xhigh
from pi_mono.ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    Cost,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Model,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingLevel,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    Usage,
    UserMessage,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OPENAI_TOOL_CALL_PROVIDERS = frozenset({"openai", "openai-codex", "opencode"})


# ---------------------------------------------------------------------------
# Provider-specific options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenAIResponsesOptions(StreamOptions):
    """Extended stream options for the Responses API."""

    reasoning_effort: ThinkingLevel | None = None
    reasoning_summary: Literal["auto", "detailed", "concise"] | None = None
    service_tier: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_streaming_json(partial: str | None) -> dict[str, Any]:
    """Parse potentially incomplete JSON from streaming.

    Always returns a valid dict, even when the JSON is truncated.
    """
    if not partial or not partial.strip():
        return {}
    try:
        return json.loads(partial)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        cleaned = partial
        open_braces = cleaned.count("{") - cleaned.count("}")
        open_brackets = cleaned.count("[") - cleaned.count("]")
        cleaned += "]" * max(open_brackets, 0)
        cleaned += "}" * max(open_braces, 0)
        try:
            return json.loads(cleaned)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            return {}


def _short_hash(s: str) -> str:
    """Fast deterministic hash to shorten long strings."""
    h1 = 0xDEADBEEF
    h2 = 0x41C6CE57
    mask = 0xFFFFFFFF
    for ch in s:
        c = ord(ch)
        h1 = ((h1 ^ c) * 2654435761) & mask
        h2 = ((h2 ^ c) * 1597334677) & mask
    h1 = (((h1 ^ (h1 >> 16)) * 2246822507) ^ ((h2 ^ (h2 >> 13)) * 3266489909)) & mask
    h2 = (((h2 ^ (h2 >> 16)) * 2246822507) ^ ((h1 ^ (h1 >> 13)) * 3266489909)) & mask
    # Convert to base-36 strings
    def to_base36(n: int) -> str:
        if n == 0:
            return "0"
        chars = "0123456789abcdefghijklmnopqrstuvwxyz"
        result = []
        while n:
            result.append(chars[n % 36])
            n //= 36
        return "".join(reversed(result))

    return to_base36(h2 & mask) + to_base36(h1 & mask)


def _clamp_reasoning(
    effort: ThinkingLevel | None,
) -> ThinkingLevel | None:
    """Clamp ``xhigh`` down to ``high`` for models that don't support it."""
    if effort == "xhigh":
        return "high"
    return effort


def _resolve_cache_retention(cache_retention: str | None) -> str:
    """Resolve cache retention preference. Defaults to ``"short"``."""
    if cache_retention:
        return cache_retention
    env_val = os.environ.get("PI_CACHE_RETENTION")
    if env_val == "long":
        return "long"
    return "short"


def _get_prompt_cache_retention(
    base_url: str,
    cache_retention: str,
) -> str | None:
    """Return prompt cache retention if applicable to direct OpenAI calls."""
    if cache_retention != "long":
        return None
    if "api.openai.com" in base_url:
        return "24h"
    return None


def _map_stop_reason(status: str | None) -> StopReason:
    """Map an OpenAI Responses API status to a :class:`StopReason`."""
    if not status:
        return "stop"
    mapping: dict[str, StopReason] = {
        "completed": "stop",
        "incomplete": "length",
        "failed": "error",
        "cancelled": "error",
        "in_progress": "stop",
        "queued": "stop",
    }
    return mapping.get(status, "stop")


def _get_service_tier_multiplier(service_tier: str | None) -> float:
    """Return cost multiplier for the given service tier."""
    if service_tier == "flex":
        return 0.5
    if service_tier == "priority":
        return 2.0
    return 1.0


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def _convert_responses_messages(
    model: Model,
    context: Context,
) -> list[dict[str, Any]]:
    """Convert pi_mono messages to OpenAI Responses API input format."""

    def normalize_tool_call_id(tool_id: str) -> str:
        if model.provider not in _OPENAI_TOOL_CALL_PROVIDERS:
            return tool_id
        if "|" not in tool_id:
            return tool_id
        import re
        call_id, item_id = tool_id.split("|", 1)
        sanitized_call = re.sub(r"[^a-zA-Z0-9_-]", "_", call_id)
        sanitized_item = re.sub(r"[^a-zA-Z0-9_-]", "_", item_id)
        if not sanitized_item.startswith("fc"):
            sanitized_item = f"fc_{sanitized_item}"
        sanitized_call = sanitized_call[:64].rstrip("_")
        sanitized_item = sanitized_item[:64].rstrip("_")
        return f"{sanitized_call}|{sanitized_item}"

    messages: list[dict[str, Any]] = []

    if context.system_prompt:
        role = "developer" if model.reasoning else "system"
        messages.append({
            "role": role,
            "content": context.system_prompt,
        })

    msg_index = 0
    for msg in context.messages:
        if msg.role == "user":
            assert isinstance(msg, UserMessage)
            if isinstance(msg.content, str):
                messages.append({
                    "role": "user",
                    "content": [{"type": "input_text", "text": msg.content}],
                })
            else:
                parts: list[dict[str, Any]] = []
                for item in msg.content:
                    if isinstance(item, TextContent):
                        parts.append({"type": "input_text", "text": item.text})
                    elif isinstance(item, ImageContent):
                        parts.append({
                            "type": "input_image",
                            "detail": "auto",
                            "image_url": f"data:{item.mime_type};base64,{item.data}",
                        })
                filtered = (
                    [p for p in parts if p.get("type") != "input_image"]
                    if "image" not in model.input
                    else parts
                )
                if not filtered:
                    msg_index += 1
                    continue
                messages.append({"role": "user", "content": filtered})

        elif msg.role == "assistant":
            assert isinstance(msg, AssistantMessage)
            output_items: list[dict[str, Any]] = []

            is_different_model = (
                msg.model != model.id
                and msg.provider == model.provider
                and msg.api == model.api
            )

            for block in msg.content:
                if isinstance(block, ThinkingContent):
                    if block.thinking_signature:
                        try:
                            reasoning_item = json.loads(block.thinking_signature)
                            output_items.append(reasoning_item)
                        except json.JSONDecodeError:
                            pass
                elif isinstance(block, TextContent):
                    msg_id = block.text_signature
                    if not msg_id:
                        msg_id = f"msg_{msg_index}"
                    elif len(msg_id) > 64:
                        msg_id = f"msg_{_short_hash(msg_id)}"
                    output_items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": block.text,
                                "annotations": [],
                            },
                        ],
                        "status": "completed",
                        "id": msg_id,
                    })
                elif isinstance(block, ToolCall):
                    normalized = normalize_tool_call_id(block.id)
                    if "|" in normalized:
                        call_id, item_id_raw = normalized.split("|", 1)
                    else:
                        call_id = normalized
                        item_id_raw = None

                    item_id: str | None = item_id_raw
                    if is_different_model and item_id and item_id.startswith("fc_"):
                        item_id = None

                    fc_item: dict[str, Any] = {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": block.name,
                        "arguments": json.dumps(block.arguments),
                    }
                    if item_id is not None:
                        fc_item["id"] = item_id
                    output_items.append(fc_item)

            if output_items:
                messages.extend(output_items)

        elif msg.role == "toolResult":
            assert isinstance(msg, ToolResultMessage)
            text_parts = [
                c.text for c in msg.content if isinstance(c, TextContent)
            ]
            text_result = "\n".join(text_parts)
            has_images = any(isinstance(c, ImageContent) for c in msg.content)

            has_text = len(text_result) > 0
            normalized_id = normalize_tool_call_id(msg.tool_call_id)
            call_id = normalized_id.split("|")[0] if "|" in normalized_id else normalized_id

            messages.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": text_result if has_text else "(see attached image)",
            })

            if has_images and "image" in model.input:
                content_parts: list[dict[str, Any]] = [
                    {"type": "input_text", "text": "Attached image(s) from tool result:"},
                ]
                for block in msg.content:
                    if isinstance(block, ImageContent):
                        content_parts.append({
                            "type": "input_image",
                            "detail": "auto",
                            "image_url": f"data:{block.mime_type};base64,{block.data}",
                        })
                messages.append({"role": "user", "content": content_parts})

        msg_index += 1

    return messages


def _convert_responses_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Convert pi_mono Tool definitions to OpenAI Responses tool format."""
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": False,
        }
        for tool in tools
    ]


# ---------------------------------------------------------------------------
# Client / params builders
# ---------------------------------------------------------------------------


def _create_client(
    model: Model,
    api_key: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> AsyncOpenAI:
    """Create an :class:`AsyncOpenAI` client for the Responses API."""
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass it as an argument."
            )

    headers: dict[str, str] = dict(model.headers) if model.headers else {}
    if extra_headers:
        headers.update(extra_headers)

    return AsyncOpenAI(
        api_key=api_key,
        base_url=model.base_url,
        default_headers=headers or None,
    )


def _build_params(
    model: Model,
    context: Context,
    messages: list[dict[str, Any]],
    options: OpenAIResponsesOptions | None,
) -> dict[str, Any]:
    """Build the ``responses.create`` request payload."""
    cache_retention = _resolve_cache_retention(
        options.cache_retention if options else None,
    )

    params: dict[str, Any] = {
        "model": model.id,
        "input": messages,
        "stream": True,
        "store": False,
    }

    if cache_retention != "none" and options and options.session_id:
        params["prompt_cache_key"] = options.session_id

    retention = _get_prompt_cache_retention(model.base_url, cache_retention)
    if retention:
        params["prompt_cache_retention"] = retention

    if options and options.max_tokens:
        params["max_output_tokens"] = options.max_tokens

    if options and options.temperature is not None:
        params["temperature"] = options.temperature

    if options and options.service_tier is not None:
        params["service_tier"] = options.service_tier

    if context.tools:
        params["tools"] = _convert_responses_tools(context.tools)

    if model.reasoning:
        reasoning_effort = options.reasoning_effort if options else None
        reasoning_summary = options.reasoning_summary if options else None
        if reasoning_effort or reasoning_summary:
            params["reasoning"] = {
                "effort": reasoning_effort or "medium",
                "summary": reasoning_summary or "auto",
            }
            params["include"] = ["reasoning.encrypted_content"]
        else:
            if model.name.startswith("gpt-5"):
                messages.append({
                    "role": "developer",
                    "content": [
                        {"type": "input_text", "text": "# Juice: 0 !important"},
                    ],
                })

    return params


# ---------------------------------------------------------------------------
# Mutable streaming accumulators
# ---------------------------------------------------------------------------


@dataclass
class _MutableBlock:
    """Mutable wrapper used while accumulating streamed content blocks."""

    type: str
    text: str = ""
    thinking: str = ""
    thinking_signature: str | None = None
    text_signature: str | None = None
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    partial_json: str = ""
    thought_signature: str | None = None


@dataclass
class _MutableUsage:
    """Mutable usage accumulator."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0

    def to_usage(self, cost: Cost | None = None) -> Usage:
        return Usage(
            input=self.input,
            output=self.output,
            cache_read=self.cache_read,
            cache_write=self.cache_write,
            total_tokens=self.total_tokens,
            cost=cost or Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
        )


@dataclass
class _MutableItem:
    """Tracks the current OpenAI response item being streamed."""

    type: str
    content: list[dict[str, Any]] = field(default_factory=list)
    summary: list[dict[str, Any]] = field(default_factory=list)
    call_id: str = ""
    item_id: str = ""
    name: str = ""
    arguments: str = ""
    id: str = ""


def _block_to_content(
    block: _MutableBlock,
) -> TextContent | ThinkingContent | ToolCall:
    """Convert a mutable block into its immutable content type."""
    if block.type == "text":
        return TextContent(
            text=block.text,
            text_signature=block.text_signature,
        )
    if block.type == "thinking":
        return ThinkingContent(
            thinking=block.thinking,
            thinking_signature=block.thinking_signature,
        )
    return ToolCall(
        id=block.id,
        name=block.name,
        arguments=block.arguments,
        thought_signature=block.thought_signature,
    )


def _build_partial(
    model: Model,
    blocks: list[_MutableBlock],
    usage: _MutableUsage,
    stop_reason: StopReason,
    timestamp: float,
    error_message: str | None = None,
) -> AssistantMessage:
    """Build an immutable :class:`AssistantMessage` snapshot."""
    return AssistantMessage(
        content=[_block_to_content(b) for b in blocks],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=usage.to_usage(),
        stop_reason=stop_reason,
        timestamp=timestamp,
        error_message=error_message,
    )


# ---------------------------------------------------------------------------
# Stream processing
# ---------------------------------------------------------------------------


async def _process_responses_stream(
    openai_stream: Any,
    model: Model,
    blocks: list[_MutableBlock],
    usage: _MutableUsage,
    timestamp: float,
    options: OpenAIResponsesOptions | None = None,
) -> AsyncIterator[AssistantMessageEvent]:
    """Process the raw OpenAI Responses streaming events."""
    stop_reason: StopReason = "stop"

    current_item: _MutableItem | None = None
    current_block: _MutableBlock | None = None

    def block_index() -> int:
        return len(blocks) - 1

    def partial() -> AssistantMessage:
        return _build_partial(model, blocks, usage, stop_reason, timestamp)

    async for event in openai_stream:
        event_type = getattr(event, "type", None)

        # -- Output item added --
        if event_type == "response.output_item.added":
            item = event.item
            item_type = getattr(item, "type", None)

            if item_type == "reasoning":
                current_item = _MutableItem(type="reasoning")
                current_block = _MutableBlock(type="thinking")
                blocks.append(current_block)
                yield ThinkingStartEvent(
                    content_index=block_index(),
                    partial=partial(),
                )

            elif item_type == "message":
                current_item = _MutableItem(type="message")
                current_block = _MutableBlock(type="text")
                blocks.append(current_block)
                yield TextStartEvent(
                    content_index=block_index(),
                    partial=partial(),
                )

            elif item_type == "function_call":
                call_id = getattr(item, "call_id", "") or ""
                item_id = getattr(item, "id", "") or ""
                fc_name = getattr(item, "name", "") or ""
                fc_args = getattr(item, "arguments", "") or ""
                current_item = _MutableItem(
                    type="function_call",
                    call_id=call_id,
                    item_id=item_id,
                    name=fc_name,
                    arguments=fc_args,
                )
                current_block = _MutableBlock(
                    type="toolCall",
                    id=f"{call_id}|{item_id}",
                    name=fc_name,
                    partial_json=fc_args,
                )
                blocks.append(current_block)
                yield ToolCallStartEvent(
                    content_index=block_index(),
                    partial=partial(),
                )

        # -- Reasoning summary part added --
        elif event_type == "response.reasoning_summary_part.added":
            if current_item and current_item.type == "reasoning":
                part = getattr(event, "part", None)
                if part:
                    part_dict = {
                        "type": getattr(part, "type", "summary_text"),
                        "text": getattr(part, "text", ""),
                    }
                    current_item.summary.append(part_dict)

        # -- Reasoning summary text delta --
        elif event_type == "response.reasoning_summary_text.delta":
            if (
                current_item
                and current_item.type == "reasoning"
                and current_block
                and current_block.type == "thinking"
            ):
                delta = getattr(event, "delta", "")
                if current_item.summary:
                    last_part = current_item.summary[-1]
                    current_block.thinking += delta
                    last_part["text"] = last_part.get("text", "") + delta
                    yield ThinkingDeltaEvent(
                        content_index=block_index(),
                        delta=delta,
                        partial=partial(),
                    )

        # -- Reasoning summary part done --
        elif event_type == "response.reasoning_summary_part.done":
            if (
                current_item
                and current_item.type == "reasoning"
                and current_block
                and current_block.type == "thinking"
            ):
                if current_item.summary:
                    last_part = current_item.summary[-1]
                    current_block.thinking += "\n\n"
                    last_part["text"] = last_part.get("text", "") + "\n\n"
                    yield ThinkingDeltaEvent(
                        content_index=block_index(),
                        delta="\n\n",
                        partial=partial(),
                    )

        # -- Content part added --
        elif event_type == "response.content_part.added":
            if current_item and current_item.type == "message":
                part = getattr(event, "part", None)
                if part:
                    part_type = getattr(part, "type", None)
                    if part_type in ("output_text", "refusal"):
                        current_item.content.append({
                            "type": part_type,
                            "text": getattr(part, "text", ""),
                            "refusal": getattr(part, "refusal", ""),
                        })

        # -- Output text delta --
        elif event_type == "response.output_text.delta":
            if (
                current_item
                and current_item.type == "message"
                and current_block
                and current_block.type == "text"
            ):
                if not current_item.content:
                    continue
                last_part = current_item.content[-1]
                if last_part.get("type") == "output_text":
                    delta = getattr(event, "delta", "")
                    current_block.text += delta
                    last_part["text"] = last_part.get("text", "") + delta
                    yield TextDeltaEvent(
                        content_index=block_index(),
                        delta=delta,
                        partial=partial(),
                    )

        # -- Refusal delta --
        elif event_type == "response.refusal.delta":
            if (
                current_item
                and current_item.type == "message"
                and current_block
                and current_block.type == "text"
            ):
                if not current_item.content:
                    continue
                last_part = current_item.content[-1]
                if last_part.get("type") == "refusal":
                    delta = getattr(event, "delta", "")
                    current_block.text += delta
                    last_part["refusal"] = last_part.get("refusal", "") + delta
                    yield TextDeltaEvent(
                        content_index=block_index(),
                        delta=delta,
                        partial=partial(),
                    )

        # -- Function call arguments delta --
        elif event_type == "response.function_call_arguments.delta":
            if (
                current_item
                and current_item.type == "function_call"
                and current_block
                and current_block.type == "toolCall"
            ):
                delta = getattr(event, "delta", "")
                current_block.partial_json += delta
                current_block.arguments = _parse_streaming_json(current_block.partial_json)
                yield ToolCallDeltaEvent(
                    content_index=block_index(),
                    delta=delta,
                    partial=partial(),
                )

        # -- Function call arguments done --
        elif event_type == "response.function_call_arguments.done":
            if (
                current_item
                and current_item.type == "function_call"
                and current_block
                and current_block.type == "toolCall"
            ):
                args_str = getattr(event, "arguments", "{}")
                current_block.partial_json = args_str
                current_block.arguments = _parse_streaming_json(args_str)

        # -- Output item done --
        elif event_type == "response.output_item.done":
            item = event.item
            item_type = getattr(item, "type", None)

            if item_type == "reasoning" and current_block and current_block.type == "thinking":
                summary = getattr(item, "summary", None) or []
                summary_texts = []
                for s in summary:
                    txt = getattr(s, "text", "") if not isinstance(s, dict) else s.get("text", "")
                    summary_texts.append(txt)
                current_block.thinking = "\n\n".join(summary_texts) if summary_texts else ""
                # Serialize the full item as thinking signature
                try:
                    item_dict = item.model_dump() if hasattr(item, "model_dump") else {"type": "reasoning"}
                    current_block.thinking_signature = json.dumps(item_dict)
                except (TypeError, ValueError):
                    current_block.thinking_signature = json.dumps({"type": "reasoning"})
                yield ThinkingEndEvent(
                    content_index=block_index(),
                    content=current_block.thinking,
                    partial=partial(),
                )
                current_block = None

            elif item_type == "message" and current_block and current_block.type == "text":
                content_list = getattr(item, "content", []) or []
                text_parts = []
                for c in content_list:
                    c_type = getattr(c, "type", None) if not isinstance(c, dict) else c.get("type")
                    if c_type == "output_text":
                        text_parts.append(
                            getattr(c, "text", "") if not isinstance(c, dict) else c.get("text", "")
                        )
                    elif c_type == "refusal":
                        text_parts.append(
                            getattr(c, "refusal", "") if not isinstance(c, dict) else c.get("refusal", "")
                        )
                current_block.text = "".join(text_parts)
                item_id = getattr(item, "id", None)
                if item_id:
                    current_block.text_signature = item_id
                yield TextEndEvent(
                    content_index=block_index(),
                    content=current_block.text,
                    partial=partial(),
                )
                current_block = None

            elif item_type == "function_call":
                call_id = getattr(item, "call_id", "") or ""
                item_id = getattr(item, "id", "") or ""
                fc_name = getattr(item, "name", "") or ""
                fc_args_str = getattr(item, "arguments", "{}") or "{}"

                if current_block and current_block.type == "toolCall" and current_block.partial_json:
                    args = _parse_streaming_json(current_block.partial_json)
                else:
                    args = _parse_streaming_json(fc_args_str)

                tool_call = ToolCall(
                    id=f"{call_id}|{item_id}",
                    name=fc_name,
                    arguments=args,
                )
                # Update the block so the final message is correct
                if current_block and current_block.type == "toolCall":
                    current_block.arguments = args

                current_block = None
                yield ToolCallEndEvent(
                    content_index=block_index(),
                    tool_call=tool_call,
                    partial=partial(),
                )

        # -- Response completed --
        elif event_type == "response.completed":
            response = getattr(event, "response", None)
            if response:
                resp_usage = getattr(response, "usage", None)
                if resp_usage:
                    cached_tokens = 0
                    input_details = getattr(resp_usage, "input_tokens_details", None)
                    if input_details:
                        cached_tokens = getattr(input_details, "cached_tokens", 0) or 0

                    usage.input = (getattr(resp_usage, "input_tokens", 0) or 0) - cached_tokens
                    usage.output = getattr(resp_usage, "output_tokens", 0) or 0
                    usage.cache_read = cached_tokens
                    usage.cache_write = 0
                    usage.total_tokens = getattr(resp_usage, "total_tokens", 0) or 0

                resp_status = getattr(response, "status", None)
                stop_reason = _map_stop_reason(resp_status)

                # If there are tool calls and stop reason is "stop", override to "toolUse"
                has_tool_calls = any(b.type == "toolCall" for b in blocks)
                if has_tool_calls and stop_reason == "stop":
                    stop_reason = "toolUse"

                # Apply service tier pricing
                service_tier = options.service_tier if options else None
                resp_service_tier = getattr(response, "service_tier", None)
                effective_tier = resp_service_tier or service_tier

                # Calculate cost first
                final_usage = usage.to_usage()
                cost = calculate_cost(model, final_usage)

                # Apply service tier multiplier
                multiplier = _get_service_tier_multiplier(effective_tier)
                if multiplier != 1.0:
                    cost = Cost(
                        input=cost.input * multiplier,
                        output=cost.output * multiplier,
                        cache_read=cost.cache_read * multiplier,
                        cache_write=cost.cache_write * multiplier,
                        total=(
                            cost.input * multiplier
                            + cost.output * multiplier
                            + cost.cache_read * multiplier
                            + cost.cache_write * multiplier
                        ),
                    )

        # -- Error --
        elif event_type == "error":
            code = getattr(event, "code", "unknown")
            message = getattr(event, "message", "Unknown error")
            raise RuntimeError(f"Error Code {code}: {message}")

        # -- Response failed --
        elif event_type == "response.failed":
            raise RuntimeError("Unknown error")


# ---------------------------------------------------------------------------
# Top-level streaming implementation
# ---------------------------------------------------------------------------


async def _stream_responses(
    model: Model,
    context: Context,
    options: OpenAIResponsesOptions | None = None,
) -> AsyncIterator[AssistantMessageEvent]:
    """Core streaming generator for the Responses API."""
    blocks: list[_MutableBlock] = []
    usage = _MutableUsage()
    stop_reason: StopReason = "stop"
    timestamp = time.time()

    def partial() -> AssistantMessage:
        return _build_partial(model, blocks, usage, stop_reason, timestamp)

    try:
        api_key = (options.api_key if options else None) or get_env_api_key(model.provider) or ""
        extra_headers = options.headers if options else None
        client = _create_client(model, api_key, extra_headers)
        converted_messages = _convert_responses_messages(model, context)
        params = _build_params(model, context, converted_messages, options)

        openai_stream = await client.responses.create(**params)

        yield StartEvent(partial=partial())

        async for event in _process_responses_stream(
            openai_stream, model, blocks, usage, timestamp, options,
        ):
            yield event

        # Build final message with cost
        final_usage_raw = usage.to_usage()
        cost = calculate_cost(model, final_usage_raw)

        # Apply service tier pricing
        effective_tier = options.service_tier if options else None
        multiplier = _get_service_tier_multiplier(effective_tier)
        if multiplier != 1.0:
            cost = Cost(
                input=cost.input * multiplier,
                output=cost.output * multiplier,
                cache_read=cost.cache_read * multiplier,
                cache_write=cost.cache_write * multiplier,
                total=(
                    cost.input * multiplier
                    + cost.output * multiplier
                    + cost.cache_read * multiplier
                    + cost.cache_write * multiplier
                ),
            )

        final_usage_with_cost = Usage(
            input=usage.input,
            output=usage.output,
            cache_read=usage.cache_read,
            cache_write=usage.cache_write,
            total_tokens=usage.total_tokens,
            cost=cost,
        )

        # Determine final stop reason (may have been set in response.completed handler)
        has_tool_calls = any(b.type == "toolCall" for b in blocks)
        final_stop = stop_reason
        if has_tool_calls and final_stop == "stop":
            final_stop = "toolUse"

        final_msg = AssistantMessage(
            content=[_block_to_content(b) for b in blocks],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=final_usage_with_cost,
            stop_reason=final_stop,
            timestamp=timestamp,
        )

        if final_stop in ("aborted", "error"):
            raise RuntimeError("An unknown error occurred")

        yield DoneEvent(reason=final_stop, message=final_msg)

    except Exception as exc:
        error_message = str(exc)
        error_msg = AssistantMessage(
            content=[_block_to_content(b) for b in blocks],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=usage.to_usage(),
            stop_reason="error",
            timestamp=timestamp,
            error_message=error_message,
        )
        yield ErrorEvent(reason="error", error=error_msg)


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class OpenAIResponsesProvider:
    """LLM provider for the OpenAI Responses API.

    Satisfies the :class:`~pi_mono.ai.providers.base.LLMProvider` protocol.
    """

    @property
    def api(self) -> str:
        return "openai-responses"

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """Stream with full :class:`OpenAIResponsesOptions`."""
        responses_opts: OpenAIResponsesOptions | None = None
        if options is not None:
            responses_opts = OpenAIResponsesOptions(
                temperature=options.temperature,
                max_tokens=options.max_tokens,
                abort_event=options.abort_event,
                api_key=options.api_key,
                transport=options.transport,
                cache_retention=options.cache_retention,
                session_id=options.session_id,
                headers=options.headers,
                max_retry_delay_ms=options.max_retry_delay_ms,
                metadata=options.metadata,
            )
        async for event in _stream_responses(model, context, responses_opts):
            yield event

    async def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """Stream with simplified options including reasoning level."""
        api_key = (options.api_key if options else None) or get_env_api_key(model.provider)
        if not api_key:
            raise ValueError(f"No API key for provider: {model.provider}")

        reasoning_effort: ThinkingLevel | None = None
        if options and options.reasoning:
            reasoning_effort = (
                options.reasoning
                if supports_xhigh(model)
                else _clamp_reasoning(options.reasoning)
            )

        responses_opts = OpenAIResponsesOptions(
            temperature=options.temperature if options else None,
            max_tokens=(
                (options.max_tokens if options and options.max_tokens else None)
                or min(model.max_tokens, 32000)
            ),
            abort_event=options.abort_event if options else None,
            api_key=api_key,
            transport=options.transport if options else None,
            cache_retention=options.cache_retention if options else None,
            session_id=options.session_id if options else None,
            headers=options.headers if options else None,
            max_retry_delay_ms=options.max_retry_delay_ms if options else None,
            metadata=options.metadata if options else None,
            reasoning_effort=reasoning_effort,
        )

        async for event in _stream_responses(model, context, responses_opts):
            yield event
