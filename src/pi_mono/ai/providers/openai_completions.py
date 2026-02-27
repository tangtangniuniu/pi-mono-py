"""OpenAI Chat Completions provider.

Implements :class:`~pi_mono.ai.providers.base.LLMProvider` using the
OpenAI Chat Completions API (``/v1/chat/completions``).  Handles
streaming, tool calls, thinking/reasoning content, and images.
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
    MaxTokensField,
    Model,
    OpenAICompletionsCompat,
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
    ThinkingFormat,
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
# Provider-specific options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenAICompletionsOptions(StreamOptions):
    """Extended stream options for the Chat Completions API."""

    tool_choice: str | dict[str, Any] | None = None
    reasoning_effort: ThinkingLevel | None = None


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
        # Attempt recovery: close open braces/brackets
        cleaned = partial
        open_braces = cleaned.count("{") - cleaned.count("}")
        open_brackets = cleaned.count("[") - cleaned.count("]")
        cleaned += "]" * max(open_brackets, 0)
        cleaned += "}" * max(open_braces, 0)
        try:
            return json.loads(cleaned)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            return {}


def _normalize_mistral_tool_id(tool_id: str) -> str:
    """Normalize a tool call ID for Mistral.

    Mistral requires tool IDs to be exactly 9 alphanumeric characters.
    """
    normalized = "".join(c for c in tool_id if c.isalnum())
    padding = "ABCDEFGHI"
    if len(normalized) < 9:
        normalized += padding[: 9 - len(normalized)]
    elif len(normalized) > 9:
        normalized = normalized[:9]
    return normalized


def _has_tool_history(messages: list[Any]) -> bool:
    """Check whether the conversation contains tool calls or tool results."""
    for msg in messages:
        if isinstance(msg, ToolResultMessage):
            return True
        if isinstance(msg, AssistantMessage):
            if any(
                isinstance(b, ToolCall) or (hasattr(b, "type") and b.type == "toolCall")
                for b in msg.content
            ):
                return True
    return False


def _map_stop_reason(reason: str | None) -> StopReason:
    """Map an OpenAI finish_reason string to our StopReason literal."""
    if reason is None:
        return "stop"
    mapping: dict[str, StopReason] = {
        "stop": "stop",
        "length": "length",
        "function_call": "toolUse",
        "tool_calls": "toolUse",
        "content_filter": "error",
    }
    return mapping.get(reason, "stop")


def _clamp_reasoning(
    effort: ThinkingLevel | None,
) -> ThinkingLevel | None:
    """Clamp ``xhigh`` down to ``high`` for models that don't support it."""
    if effort == "xhigh":
        return "high"
    return effort


# ---------------------------------------------------------------------------
# Compatibility detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResolvedCompat:
    """Fully resolved compatibility settings for the Chat Completions API."""

    supports_store: bool
    supports_developer_role: bool
    supports_reasoning_effort: bool
    supports_usage_in_streaming: bool
    max_tokens_field: MaxTokensField
    requires_tool_result_name: bool
    requires_assistant_after_tool_result: bool
    requires_thinking_as_text: bool
    requires_mistral_tool_ids: bool
    thinking_format: ThinkingFormat
    supports_strict_mode: bool


def _detect_compat(model: Model) -> _ResolvedCompat:
    provider = model.provider
    base_url = model.base_url

    is_zai = provider == "zai" or "api.z.ai" in base_url
    is_non_standard = any([
        provider == "cerebras",
        "cerebras.ai" in base_url,
        provider == "xai",
        "api.x.ai" in base_url,
        provider == "mistral",
        "mistral.ai" in base_url,
        "chutes.ai" in base_url,
        "deepseek.com" in base_url,
        is_zai,
        provider == "opencode",
        "opencode.ai" in base_url,
    ])
    use_max_tokens = (
        provider == "mistral"
        or "mistral.ai" in base_url
        or "chutes.ai" in base_url
    )
    is_grok = provider == "xai" or "api.x.ai" in base_url
    is_mistral = provider == "mistral" or "mistral.ai" in base_url

    return _ResolvedCompat(
        supports_store=not is_non_standard,
        supports_developer_role=not is_non_standard,
        supports_reasoning_effort=not is_grok and not is_zai,
        supports_usage_in_streaming=True,
        max_tokens_field="max_tokens" if use_max_tokens else "max_completion_tokens",
        requires_tool_result_name=is_mistral,
        requires_assistant_after_tool_result=False,
        requires_thinking_as_text=is_mistral,
        requires_mistral_tool_ids=is_mistral,
        thinking_format="zai" if is_zai else "openai",
        supports_strict_mode=True,
    )


def _get_compat(model: Model) -> _ResolvedCompat:
    detected = _detect_compat(model)
    compat = model.compat
    if compat is None or not isinstance(compat, OpenAICompletionsCompat):
        return detected

    return _ResolvedCompat(
        supports_store=(
            compat.supports_store
            if compat.supports_store is not None
            else detected.supports_store
        ),
        supports_developer_role=(
            compat.supports_developer_role
            if compat.supports_developer_role is not None
            else detected.supports_developer_role
        ),
        supports_reasoning_effort=(
            compat.supports_reasoning_effort
            if compat.supports_reasoning_effort is not None
            else detected.supports_reasoning_effort
        ),
        supports_usage_in_streaming=(
            compat.supports_usage_in_streaming
            if compat.supports_usage_in_streaming is not None
            else detected.supports_usage_in_streaming
        ),
        max_tokens_field=(
            compat.max_tokens_field
            if compat.max_tokens_field is not None
            else detected.max_tokens_field
        ),
        requires_tool_result_name=(
            compat.requires_tool_result_name
            if compat.requires_tool_result_name is not None
            else detected.requires_tool_result_name
        ),
        requires_assistant_after_tool_result=(
            compat.requires_assistant_after_tool_result
            if compat.requires_assistant_after_tool_result is not None
            else detected.requires_assistant_after_tool_result
        ),
        requires_thinking_as_text=(
            compat.requires_thinking_as_text
            if compat.requires_thinking_as_text is not None
            else detected.requires_thinking_as_text
        ),
        requires_mistral_tool_ids=(
            compat.requires_mistral_tool_ids
            if compat.requires_mistral_tool_ids is not None
            else detected.requires_mistral_tool_ids
        ),
        thinking_format=(
            compat.thinking_format
            if compat.thinking_format is not None
            else detected.thinking_format
        ),
        supports_strict_mode=(
            compat.supports_strict_mode
            if compat.supports_strict_mode is not None
            else detected.supports_strict_mode
        ),
    )


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def _convert_messages(
    model: Model,
    context: Context,
    compat: _ResolvedCompat,
) -> list[dict[str, Any]]:
    """Convert pi_mono messages to OpenAI Chat Completions message dicts."""

    def normalize_tool_call_id(tool_id: str) -> str:
        if compat.requires_mistral_tool_ids:
            return _normalize_mistral_tool_id(tool_id)
        if "|" in tool_id:
            call_id = tool_id.split("|")[0]
            import re
            sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", call_id)
            return sanitized[:40]
        if model.provider == "openai":
            return tool_id[:40] if len(tool_id) > 40 else tool_id
        return tool_id

    params: list[dict[str, Any]] = []

    if context.system_prompt:
        use_developer = model.reasoning and compat.supports_developer_role
        role = "developer" if use_developer else "system"
        params.append({"role": role, "content": context.system_prompt})

    messages = context.messages
    last_role: str | None = None
    i = 0

    while i < len(messages):
        msg = messages[i]

        # Bridge assistant message after tool result for providers that need it
        if (
            compat.requires_assistant_after_tool_result
            and last_role == "toolResult"
            and msg.role == "user"
        ):
            params.append({
                "role": "assistant",
                "content": "I have processed the tool results.",
            })

        if msg.role == "user":
            assert isinstance(msg, UserMessage)
            if isinstance(msg.content, str):
                params.append({"role": "user", "content": msg.content})
            else:
                parts: list[dict[str, Any]] = []
                for item in msg.content:
                    if isinstance(item, TextContent):
                        parts.append({"type": "text", "text": item.text})
                    elif isinstance(item, ImageContent):
                        parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{item.mime_type};base64,{item.data}",
                            },
                        })
                filtered = (
                    [p for p in parts if p.get("type") != "image_url"]
                    if "image" not in model.input
                    else parts
                )
                if not filtered:
                    i += 1
                    continue
                params.append({"role": "user", "content": filtered})
            last_role = "user"

        elif msg.role == "assistant":
            assert isinstance(msg, AssistantMessage)
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "" if compat.requires_assistant_after_tool_result else None,
            }

            # Text blocks
            text_blocks = [b for b in msg.content if isinstance(b, TextContent)]
            non_empty_text = [b for b in text_blocks if b.text and b.text.strip()]
            if non_empty_text:
                if model.provider == "github-copilot":
                    assistant_msg["content"] = "".join(b.text for b in non_empty_text)
                else:
                    assistant_msg["content"] = [
                        {"type": "text", "text": b.text} for b in non_empty_text
                    ]

            # Thinking blocks
            thinking_blocks = [b for b in msg.content if isinstance(b, ThinkingContent)]
            non_empty_thinking = [
                b for b in thinking_blocks if b.thinking and b.thinking.strip()
            ]
            if non_empty_thinking:
                if compat.requires_thinking_as_text:
                    thinking_text = "\n\n".join(b.thinking for b in non_empty_thinking)
                    content = assistant_msg.get("content")
                    if isinstance(content, list):
                        content.insert(0, {"type": "text", "text": thinking_text})
                    else:
                        assistant_msg["content"] = [{"type": "text", "text": thinking_text}]
                else:
                    sig = non_empty_thinking[0].thinking_signature
                    if sig and len(sig) > 0:
                        assistant_msg[sig] = "\n".join(
                            b.thinking for b in non_empty_thinking
                        )

            # Tool calls
            tool_calls = [b for b in msg.content if isinstance(b, ToolCall)]
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": normalize_tool_call_id(tc.id),
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ]
                reasoning_details = []
                for tc in tool_calls:
                    if tc.thought_signature:
                        try:
                            reasoning_details.append(json.loads(tc.thought_signature))
                        except json.JSONDecodeError:
                            pass
                if reasoning_details:
                    assistant_msg["reasoning_details"] = reasoning_details

            # Skip empty assistant messages
            content_val = assistant_msg.get("content")
            has_content = content_val is not None and (
                (isinstance(content_val, str) and len(content_val) > 0)
                or (isinstance(content_val, list) and len(content_val) > 0)
            )
            if not has_content and "tool_calls" not in assistant_msg:
                i += 1
                continue

            params.append(assistant_msg)
            last_role = "assistant"

        elif msg.role == "toolResult":
            assert isinstance(msg, ToolResultMessage)
            image_blocks: list[dict[str, Any]] = []
            j = i

            while j < len(messages) and messages[j].role == "toolResult":
                tool_msg = messages[j]
                assert isinstance(tool_msg, ToolResultMessage)

                text_parts = [
                    c.text
                    for c in tool_msg.content
                    if isinstance(c, TextContent)
                ]
                text_result = "\n".join(text_parts)
                has_images = any(
                    isinstance(c, ImageContent) for c in tool_msg.content
                )

                has_text = len(text_result) > 0
                tool_result_msg: dict[str, Any] = {
                    "role": "tool",
                    "content": text_result if has_text else "(see attached image)",
                    "tool_call_id": normalize_tool_call_id(tool_msg.tool_call_id),
                }
                if compat.requires_tool_result_name and tool_msg.tool_name:
                    tool_result_msg["name"] = tool_msg.tool_name
                params.append(tool_result_msg)

                if has_images and "image" in model.input:
                    for block in tool_msg.content:
                        if isinstance(block, ImageContent):
                            image_blocks.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{block.mime_type};base64,{block.data}",
                                },
                            })

                j += 1

            i = j

            if image_blocks:
                if compat.requires_assistant_after_tool_result:
                    params.append({
                        "role": "assistant",
                        "content": "I have processed the tool results.",
                    })
                params.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Attached image(s) from tool result:"},
                        *image_blocks,
                    ],
                })
                last_role = "user"
            else:
                last_role = "toolResult"
            continue

        i += 1

    return params


def _convert_tools(
    tools: list[Tool],
    compat: _ResolvedCompat,
) -> list[dict[str, Any]]:
    """Convert pi_mono Tool definitions to OpenAI function tool dicts."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        func: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        if compat.supports_strict_mode is not False:
            func["strict"] = False
        result.append({"type": "function", "function": func})
    return result


# ---------------------------------------------------------------------------
# Client / params builders
# ---------------------------------------------------------------------------


def _create_client(
    model: Model,
    api_key: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> AsyncOpenAI:
    """Create an :class:`AsyncOpenAI` client configured for *model*."""
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
    options: OpenAICompletionsOptions | None,
) -> dict[str, Any]:
    """Build the ``chat.completions.create`` request payload."""
    compat = _get_compat(model)
    messages = _convert_messages(model, context, compat)

    params: dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "stream": True,
    }

    if compat.supports_usage_in_streaming:
        params["stream_options"] = {"include_usage": True}

    if compat.supports_store:
        params["store"] = False

    if options and options.max_tokens:
        if compat.max_tokens_field == "max_tokens":
            params["max_tokens"] = options.max_tokens
        else:
            params["max_completion_tokens"] = options.max_tokens

    if options and options.temperature is not None:
        params["temperature"] = options.temperature

    if context.tools:
        params["tools"] = _convert_tools(context.tools, compat)
    elif _has_tool_history(context.messages):
        params["tools"] = []

    if options and options.tool_choice is not None:
        params["tool_choice"] = options.tool_choice

    reasoning_effort = options.reasoning_effort if options else None
    if compat.thinking_format == "zai" and model.reasoning:
        params["thinking"] = {
            "type": "enabled" if reasoning_effort else "disabled",
        }
    elif compat.thinking_format == "qwen" and model.reasoning:
        params["enable_thinking"] = bool(reasoning_effort)
    elif reasoning_effort and model.reasoning and compat.supports_reasoning_effort:
        params["reasoning_effort"] = reasoning_effort

    return params


# ---------------------------------------------------------------------------
# Mutable streaming accumulator
# ---------------------------------------------------------------------------


@dataclass
class _MutableBlock:
    """Mutable wrapper used while accumulating streamed content blocks."""

    type: str
    text: str = ""
    thinking: str = ""
    thinking_signature: str | None = None
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    partial_args: str = ""
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


def _block_to_content(
    block: _MutableBlock,
) -> TextContent | ThinkingContent | ToolCall:
    """Convert a mutable block into its immutable content type."""
    if block.type == "text":
        return TextContent(text=block.text)
    if block.type == "thinking":
        return ThinkingContent(
            thinking=block.thinking,
            thinking_signature=block.thinking_signature,
        )
    # toolCall
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
# Streaming implementation
# ---------------------------------------------------------------------------


async def _stream_completions(
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions | None = None,
) -> AsyncIterator[AssistantMessageEvent]:
    """Core streaming generator for the Chat Completions API."""
    blocks: list[_MutableBlock] = []
    usage = _MutableUsage()
    stop_reason: StopReason = "stop"
    timestamp = time.time()
    error_message: str | None = None

    def partial() -> AssistantMessage:
        return _build_partial(model, blocks, usage, stop_reason, timestamp, error_message)

    def block_index() -> int:
        return len(blocks) - 1

    try:
        api_key = (options.api_key if options else None) or get_env_api_key(model.provider) or ""
        extra_headers = options.headers if options else None
        client = _create_client(model, api_key, extra_headers)
        params = _build_params(model, context, options)

        openai_stream = await client.chat.completions.create(**params)

        yield StartEvent(partial=partial())

        current_block: _MutableBlock | None = None

        def _finish_current_block(block: _MutableBlock | None) -> list[AssistantMessageEvent]:
            """Finish the current block and return events to yield."""
            events: list[AssistantMessageEvent] = []
            if block is None:
                return events
            if block.type == "text":
                events.append(TextEndEvent(
                    content_index=block_index(),
                    content=block.text,
                    partial=partial(),
                ))
            elif block.type == "thinking":
                events.append(ThinkingEndEvent(
                    content_index=block_index(),
                    content=block.thinking,
                    partial=partial(),
                ))
            elif block.type == "toolCall":
                block.arguments = _parse_streaming_json(block.partial_args)
                events.append(ToolCallEndEvent(
                    content_index=block_index(),
                    tool_call=ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.arguments,
                        thought_signature=block.thought_signature,
                    ),
                    partial=partial(),
                ))
            return events

        async for chunk in openai_stream:
            # Process usage from the chunk
            if hasattr(chunk, "usage") and chunk.usage is not None:
                cached_tokens = 0
                if hasattr(chunk.usage, "prompt_tokens_details") and chunk.usage.prompt_tokens_details:
                    cached_tokens = getattr(chunk.usage.prompt_tokens_details, "cached_tokens", 0) or 0
                reasoning_tokens = 0
                if hasattr(chunk.usage, "completion_tokens_details") and chunk.usage.completion_tokens_details:
                    reasoning_tokens = getattr(chunk.usage.completion_tokens_details, "reasoning_tokens", 0) or 0

                input_tokens = (chunk.usage.prompt_tokens or 0) - cached_tokens
                output_tokens = (chunk.usage.completion_tokens or 0) + reasoning_tokens

                usage.input = input_tokens
                usage.output = output_tokens
                usage.cache_read = cached_tokens
                usage.cache_write = 0
                usage.total_tokens = input_tokens + output_tokens + cached_tokens

            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            choice = choices[0]

            if choice.finish_reason:
                stop_reason = _map_stop_reason(choice.finish_reason)

            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            # -- Text content --
            delta_content = getattr(delta, "content", None)
            if delta_content is not None and len(delta_content) > 0:
                if current_block is None or current_block.type != "text":
                    for evt in _finish_current_block(current_block):
                        yield evt
                    current_block = _MutableBlock(type="text")
                    blocks.append(current_block)
                    yield TextStartEvent(
                        content_index=block_index(),
                        partial=partial(),
                    )
                current_block.text += delta_content
                yield TextDeltaEvent(
                    content_index=block_index(),
                    delta=delta_content,
                    partial=partial(),
                )

            # -- Reasoning / thinking content --
            reasoning_fields = ["reasoning_content", "reasoning", "reasoning_text"]
            found_field: str | None = None
            for rf in reasoning_fields:
                val = getattr(delta, rf, None)
                if val is not None and len(val) > 0:
                    found_field = rf
                    break

            if found_field:
                reasoning_val = getattr(delta, found_field)
                if current_block is None or current_block.type != "thinking":
                    for evt in _finish_current_block(current_block):
                        yield evt
                    current_block = _MutableBlock(
                        type="thinking",
                        thinking_signature=found_field,
                    )
                    blocks.append(current_block)
                    yield ThinkingStartEvent(
                        content_index=block_index(),
                        partial=partial(),
                    )
                current_block.thinking += reasoning_val
                yield ThinkingDeltaEvent(
                    content_index=block_index(),
                    delta=reasoning_val,
                    partial=partial(),
                )

            # -- Tool calls --
            delta_tool_calls = getattr(delta, "tool_calls", None)
            if delta_tool_calls:
                for tc in delta_tool_calls:
                    tc_id = getattr(tc, "id", None)
                    tc_func = getattr(tc, "function", None)
                    tc_name = tc_func.name if tc_func and tc_func.name else None
                    tc_args = tc_func.arguments if tc_func and tc_func.arguments else None

                    if (
                        current_block is None
                        or current_block.type != "toolCall"
                        or (tc_id and current_block.id != tc_id)
                    ):
                        for evt in _finish_current_block(current_block):
                            yield evt
                        current_block = _MutableBlock(
                            type="toolCall",
                            id=tc_id or "",
                            name=tc_name or "",
                        )
                        blocks.append(current_block)
                        yield ToolCallStartEvent(
                            content_index=block_index(),
                            partial=partial(),
                        )

                    if tc_id:
                        current_block.id = tc_id
                    if tc_name:
                        current_block.name = tc_name

                    arg_delta = ""
                    if tc_args:
                        arg_delta = tc_args
                        current_block.partial_args += tc_args
                        current_block.arguments = _parse_streaming_json(current_block.partial_args)

                    yield ToolCallDeltaEvent(
                        content_index=block_index(),
                        delta=arg_delta,
                        partial=partial(),
                    )

            # -- Encrypted reasoning details (for tool calls) --
            reasoning_details = getattr(delta, "reasoning_details", None)
            if reasoning_details and isinstance(reasoning_details, list):
                for detail in reasoning_details:
                    detail_type = getattr(detail, "type", None) if not isinstance(detail, dict) else detail.get("type")
                    detail_id = getattr(detail, "id", None) if not isinstance(detail, dict) else detail.get("id")
                    detail_data = getattr(detail, "data", None) if not isinstance(detail, dict) else detail.get("data")
                    if detail_type == "reasoning.encrypted" and detail_id and detail_data:
                        for b in blocks:
                            if b.type == "toolCall" and b.id == detail_id:
                                raw = detail if isinstance(detail, dict) else {
                                    "type": detail_type,
                                    "id": detail_id,
                                    "data": detail_data,
                                }
                                b.thought_signature = json.dumps(raw)

        # Finish the last block
        for evt in _finish_current_block(current_block):
            yield evt

        # Calculate cost
        final_usage = usage.to_usage()
        cost = calculate_cost(model, final_usage)
        final_usage_with_cost = Usage(
            input=usage.input,
            output=usage.output,
            cache_read=usage.cache_read,
            cache_write=usage.cache_write,
            total_tokens=usage.total_tokens,
            cost=cost,
        )

        final_msg = AssistantMessage(
            content=[_block_to_content(b) for b in blocks],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=final_usage_with_cost,
            stop_reason=stop_reason,
            timestamp=timestamp,
        )

        if stop_reason in ("aborted", "error"):
            raise RuntimeError("An unknown error occurred")

        yield DoneEvent(reason=stop_reason, message=final_msg)

    except Exception as exc:
        error_message = str(exc)
        err_stop: StopReason = "error"
        error_msg = AssistantMessage(
            content=[_block_to_content(b) for b in blocks],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=usage.to_usage(),
            stop_reason=err_stop,
            timestamp=timestamp,
            error_message=error_message,
        )
        yield ErrorEvent(reason=err_stop, error=error_msg)


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class OpenAICompletionsProvider:
    """LLM provider for the OpenAI Chat Completions API.

    Satisfies the :class:`~pi_mono.ai.providers.base.LLMProvider` protocol.
    """

    @property
    def api(self) -> str:
        return "openai-completions"

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """Stream with full :class:`OpenAICompletionsOptions`."""
        completions_opts: OpenAICompletionsOptions | None = None
        if options is not None:
            completions_opts = OpenAICompletionsOptions(
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
        async for event in _stream_completions(model, context, completions_opts):
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

        completions_opts = OpenAICompletionsOptions(
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

        async for event in _stream_completions(model, context, completions_opts):
            yield event
