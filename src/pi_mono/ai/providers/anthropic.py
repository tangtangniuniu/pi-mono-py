"""Anthropic Messages API provider.

Translates pi_mono types into Anthropic SDK calls and streams
:class:`AssistantMessageEvent` objects back to the caller.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator, Literal

import anthropic

from pi_mono.ai.env_api_keys import get_env_api_key
from pi_mono.ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    Cost,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Message,
    Model,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingBudgets,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ThinkingLevel,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    Usage,
)

# ---------------------------------------------------------------------------
# Effort / adaptive-thinking helpers
# ---------------------------------------------------------------------------

AnthropicEffort = Literal["low", "medium", "high", "max"]


@dataclass(frozen=True)
class _AnthropicOptions:
    """Internal options specific to the Anthropic streaming call."""

    thinking_enabled: bool = False
    thinking_budget_tokens: int | None = None
    effort: AnthropicEffort | None = None
    interleaved_thinking: bool = True
    tool_choice: str | dict[str, Any] | None = None
    # Base stream options forwarded through
    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    cache_retention: str | None = None
    headers: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None
    abort_event: Any | None = None


# ---------------------------------------------------------------------------
# Cache-control helpers
# ---------------------------------------------------------------------------

CacheRetentionLiteral = Literal["none", "short", "long"]


def _resolve_cache_retention(
    cache_retention: str | None,
) -> CacheRetentionLiteral:
    if cache_retention is not None:
        return cache_retention  # type: ignore[return-value]
    env_val = os.environ.get("PI_CACHE_RETENTION")
    if env_val == "long":
        return "long"
    return "short"


def _get_cache_control(
    base_url: str,
    cache_retention: str | None = None,
) -> dict[str, Any] | None:
    retention = _resolve_cache_retention(cache_retention)
    if retention == "none":
        return None
    ttl = "1h" if retention == "long" and "api.anthropic.com" in base_url else None
    cc: dict[str, Any] = {"type": "ephemeral"}
    if ttl is not None:
        cc["ttl"] = ttl
    return cc


# ---------------------------------------------------------------------------
# Adaptive thinking helpers
# ---------------------------------------------------------------------------


def _supports_adaptive_thinking(model_id: str) -> bool:
    return any(
        tag in model_id
        for tag in ("opus-4-6", "opus-4.6", "sonnet-4-6", "sonnet-4.6")
    )


def _map_thinking_level_to_effort(
    level: ThinkingLevel | None,
    model_id: str,
) -> AnthropicEffort:
    mapping: dict[str, AnthropicEffort] = {
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
    }
    if level == "xhigh":
        if "opus-4-6" in model_id or "opus-4.6" in model_id:
            return "max"
        return "high"
    return mapping.get(level or "high", "high")


def _clamp_reasoning(
    level: ThinkingLevel | None,
) -> ThinkingLevel | None:
    if level == "xhigh":
        return "high"
    return level


def _adjust_max_tokens_for_thinking(
    base_max_tokens: int,
    model_max_tokens: int,
    reasoning_level: ThinkingLevel,
    custom_budgets: ThinkingBudgets | None = None,
) -> tuple[int, int]:
    """Return ``(max_tokens, thinking_budget)``."""
    default_budgets: dict[str, int] = {
        "minimal": 1024,
        "low": 2048,
        "medium": 8192,
        "high": 16384,
    }
    if custom_budgets is not None:
        for key in ("minimal", "low", "medium", "high"):
            val = getattr(custom_budgets, key, None)
            if val is not None:
                default_budgets[key] = val

    clamped = _clamp_reasoning(reasoning_level) or "high"
    thinking_budget = default_budgets.get(clamped, 16384)
    max_tokens = min(base_max_tokens + thinking_budget, model_max_tokens)

    min_output_tokens = 1024
    if max_tokens <= thinking_budget:
        thinking_budget = max(0, max_tokens - min_output_tokens)

    return max_tokens, thinking_budget


# ---------------------------------------------------------------------------
# Message / tool conversion
# ---------------------------------------------------------------------------


def _convert_content_blocks(
    content: list[TextContent | ImageContent],
) -> str | list[dict[str, Any]]:
    """Convert content blocks to Anthropic's message format.

    Returns a plain string when only text is present, otherwise a list
    of content-block dicts.
    """
    has_images = any(c.type == "image" for c in content)
    if not has_images:
        return "\n".join(c.text for c in content if isinstance(c, TextContent))

    blocks: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextContent):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": block.mime_type,
                    "data": block.data,
                },
            })

    if not any(b["type"] == "text" for b in blocks):
        blocks.insert(0, {"type": "text", "text": "(see attached image)"})

    return blocks


def _convert_messages(
    messages: list[Message],
    model: Model,
    cache_control: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build the ``messages`` parameter for the Anthropic API."""
    params: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.role == "user":
            if isinstance(msg.content, str):
                if msg.content.strip():
                    params.append({"role": "user", "content": msg.content})
            else:
                blocks: list[dict[str, Any]] = []
                for item in msg.content:
                    if isinstance(item, TextContent):
                        if item.text.strip():
                            blocks.append({"type": "text", "text": item.text})
                    elif isinstance(item, ImageContent):
                        if "image" in model.input:
                            blocks.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": item.mime_type,
                                    "data": item.data,
                                },
                            })
                if blocks:
                    params.append({"role": "user", "content": blocks})

        elif msg.role == "assistant":
            blocks = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    if block.text.strip():
                        blocks.append({"type": "text", "text": block.text})
                elif isinstance(block, ThinkingContent):
                    if block.thinking.strip():
                        if (
                            block.thinking_signature
                            and block.thinking_signature.strip()
                        ):
                            blocks.append({
                                "type": "thinking",
                                "thinking": block.thinking,
                                "signature": block.thinking_signature,
                            })
                        else:
                            # No valid signature -> convert to plain text
                            blocks.append({"type": "text", "text": block.thinking})
                elif isinstance(block, ToolCall):
                    blocks.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.arguments or {},
                    })
            if blocks:
                params.append({"role": "assistant", "content": blocks})

        elif msg.role == "toolResult":
            tool_results: list[dict[str, Any]] = []
            tool_msg: ToolResultMessage = msg  # type: ignore[assignment]
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_msg.tool_call_id,
                "content": _convert_content_blocks(tool_msg.content),
                "is_error": tool_msg.is_error,
            })
            # Collect consecutive tool-result messages into one user turn
            j = i + 1
            while j < len(messages) and messages[j].role == "toolResult":
                next_msg: ToolResultMessage = messages[j]  # type: ignore[assignment]
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": next_msg.tool_call_id,
                    "content": _convert_content_blocks(next_msg.content),
                    "is_error": next_msg.is_error,
                })
                j += 1
            i = j - 1
            params.append({"role": "user", "content": tool_results})

        i += 1

    # Attach cache_control to the last user message
    if cache_control and params:
        last = params[-1]
        if last["role"] == "user":
            content = last["content"]
            if isinstance(content, list) and content:
                last_block = content[-1]
                if last_block.get("type") in ("text", "image", "tool_result"):
                    last_block["cache_control"] = cache_control
            elif isinstance(content, str):
                last["content"] = [
                    {"type": "text", "text": content, "cache_control": cache_control},
                ]

    return params


def _convert_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Convert pi_mono tools to the Anthropic tool format."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        schema = tool.parameters or {}
        result.append({
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            },
        })
    return result


def _map_stop_reason(reason: str) -> StopReason:
    mapping: dict[str, StopReason] = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "toolUse",
        "refusal": "error",
        "pause_turn": "stop",
        "stop_sequence": "stop",
        "sensitive": "error",
    }
    result = mapping.get(reason)
    if result is None:
        raise ValueError(f"Unhandled stop reason: {reason}")
    return result


def _compute_cost(
    model: Model,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
) -> Cost:
    """Compute dollar cost from model pricing and raw token counts.

    Pricing in ``model.cost`` is expressed as dollars per million tokens.
    """
    input_cost = (model.cost.input / 1_000_000) * input_tokens
    output_cost = (model.cost.output / 1_000_000) * output_tokens
    cache_read_cost = (model.cost.cache_read / 1_000_000) * cache_read_tokens
    cache_write_cost = (model.cost.cache_write / 1_000_000) * cache_write_tokens
    total = input_cost + output_cost + cache_read_cost + cache_write_cost
    return Cost(
        input=input_cost,
        output=output_cost,
        cache_read=cache_read_cost,
        cache_write=cache_write_cost,
        total=total,
    )


# ---------------------------------------------------------------------------
# Streaming JSON parser (simplified)
# ---------------------------------------------------------------------------


def _parse_streaming_json(partial_json: str) -> dict[str, Any]:
    """Best-effort parse of possibly incomplete JSON from a streaming tool call.

    Returns an empty dict if the JSON cannot be parsed.
    """
    s = partial_json.strip()
    if not s:
        return {}
    try:
        return json.loads(s)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass
    # Attempt to close the JSON object
    for suffix in ("}", '"}', '"]}', "]}"):
        try:
            return json.loads(s + suffix)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            continue
    return {}


# ---------------------------------------------------------------------------
# Mutable helper used during streaming to accumulate content blocks
# ---------------------------------------------------------------------------


class _StreamingBlock:
    """Mutable accumulator for a single content block during streaming."""

    __slots__ = (
        "stream_index",
        "block_type",
        "text",
        "thinking",
        "thinking_signature",
        "tool_id",
        "tool_name",
        "tool_partial_json",
        "tool_arguments",
    )

    def __init__(self, stream_index: int, block_type: str) -> None:
        self.stream_index = stream_index
        self.block_type = block_type
        self.text = ""
        self.thinking = ""
        self.thinking_signature = ""
        self.tool_id = ""
        self.tool_name = ""
        self.tool_partial_json = ""
        self.tool_arguments: dict[str, Any] = {}

    def to_content(self) -> TextContent | ThinkingContent | ToolCall:
        if self.block_type == "text":
            return TextContent(text=self.text)
        if self.block_type == "thinking":
            return ThinkingContent(
                thinking=self.thinking,
                thinking_signature=self.thinking_signature or None,
            )
        # toolCall
        return ToolCall(
            id=self.tool_id,
            name=self.tool_name,
            arguments=self.tool_arguments,
        )


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class AnthropicProvider:
    """LLM provider for the Anthropic Messages API.

    Implements the :class:`LLMProvider` protocol with ``api``,
    ``stream``, and ``stream_simple`` methods.
    """

    @property
    def api(self) -> str:  # noqa: D401
        """API identifier for this provider."""
        return "anthropic-messages"

    # -- public entry points ------------------------------------------------

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """Stream with full provider-specific options.

        Callers should use ``async for event in provider.stream(...)``.
        """
        opts = _anthropic_options_from_stream(options)
        async for event in self._run_stream(model, context, opts):
            yield event

    async def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """Stream using the simplified reasoning-level interface."""
        opts = self._build_simple_options(model, options)
        async for event in self._run_stream(model, context, opts):
            yield event

    # -- simple-option builder ----------------------------------------------

    @staticmethod
    def _build_simple_options(
        model: Model,
        options: SimpleStreamOptions | None,
    ) -> _AnthropicOptions:
        api_key = (
            (options.api_key if options else None)
            or get_env_api_key(model.provider)
        )
        if not api_key:
            raise ValueError(f"No API key for provider: {model.provider}")

        base_max = (
            (options.max_tokens if options else None)
            or min(model.max_tokens, 32_000)
        )
        base = _AnthropicOptions(
            temperature=options.temperature if options else None,
            max_tokens=base_max,
            api_key=api_key,
            cache_retention=options.cache_retention if options else None,
            headers=options.headers if options else None,
            metadata=options.metadata if options else None,
            abort_event=options.abort_event if options else None,
        )

        reasoning = options.reasoning if options else None
        if not reasoning:
            return replace(base, thinking_enabled=False)

        if _supports_adaptive_thinking(model.id):
            effort = _map_thinking_level_to_effort(reasoning, model.id)
            return replace(base, thinking_enabled=True, effort=effort)

        max_tokens, thinking_budget = _adjust_max_tokens_for_thinking(
            base.max_tokens or 0,
            model.max_tokens,
            reasoning,
            options.thinking_budgets if options else None,
        )
        return replace(
            base,
            max_tokens=max_tokens,
            thinking_enabled=True,
            thinking_budget_tokens=thinking_budget,
        )

    # -- core streaming implementation --------------------------------------

    async def _run_stream(
        self,
        model: Model,
        context: Context,
        opts: _AnthropicOptions,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """Core streaming loop that yields :class:`AssistantMessageEvent`."""
        now = time.time()
        # We track mutable state via simple dicts/lists because the frozen
        # dataclasses are immutable.  We rebuild the AssistantMessage on each
        # event yield so the caller always sees a consistent snapshot.
        content_blocks: list[_StreamingBlock] = []
        stop_reason: StopReason = "stop"
        error_message: str | None = None

        # Mutable usage counters
        u_input = 0
        u_output = 0
        u_cache_read = 0
        u_cache_write = 0

        def _snapshot() -> AssistantMessage:
            frozen_content = [b.to_content() for b in content_blocks]
            total_tokens = u_input + u_output + u_cache_read + u_cache_write
            cost = _compute_cost(model, u_input, u_output, u_cache_read, u_cache_write)
            return AssistantMessage(
                content=frozen_content,
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=Usage(
                    input=u_input,
                    output=u_output,
                    cache_read=u_cache_read,
                    cache_write=u_cache_write,
                    total_tokens=total_tokens,
                    cost=cost,
                ),
                stop_reason=stop_reason,
                timestamp=now,
                error_message=error_message,
            )

        def _find_block(stream_index: int) -> tuple[int, _StreamingBlock | None]:
            for idx, b in enumerate(content_blocks):
                if b.stream_index == stream_index:
                    return idx, b
            return -1, None

        try:
            api_key = opts.api_key or get_env_api_key(model.provider) or ""
            client = _create_client(model, api_key, opts)
            params = _build_params(model, context, opts)

            raw_stream = client.messages.stream(**params)

            yield StartEvent(partial=_snapshot())

            async with raw_stream as stream_mgr:
                async for event in stream_mgr:
                    # -- message_start ----------------------------------
                    if event.type == "message_start":
                        msg_usage = event.message.usage
                        u_input = getattr(msg_usage, "input_tokens", 0) or 0
                        u_output = getattr(msg_usage, "output_tokens", 0) or 0
                        u_cache_read = getattr(msg_usage, "cache_read_input_tokens", 0) or 0
                        u_cache_write = getattr(msg_usage, "cache_creation_input_tokens", 0) or 0

                    # -- content_block_start ----------------------------
                    elif event.type == "content_block_start":
                        cb = event.content_block
                        if cb.type == "text":
                            blk = _StreamingBlock(event.index, "text")
                            content_blocks.append(blk)
                            yield TextStartEvent(
                                content_index=len(content_blocks) - 1,
                                partial=_snapshot(),
                            )
                        elif cb.type == "thinking":
                            blk = _StreamingBlock(event.index, "thinking")
                            content_blocks.append(blk)
                            yield ThinkingStartEvent(
                                content_index=len(content_blocks) - 1,
                                partial=_snapshot(),
                            )
                        elif cb.type == "tool_use":
                            blk = _StreamingBlock(event.index, "toolCall")
                            blk.tool_id = cb.id
                            blk.tool_name = cb.name
                            blk.tool_arguments = (
                                cb.input if isinstance(cb.input, dict) else {}
                            )
                            content_blocks.append(blk)
                            yield ToolCallStartEvent(
                                content_index=len(content_blocks) - 1,
                                partial=_snapshot(),
                            )

                    # -- content_block_delta -----------------------------
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            ci, blk = _find_block(event.index)
                            if blk is not None and blk.block_type == "text":
                                blk.text += delta.text
                                yield TextDeltaEvent(
                                    content_index=ci,
                                    delta=delta.text,
                                    partial=_snapshot(),
                                )
                        elif delta.type == "thinking_delta":
                            ci, blk = _find_block(event.index)
                            if blk is not None and blk.block_type == "thinking":
                                blk.thinking += delta.thinking
                                yield ThinkingDeltaEvent(
                                    content_index=ci,
                                    delta=delta.thinking,
                                    partial=_snapshot(),
                                )
                        elif delta.type == "input_json_delta":
                            ci, blk = _find_block(event.index)
                            if blk is not None and blk.block_type == "toolCall":
                                blk.tool_partial_json += delta.partial_json
                                blk.tool_arguments = _parse_streaming_json(
                                    blk.tool_partial_json
                                )
                                yield ToolCallDeltaEvent(
                                    content_index=ci,
                                    delta=delta.partial_json,
                                    partial=_snapshot(),
                                )
                        elif delta.type == "signature_delta":
                            _ci, blk = _find_block(event.index)
                            if blk is not None and blk.block_type == "thinking":
                                blk.thinking_signature += delta.signature

                    # -- content_block_stop ------------------------------
                    elif event.type == "content_block_stop":
                        ci, blk = _find_block(event.index)
                        if blk is not None:
                            if blk.block_type == "text":
                                yield TextEndEvent(
                                    content_index=ci,
                                    content=blk.text,
                                    partial=_snapshot(),
                                )
                            elif blk.block_type == "thinking":
                                yield ThinkingEndEvent(
                                    content_index=ci,
                                    content=blk.thinking,
                                    partial=_snapshot(),
                                )
                            elif blk.block_type == "toolCall":
                                blk.tool_arguments = _parse_streaming_json(
                                    blk.tool_partial_json
                                )
                                yield ToolCallEndEvent(
                                    content_index=ci,
                                    tool_call=blk.to_content(),  # type: ignore[arg-type]
                                    partial=_snapshot(),
                                )

                    # -- message_delta -----------------------------------
                    elif event.type == "message_delta":
                        d = event.delta
                        if hasattr(d, "stop_reason") and d.stop_reason:
                            stop_reason = _map_stop_reason(d.stop_reason)
                        eu = event.usage
                        if getattr(eu, "input_tokens", None) is not None:
                            u_input = eu.input_tokens
                        if getattr(eu, "output_tokens", None) is not None:
                            u_output = eu.output_tokens
                        if getattr(eu, "cache_read_input_tokens", None) is not None:
                            u_cache_read = eu.cache_read_input_tokens
                        if getattr(eu, "cache_creation_input_tokens", None) is not None:
                            u_cache_write = eu.cache_creation_input_tokens

            # Check abort
            if opts.abort_event is not None and opts.abort_event.is_set():
                raise RuntimeError("Request was aborted")

            if stop_reason in ("aborted", "error"):
                raise RuntimeError("An unknown error occurred")

            yield DoneEvent(reason=stop_reason, message=_snapshot())  # type: ignore[arg-type]

        except Exception as exc:
            is_aborted = (
                opts.abort_event is not None and opts.abort_event.is_set()
            )
            stop_reason = "aborted" if is_aborted else "error"
            error_message = str(exc)
            yield ErrorEvent(reason=stop_reason, error=_snapshot())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Client / param builders (module-private)
# ---------------------------------------------------------------------------


def _create_client(
    model: Model,
    api_key: str,
    opts: _AnthropicOptions,
) -> anthropic.AsyncAnthropic:
    """Create an ``AsyncAnthropic`` client for the given model."""
    headers: dict[str, str] = {}
    if model.headers:
        headers.update(model.headers)
    if opts.headers:
        headers.update(opts.headers)

    return anthropic.AsyncAnthropic(
        api_key=api_key,
        base_url=model.base_url,
        default_headers=headers or None,
    )


def _build_params(
    model: Model,
    context: Context,
    opts: _AnthropicOptions,
) -> dict[str, Any]:
    """Build the parameters dict for ``client.messages.stream(...)``."""
    cache_control = _get_cache_control(model.base_url, opts.cache_retention)

    params: dict[str, Any] = {
        "model": model.id,
        "messages": _convert_messages(context.messages, model, cache_control),
        "max_tokens": opts.max_tokens or (model.max_tokens // 3),
    }

    # System prompt
    if context.system_prompt:
        system_block: dict[str, Any] = {
            "type": "text",
            "text": context.system_prompt,
        }
        if cache_control:
            system_block["cache_control"] = cache_control
        params["system"] = [system_block]

    # Temperature
    if opts.temperature is not None:
        params["temperature"] = opts.temperature

    # Tools
    if context.tools:
        params["tools"] = _convert_tools(context.tools)

    # Thinking configuration
    if opts.thinking_enabled and model.reasoning:
        if _supports_adaptive_thinking(model.id):
            params["thinking"] = {"type": "adaptive"}
            if opts.effort:
                params["output_config"] = {"effort": opts.effort}
        else:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": opts.thinking_budget_tokens or 1024,
            }

    # Metadata
    if opts.metadata:
        user_id = opts.metadata.get("user_id")
        if isinstance(user_id, str):
            params["metadata"] = {"user_id": user_id}

    # Tool choice
    if opts.tool_choice is not None:
        if isinstance(opts.tool_choice, str):
            params["tool_choice"] = {"type": opts.tool_choice}
        else:
            params["tool_choice"] = opts.tool_choice

    return params


def _anthropic_options_from_stream(
    options: StreamOptions | None,
) -> _AnthropicOptions:
    """Convert generic :class:`StreamOptions` into internal Anthropic options."""
    if options is None:
        return _AnthropicOptions()
    return _AnthropicOptions(
        temperature=options.temperature,
        max_tokens=options.max_tokens,
        api_key=options.api_key,
        cache_retention=options.cache_retention,
        headers=options.headers,
        metadata=options.metadata,
        abort_event=options.abort_event,
    )
