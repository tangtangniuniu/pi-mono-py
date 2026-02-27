"""Google Generative AI (Gemini) provider.

Implements the :class:`LLMProvider` protocol using the ``google-genai`` SDK
async client.  Converts pi_mono message types to/from Gemini ``Content``
objects, handles streaming, thinking, tool use, and images.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Literal,
)

from google import genai
from google.genai import types as gtypes

from pi_mono.ai.env_api_keys import get_env_api_key
from pi_mono.ai.models import calculate_cost
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
    ThinkingBudgets,
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
    Usage,
)

if TYPE_CHECKING:
    from pi_mono.ai.types import Message


# ---------------------------------------------------------------------------
# Literal types
# ---------------------------------------------------------------------------

GoogleThinkingLevel = Literal[
    "THINKING_LEVEL_UNSPECIFIED",
    "MINIMAL",
    "LOW",
    "MEDIUM",
    "HIGH",
]

ClampedThinkingLevel = Literal["minimal", "low", "medium", "high"]

ToolChoiceMode = Literal["auto", "none", "any"]


# ---------------------------------------------------------------------------
# Google-specific stream options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoogleStreamOptions(StreamOptions):
    """Extended stream options for the Google Generative AI provider."""

    tool_choice: ToolChoiceMode | None = None
    thinking_enabled: bool = False
    thinking_budget_tokens: int | None = None
    thinking_level: GoogleThinkingLevel | None = None


# ---------------------------------------------------------------------------
# Module-level counter for generating unique tool call IDs
# ---------------------------------------------------------------------------

_tool_call_counter: int = 0


def _next_tool_call_id() -> int:
    global _tool_call_counter  # noqa: PLW0603
    _tool_call_counter += 1
    return _tool_call_counter


# ---------------------------------------------------------------------------
# Surrogate sanitisation
# ---------------------------------------------------------------------------

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _sanitize_surrogates(text: str) -> str:
    """Replace unpaired Unicode surrogates with the replacement character."""
    return _SURROGATE_RE.sub("\ufffd", text)


# ---------------------------------------------------------------------------
# Thought-signature helpers
# ---------------------------------------------------------------------------

_BASE64_SIGNATURE_RE = re.compile(r"^[A-Za-z0-9+/]+=*$")


def _is_valid_thought_signature(signature: str | None) -> bool:
    if not signature:
        return False
    if len(signature) % 4 != 0:
        return False
    return bool(_BASE64_SIGNATURE_RE.match(signature))


def _resolve_thought_signature(
    is_same_provider_and_model: bool,
    signature: str | None,
) -> str | None:
    if is_same_provider_and_model and _is_valid_thought_signature(signature):
        return signature
    return None


def _retain_thought_signature(
    existing: str | None,
    incoming: str | None,
) -> str | None:
    """Keep the most recent non-empty signature within a streamed block."""
    if isinstance(incoming, str) and len(incoming) > 0:
        return incoming
    return existing


def _is_thinking_part(part: Any) -> bool:
    """Return whether a Gemini part represents thinking content."""
    return getattr(part, "thought", None) is True


# ---------------------------------------------------------------------------
# Model-id helpers
# ---------------------------------------------------------------------------


def _requires_tool_call_id(model_id: str) -> bool:
    return model_id.startswith("claude-") or model_id.startswith("gpt-oss-")


def _is_gemini_3_pro(model_id: str) -> bool:
    return "3-pro" in model_id


def _is_gemini_3_flash(model_id: str) -> bool:
    return "3-flash" in model_id


def _is_gemini_3(model_id: str) -> bool:
    return "gemini-3" in model_id.lower()


# ---------------------------------------------------------------------------
# Message conversion  (pi_mono -> Gemini Content[])
# ---------------------------------------------------------------------------


def _transform_messages(
    messages: list[Message],
    model: Model,
) -> list[Message]:
    """Normalise messages before converting to Gemini format.

    * Strips thinking blocks when replaying across models.
    * Inserts synthetic tool-result messages for orphaned tool calls.
    * Normalises tool call IDs when required by the target model.
    """
    from pi_mono.ai.types import (
        AssistantMessage as _AM,
        TextContent as _TC,
        ThinkingContent as _ThC,
        ToolCall as _ToolCall,
        ToolResultMessage as _TRM,
    )

    tool_call_id_map: dict[str, str] = {}

    def _normalize_id(original: str) -> str:
        if not _requires_tool_call_id(model.id):
            return original
        normalised = re.sub(r"[^a-zA-Z0-9_-]", "_", original)[:64]
        if normalised != original:
            tool_call_id_map[original] = normalised
        return normalised

    first_pass: list[Message] = []

    for msg in messages:
        if msg.role == "user":
            first_pass.append(msg)
            continue

        if msg.role == "toolResult":
            new_id = tool_call_id_map.get(msg.tool_call_id, msg.tool_call_id)
            if new_id != msg.tool_call_id:
                first_pass.append(
                    _TRM(
                        tool_call_id=new_id,
                        tool_name=msg.tool_name,
                        content=msg.content,
                        is_error=msg.is_error,
                        timestamp=msg.timestamp,
                        details=msg.details,
                    )
                )
            else:
                first_pass.append(msg)
            continue

        if msg.role == "assistant":
            assert isinstance(msg, _AM)
            is_same = (
                msg.provider == model.provider and msg.model == model.id
            )

            new_content: list[TextContent | ThinkingContent | ToolCall] = []
            for block in msg.content:
                if isinstance(block, _ThC):
                    if is_same and block.thinking_signature:
                        new_content.append(block)
                    elif not block.thinking or block.thinking.strip() == "":
                        continue
                    elif is_same:
                        new_content.append(block)
                    else:
                        new_content.append(_TC(text=block.thinking))
                elif isinstance(block, _TC):
                    if is_same:
                        new_content.append(block)
                    else:
                        new_content.append(_TC(text=block.text))
                elif isinstance(block, _ToolCall):
                    tc = block
                    sig = tc.thought_signature if is_same else None
                    norm_id = _normalize_id(tc.id) if not is_same else tc.id
                    new_content.append(
                        _ToolCall(
                            id=norm_id,
                            name=tc.name,
                            arguments=tc.arguments,
                            thought_signature=sig,
                        )
                    )
                else:
                    new_content.append(block)

            first_pass.append(
                _AM(
                    content=new_content,
                    api=msg.api,
                    provider=msg.provider,
                    model=msg.model,
                    usage=msg.usage,
                    stop_reason=msg.stop_reason,
                    timestamp=msg.timestamp,
                    error_message=msg.error_message,
                )
            )
            continue

        first_pass.append(msg)

    # Second pass: insert synthetic tool results for orphaned tool calls
    result: list[Message] = []
    pending_tool_calls: list[ToolCall] = []
    existing_tool_result_ids: set[str] = set()

    def _flush_orphans() -> None:
        for tc in pending_tool_calls:
            if tc.id not in existing_tool_result_ids:
                result.append(
                    _TRM(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        content=[_TC(text="No result provided")],
                        is_error=True,
                        timestamp=time.time() * 1000,
                    )
                )

    for msg in first_pass:
        if msg.role == "assistant":
            assert isinstance(msg, _AM)
            if pending_tool_calls:
                _flush_orphans()
                pending_tool_calls = []
                existing_tool_result_ids = set()

            if msg.stop_reason in ("error", "aborted"):
                continue

            tcs = [b for b in msg.content if isinstance(b, _ToolCall)]
            if tcs:
                pending_tool_calls = tcs
                existing_tool_result_ids = set()
            result.append(msg)

        elif msg.role == "toolResult":
            existing_tool_result_ids.add(msg.tool_call_id)
            result.append(msg)

        elif msg.role == "user":
            if pending_tool_calls:
                _flush_orphans()
                pending_tool_calls = []
                existing_tool_result_ids = set()
            result.append(msg)
        else:
            result.append(msg)

    return result


def _convert_messages(model: Model, context: Context) -> list[gtypes.Content]:
    """Convert pi_mono messages to Gemini ``Content`` objects."""
    contents: list[gtypes.Content] = []
    transformed = _transform_messages(context.messages, model)

    for msg in transformed:
        if msg.role == "user":
            if isinstance(msg.content, str):
                contents.append(
                    gtypes.Content(
                        role="user",
                        parts=[gtypes.Part(text=_sanitize_surrogates(msg.content))],
                    )
                )
            else:
                parts: list[gtypes.Part] = []
                for item in msg.content:
                    if isinstance(item, TextContent):
                        parts.append(
                            gtypes.Part(text=_sanitize_surrogates(item.text))
                        )
                    elif isinstance(item, ImageContent):
                        parts.append(
                            gtypes.Part(
                                inline_data=gtypes.Blob(
                                    mime_type=item.mime_type,
                                    data=item.data,
                                ),
                            )
                        )
                if "image" not in model.input:
                    parts = [p for p in parts if p.text is not None]
                if not parts:
                    continue
                contents.append(gtypes.Content(role="user", parts=parts))

        elif msg.role == "assistant":
            assert isinstance(msg, AssistantMessage)
            is_same = msg.provider == model.provider and msg.model == model.id
            parts = []
            for block in msg.content:
                if isinstance(block, TextContent):
                    if not block.text or block.text.strip() == "":
                        continue
                    sig = _resolve_thought_signature(is_same, block.text_signature)
                    p = gtypes.Part(text=_sanitize_surrogates(block.text))
                    if sig:
                        p.thought_signature = sig
                    parts.append(p)

                elif isinstance(block, ThinkingContent):
                    if not block.thinking or block.thinking.strip() == "":
                        continue
                    if is_same:
                        sig = _resolve_thought_signature(
                            is_same, block.thinking_signature
                        )
                        p = gtypes.Part(
                            thought=True,
                            text=_sanitize_surrogates(block.thinking),
                        )
                        if sig:
                            p.thought_signature = sig
                        parts.append(p)
                    else:
                        parts.append(
                            gtypes.Part(text=_sanitize_surrogates(block.thinking))
                        )

                elif isinstance(block, ToolCall):
                    sig = _resolve_thought_signature(is_same, block.thought_signature)
                    if _is_gemini_3(model.id) and not sig:
                        args_str = json.dumps(block.arguments or {}, indent=2)
                        parts.append(
                            gtypes.Part(
                                text=(
                                    f'[Historical context: a different model called tool '
                                    f'"{block.name}" with arguments: {args_str}. '
                                    f'Do not mimic this format - use proper function calling.]'
                                ),
                            )
                        )
                    else:
                        fc = gtypes.FunctionCall(
                            name=block.name,
                            args=block.arguments or {},
                        )
                        if _requires_tool_call_id(model.id):
                            fc.id = block.id
                        p = gtypes.Part(function_call=fc)
                        if sig:
                            p.thought_signature = sig
                        parts.append(p)

            if not parts:
                continue
            contents.append(gtypes.Content(role="model", parts=parts))

        elif msg.role == "toolResult":
            text_pieces = [
                c.text for c in msg.content if isinstance(c, TextContent)
            ]
            text_result = "\n".join(text_pieces)
            image_blocks = (
                [c for c in msg.content if isinstance(c, ImageContent)]
                if "image" in model.input
                else []
            )
            has_text = len(text_result) > 0
            has_images = len(image_blocks) > 0

            supports_multimodal_fr = "gemini-3" in model.id
            response_value = (
                _sanitize_surrogates(text_result)
                if has_text
                else ("(see attached image)" if has_images else "")
            )

            image_parts = [
                gtypes.Part(
                    inline_data=gtypes.Blob(
                        mime_type=img.mime_type,
                        data=img.data,
                    ),
                )
                for img in image_blocks
            ]

            response_dict: dict[str, Any] = (
                {"error": response_value}
                if msg.is_error
                else {"output": response_value}
            )

            fr = gtypes.FunctionResponse(
                name=msg.tool_name,
                response=response_dict,
            )
            if _requires_tool_call_id(model.id):
                fr.id = msg.tool_call_id

            fr_part = gtypes.Part(function_response=fr)

            # Merge consecutive tool-result parts into the same user turn
            last = contents[-1] if contents else None
            if (
                last is not None
                and last.role == "user"
                and last.parts
                and any(
                    getattr(p, "function_response", None) is not None
                    for p in last.parts
                )
            ):
                last.parts.append(fr_part)
            else:
                contents.append(
                    gtypes.Content(role="user", parts=[fr_part])
                )

            # For older models, add images in a separate user turn
            if has_images and not supports_multimodal_fr:
                contents.append(
                    gtypes.Content(
                        role="user",
                        parts=[
                            gtypes.Part(text="Tool result image:"),
                            *image_parts,
                        ],
                    )
                )

    return contents


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


def _convert_tools(
    tools: list[Tool],
) -> list[gtypes.Tool] | None:
    """Convert pi_mono ``Tool`` definitions to Gemini ``FunctionDeclaration``s."""
    if not tools:
        return None
    declarations = [
        gtypes.FunctionDeclaration(
            name=t.name,
            description=t.description,
            parameters_json_schema=t.parameters,
        )
        for t in tools
    ]
    return [gtypes.Tool(function_declarations=declarations)]


# ---------------------------------------------------------------------------
# Enum / mapping helpers
# ---------------------------------------------------------------------------


def _map_tool_choice(choice: str) -> str:
    """Map tool choice string to Gemini function-calling mode string."""
    mapping = {
        "auto": "AUTO",
        "none": "NONE",
        "any": "ANY",
    }
    return mapping.get(choice, "AUTO")


def _map_stop_reason(reason: str) -> StopReason:
    """Map a Gemini finish reason to a pi_mono ``StopReason``."""
    if reason == "STOP":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    return "error"


# ---------------------------------------------------------------------------
# Thinking-level / budget helpers
# ---------------------------------------------------------------------------


def _clamp_reasoning(
    effort: ThinkingLevel | None,
) -> ClampedThinkingLevel | None:
    if effort == "xhigh":
        return "high"
    return effort  # type: ignore[return-value]


_GEMINI_25_PRO_BUDGETS: dict[ClampedThinkingLevel, int] = {
    "minimal": 128,
    "low": 2048,
    "medium": 8192,
    "high": 32768,
}

_GEMINI_25_FLASH_BUDGETS: dict[ClampedThinkingLevel, int] = {
    "minimal": 128,
    "low": 2048,
    "medium": 8192,
    "high": 24576,
}


def _get_google_budget(
    model: Model,
    effort: ClampedThinkingLevel,
    custom_budgets: ThinkingBudgets | None = None,
) -> int:
    """Return the thinking budget in tokens for a given model and effort."""
    if custom_budgets is not None:
        custom = getattr(custom_budgets, effort, None)
        if custom is not None:
            return custom

    if "2.5-pro" in model.id:
        return _GEMINI_25_PRO_BUDGETS[effort]
    if "2.5-flash" in model.id:
        return _GEMINI_25_FLASH_BUDGETS[effort]
    return -1  # dynamic


def _get_gemini_3_thinking_level(
    effort: ClampedThinkingLevel,
    model: Model,
) -> GoogleThinkingLevel:
    """Return the Gemini 3 thinking-level enum value."""
    if _is_gemini_3_pro(model.id):
        return {"minimal": "LOW", "low": "LOW", "medium": "HIGH", "high": "HIGH"}[
            effort
        ]  # type: ignore[return-value]
    return {
        "minimal": "MINIMAL",
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "HIGH",
    }[effort]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Mutable helper used during streaming  (not part of public API)
# ---------------------------------------------------------------------------


class _MutableAssistantMessage:
    """Accumulator for building an ``AssistantMessage`` during streaming.

    pi_mono types are frozen dataclasses.  During streaming we need to mutate
    the in-progress message, so we keep mutable state here and freeze a
    snapshot whenever we need to emit an event.
    """

    def __init__(self, model: Model) -> None:
        self.content: list[TextContent | ThinkingContent | ToolCall] = []
        self.api: str = "google-generative-ai"
        self.provider: str = model.provider
        self.model_id: str = model.id
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cache_read_tokens: int = 0
        self.total_tokens: int = 0
        self.stop_reason: StopReason = "stop"
        self.error_message: str | None = None
        self.timestamp: float = time.time() * 1000

    def freeze(self) -> AssistantMessage:
        cost = Cost(
            input=0.0, output=0.0, cache_read=0.0, cache_write=0.0, total=0.0,
        )
        usage = Usage(
            input=self.input_tokens,
            output=self.output_tokens,
            cache_read=self.cache_read_tokens,
            cache_write=0,
            total_tokens=self.total_tokens,
            cost=cost,
        )
        return AssistantMessage(
            content=list(self.content),
            api=self.api,
            provider=self.provider,
            model=self.model_id,
            usage=usage,
            stop_reason=self.stop_reason,
            timestamp=self.timestamp,
            error_message=self.error_message,
        )


# ---------------------------------------------------------------------------
# Client creation
# ---------------------------------------------------------------------------


def _create_client(
    model: Model,
    api_key: str,
    extra_headers: dict[str, str] | None = None,
) -> genai.Client:
    """Instantiate an *async* ``google.genai`` client."""
    http_options: dict[str, Any] = {}
    if model.base_url:
        http_options["base_url"] = model.base_url
        http_options["api_version"] = ""
    merged_headers = {**(model.headers or {}), **(extra_headers or {})}
    if merged_headers:
        http_options["headers"] = merged_headers

    return genai.Client(
        api_key=api_key,
        http_options=http_options if http_options else None,
    )


# ---------------------------------------------------------------------------
# Request parameters builder
# ---------------------------------------------------------------------------


def _build_generate_config(
    model: Model,
    context: Context,
    options: GoogleStreamOptions,
) -> tuple[list[gtypes.Content], gtypes.GenerateContentConfig]:
    """Return ``(contents, config)`` ready for ``generate_content_stream``."""
    contents = _convert_messages(model, context)

    config = gtypes.GenerateContentConfig()

    if options.temperature is not None:
        config.temperature = options.temperature
    if options.max_tokens is not None:
        config.max_output_tokens = options.max_tokens

    if context.system_prompt:
        config.system_instruction = _sanitize_surrogates(context.system_prompt)

    if context.tools:
        converted = _convert_tools(context.tools)
        if converted:
            config.tools = converted

    # Tool choice
    if context.tools and options.tool_choice:
        config.tool_config = gtypes.ToolConfig(
            function_calling_config=gtypes.FunctionCallingConfig(
                mode=_map_tool_choice(options.tool_choice),
            ),
        )

    # Thinking configuration
    if options.thinking_enabled and model.reasoning:
        thinking_config = gtypes.ThinkingConfig(include_thoughts=True)
        if options.thinking_level is not None:
            thinking_config.thinking_level = options.thinking_level
        elif options.thinking_budget_tokens is not None:
            thinking_config.thinking_budget = options.thinking_budget_tokens
        config.thinking_config = thinking_config

    return contents, config


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class GoogleProvider:
    """Google Generative AI provider implementing :class:`LLMProvider`."""

    @property
    def api(self) -> str:
        return "google-generative-ai"

    # -- public streaming entry points -------------------------------------

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """Stream assistant events using provider-specific options.

        *options* should be a :class:`GoogleStreamOptions` instance.  A plain
        :class:`StreamOptions` is accepted and treated as having thinking
        disabled and no tool-choice override.
        """
        google_opts = (
            options
            if isinstance(options, GoogleStreamOptions)
            else _to_google_options(options)
        )
        return self._stream_impl(model, context, google_opts)

    async def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """Stream events using the simplified reasoning-level interface."""
        api_key = (
            (options.api_key if options else None)
            or get_env_api_key(model.provider)
        )
        if not api_key:
            raise ValueError(f"No API key for provider: {model.provider}")

        base = _build_base_options(model, options, api_key)

        if options is None or options.reasoning is None:
            google_opts = GoogleStreamOptions(
                **_stream_opts_dict(base),
                thinking_enabled=False,
            )
            return self._stream_impl(model, context, google_opts)

        effort = _clamp_reasoning(options.reasoning)
        assert effort is not None

        if _is_gemini_3_pro(model.id) or _is_gemini_3_flash(model.id):
            google_opts = GoogleStreamOptions(
                **_stream_opts_dict(base),
                thinking_enabled=True,
                thinking_level=_get_gemini_3_thinking_level(effort, model),
            )
            return self._stream_impl(model, context, google_opts)

        budget = _get_google_budget(
            model,
            effort,
            options.thinking_budgets if options else None,
        )
        google_opts = GoogleStreamOptions(
            **_stream_opts_dict(base),
            thinking_enabled=True,
            thinking_budget_tokens=budget,
        )
        return self._stream_impl(model, context, google_opts)

    # -- internal streaming implementation ---------------------------------

    async def _stream_impl(
        self,
        model: Model,
        context: Context,
        options: GoogleStreamOptions,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """Core streaming generator shared by ``stream`` and ``stream_simple``."""
        return self._generate_events(model, context, options)

    async def _generate_events(
        self,
        model: Model,
        context: Context,
        options: GoogleStreamOptions,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """Async generator that yields ``AssistantMessageEvent`` objects."""
        acc = _MutableAssistantMessage(model)

        try:
            api_key = options.api_key or get_env_api_key(model.provider) or ""
            client = _create_client(model, api_key, options.headers)
            contents, config = _build_generate_config(model, context, options)

            google_stream = await client.aio.models.generate_content_stream(
                model=model.id,
                contents=contents,
                config=config,
            )

            yield StartEvent(partial=acc.freeze())

            current_block: TextContent | ThinkingContent | None = None
            # We track text/thinking accumulators mutably as plain dicts so
            # we can append deltas without re-creating frozen dataclasses.
            current_text: str = ""
            current_thinking: str = ""
            current_text_sig: str | None = None
            current_thinking_sig: str | None = None
            current_block_type: str | None = None  # "text" | "thinking" | None

            def _block_index() -> int:
                return len(acc.content) - 1

            async for chunk in google_stream:
                candidate = (
                    chunk.candidates[0]
                    if chunk.candidates
                    else None
                )
                if candidate and candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        part_text = getattr(part, "text", None)
                        if part_text is not None:
                            is_thinking = _is_thinking_part(part)

                            # Check if we need to start a new block
                            if (
                                current_block_type is None
                                or (is_thinking and current_block_type != "thinking")
                                or (not is_thinking and current_block_type != "text")
                            ):
                                # End previous block
                                if current_block_type == "text":
                                    frozen_text = TextContent(
                                        text=current_text,
                                        text_signature=current_text_sig,
                                    )
                                    acc.content[-1] = frozen_text
                                    yield TextEndEvent(
                                        content_index=_block_index(),
                                        content=current_text,
                                        partial=acc.freeze(),
                                    )
                                elif current_block_type == "thinking":
                                    frozen_thinking = ThinkingContent(
                                        thinking=current_thinking,
                                        thinking_signature=current_thinking_sig,
                                    )
                                    acc.content[-1] = frozen_thinking
                                    yield ThinkingEndEvent(
                                        content_index=_block_index(),
                                        content=current_thinking,
                                        partial=acc.freeze(),
                                    )

                                # Start new block
                                if is_thinking:
                                    current_thinking = ""
                                    current_thinking_sig = None
                                    current_block_type = "thinking"
                                    placeholder = ThinkingContent(thinking="")
                                    acc.content.append(placeholder)
                                    yield ThinkingStartEvent(
                                        content_index=_block_index(),
                                        partial=acc.freeze(),
                                    )
                                else:
                                    current_text = ""
                                    current_text_sig = None
                                    current_block_type = "text"
                                    placeholder = TextContent(text="")
                                    acc.content.append(placeholder)
                                    yield TextStartEvent(
                                        content_index=_block_index(),
                                        partial=acc.freeze(),
                                    )

                            # Accumulate delta
                            thought_sig = getattr(part, "thought_signature", None)
                            if current_block_type == "thinking":
                                current_thinking += part_text
                                current_thinking_sig = _retain_thought_signature(
                                    current_thinking_sig, thought_sig,
                                )
                                acc.content[-1] = ThinkingContent(
                                    thinking=current_thinking,
                                    thinking_signature=current_thinking_sig,
                                )
                                yield ThinkingDeltaEvent(
                                    content_index=_block_index(),
                                    delta=part_text,
                                    partial=acc.freeze(),
                                )
                            else:
                                current_text += part_text
                                current_text_sig = _retain_thought_signature(
                                    current_text_sig, thought_sig,
                                )
                                acc.content[-1] = TextContent(
                                    text=current_text,
                                    text_signature=current_text_sig,
                                )
                                yield TextDeltaEvent(
                                    content_index=_block_index(),
                                    delta=part_text,
                                    partial=acc.freeze(),
                                )

                        # Handle function calls
                        fc = getattr(part, "function_call", None)
                        if fc is not None:
                            # Close current text/thinking block
                            if current_block_type == "text":
                                acc.content[-1] = TextContent(
                                    text=current_text,
                                    text_signature=current_text_sig,
                                )
                                yield TextEndEvent(
                                    content_index=_block_index(),
                                    content=current_text,
                                    partial=acc.freeze(),
                                )
                            elif current_block_type == "thinking":
                                acc.content[-1] = ThinkingContent(
                                    thinking=current_thinking,
                                    thinking_signature=current_thinking_sig,
                                )
                                yield ThinkingEndEvent(
                                    content_index=_block_index(),
                                    content=current_thinking,
                                    partial=acc.freeze(),
                                )
                            current_block_type = None

                            # Generate unique tool call ID
                            provided_id = getattr(fc, "id", None)
                            needs_new_id = not provided_id or any(
                                isinstance(b, ToolCall) and b.id == provided_id
                                for b in acc.content
                            )
                            tool_call_id = (
                                f"{fc.name}_{int(time.time() * 1000)}_{_next_tool_call_id()}"
                                if needs_new_id
                                else provided_id
                            )

                            thought_sig = getattr(part, "thought_signature", None)
                            tool_call = ToolCall(
                                id=tool_call_id,
                                name=fc.name or "",
                                arguments=dict(fc.args) if fc.args else {},
                                thought_signature=thought_sig if thought_sig else None,
                            )

                            acc.content.append(tool_call)
                            yield ToolCallStartEvent(
                                content_index=_block_index(),
                                partial=acc.freeze(),
                            )
                            yield ToolCallDeltaEvent(
                                content_index=_block_index(),
                                delta=json.dumps(tool_call.arguments),
                                partial=acc.freeze(),
                            )
                            yield ToolCallEndEvent(
                                content_index=_block_index(),
                                tool_call=tool_call,
                                partial=acc.freeze(),
                            )

                # Finish reason
                if candidate and getattr(candidate, "finish_reason", None):
                    reason_str = str(candidate.finish_reason)
                    # The enum may serialise as e.g. "FinishReason.STOP"
                    if "." in reason_str:
                        reason_str = reason_str.rsplit(".", 1)[-1]
                    acc.stop_reason = _map_stop_reason(reason_str)
                    if any(isinstance(b, ToolCall) for b in acc.content):
                        acc.stop_reason = "toolUse"

                # Usage metadata
                usage_meta = getattr(chunk, "usage_metadata", None)
                if usage_meta is not None:
                    acc.input_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
                    acc.output_tokens = (
                        (getattr(usage_meta, "candidates_token_count", 0) or 0)
                        + (getattr(usage_meta, "thoughts_token_count", 0) or 0)
                    )
                    acc.cache_read_tokens = (
                        getattr(usage_meta, "cached_content_token_count", 0) or 0
                    )
                    acc.total_tokens = getattr(usage_meta, "total_token_count", 0) or 0

            # End the last open block
            if current_block_type == "text":
                acc.content[-1] = TextContent(
                    text=current_text, text_signature=current_text_sig,
                )
                yield TextEndEvent(
                    content_index=_block_index(),
                    content=current_text,
                    partial=acc.freeze(),
                )
            elif current_block_type == "thinking":
                acc.content[-1] = ThinkingContent(
                    thinking=current_thinking,
                    thinking_signature=current_thinking_sig,
                )
                yield ThinkingEndEvent(
                    content_index=_block_index(),
                    content=current_thinking,
                    partial=acc.freeze(),
                )

            # Abort check
            if options.abort_event and options.abort_event.is_set():
                raise RuntimeError("Request was aborted")

            if acc.stop_reason in ("aborted", "error"):
                raise RuntimeError("An unknown error occurred")

            # Apply cost calculation
            final = acc.freeze()
            cost = calculate_cost(model, final.usage)
            final = AssistantMessage(
                content=final.content,
                api=final.api,
                provider=final.provider,
                model=final.model,
                usage=Usage(
                    input=final.usage.input,
                    output=final.usage.output,
                    cache_read=final.usage.cache_read,
                    cache_write=final.usage.cache_write,
                    total_tokens=final.usage.total_tokens,
                    cost=cost,
                ),
                stop_reason=final.stop_reason,
                timestamp=final.timestamp,
                error_message=final.error_message,
            )

            yield DoneEvent(reason=final.stop_reason, message=final)  # type: ignore[arg-type]

        except Exception as exc:
            acc.stop_reason = (
                "aborted"
                if options.abort_event and options.abort_event.is_set()
                else "error"
            )
            acc.error_message = str(exc)
            yield ErrorEvent(reason=acc.stop_reason, error=acc.freeze())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_google_options(
    options: StreamOptions | None,
) -> GoogleStreamOptions:
    """Promote a generic ``StreamOptions`` to ``GoogleStreamOptions``."""
    if options is None:
        return GoogleStreamOptions()
    return GoogleStreamOptions(
        **_stream_opts_dict(options),
    )


def _stream_opts_dict(options: StreamOptions) -> dict[str, Any]:
    """Extract the ``StreamOptions`` fields as a plain dict."""
    return {
        "temperature": options.temperature,
        "max_tokens": options.max_tokens,
        "abort_event": options.abort_event,
        "api_key": options.api_key,
        "transport": options.transport,
        "cache_retention": options.cache_retention,
        "session_id": options.session_id,
        "headers": options.headers,
        "max_retry_delay_ms": options.max_retry_delay_ms,
        "metadata": options.metadata,
    }


def _build_base_options(
    model: Model,
    options: SimpleStreamOptions | None,
    api_key: str,
) -> StreamOptions:
    """Build the base ``StreamOptions`` from a ``SimpleStreamOptions``."""
    max_tokens = (
        (options.max_tokens if options and options.max_tokens else None)
        or min(model.max_tokens, 32000)
    )
    return StreamOptions(
        temperature=options.temperature if options else None,
        max_tokens=max_tokens,
        abort_event=options.abort_event if options else None,
        api_key=api_key or (options.api_key if options else None),
        cache_retention=options.cache_retention if options else None,
        session_id=options.session_id if options else None,
        headers=options.headers if options else None,
        max_retry_delay_ms=options.max_retry_delay_ms if options else None,
        metadata=options.metadata if options else None,
    )
