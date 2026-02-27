"""Core type definitions for the unified AI API.

Python equivalents of the TypeScript types from packages/ai/src/types.ts.
All value objects are frozen dataclasses (immutable).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal, Union


# ---------------------------------------------------------------------------
# Literal type aliases
# ---------------------------------------------------------------------------

KnownApi = Literal[
    "openai-completions",
    "openai-responses",
    "azure-openai-responses",
    "openai-codex-responses",
    "anthropic-messages",
    "bedrock-converse-stream",
    "google-generative-ai",
    "google-gemini-cli",
    "google-vertex",
]

Api = str
"""API identifier. One of KnownApi or any provider-specific string."""

KnownProvider = Literal[
    "amazon-bedrock",
    "anthropic",
    "google",
    "google-gemini-cli",
    "google-antigravity",
    "google-vertex",
    "openai",
    "azure-openai-responses",
    "openai-codex",
    "github-copilot",
    "xai",
    "groq",
    "cerebras",
    "openrouter",
    "vercel-ai-gateway",
    "zai",
    "mistral",
    "minimax",
    "minimax-cn",
    "huggingface",
    "opencode",
    "kimi-coding",
]

Provider = str
"""Provider identifier. One of KnownProvider or any custom string."""

ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh"]

CacheRetention = Literal["none", "short", "long"]

Transport = Literal["sse", "websocket", "auto"]

StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]

InputModality = Literal["text", "image"]

MaxTokensField = Literal["max_completion_tokens", "max_tokens"]

ThinkingFormat = Literal["openai", "zai", "qwen"]


# ---------------------------------------------------------------------------
# Content dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextContent:
    """A block of text content."""

    text: str
    type: Literal["text"] = "text"
    text_signature: str | None = None


@dataclass(frozen=True)
class ThinkingContent:
    """A block of model thinking / chain-of-thought content."""

    thinking: str
    type: Literal["thinking"] = "thinking"
    thinking_signature: str | None = None


@dataclass(frozen=True)
class ImageContent:
    """Base64-encoded image content."""

    data: str
    mime_type: str
    type: Literal["image"] = "image"


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation issued by the model."""

    id: str
    name: str
    arguments: dict[str, Any]
    type: Literal["toolCall"] = "toolCall"
    thought_signature: str | None = None


Content = Union[TextContent, ThinkingContent, ImageContent, ToolCall]
"""Union of all content block types."""

UserContent = Union[str, list[Union[TextContent, ImageContent]]]
"""Content that can appear in a user message: plain string or mixed text/image blocks."""

AssistantContent = list[Union[TextContent, ThinkingContent, ToolCall]]
"""Content that can appear in an assistant message."""

ToolResultContent = list[Union[TextContent, ImageContent]]
"""Content that can appear in a tool result message."""


# ---------------------------------------------------------------------------
# Cost / Usage dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cost:
    """Monetary cost breakdown for a single API call."""

    input: float
    output: float
    cache_read: float
    cache_write: float
    total: float


@dataclass(frozen=True)
class Usage:
    """Token usage and cost for a single API call."""

    input: int
    output: int
    cache_read: int
    cache_write: int
    total_tokens: int
    cost: Cost


# ---------------------------------------------------------------------------
# Message dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UserMessage:
    """A message sent by the user."""

    content: UserContent
    timestamp: float
    role: Literal["user"] = "user"


@dataclass(frozen=True)
class AssistantMessage:
    """A message produced by the model."""

    content: AssistantContent
    api: Api
    provider: Provider
    model: str
    usage: Usage
    stop_reason: StopReason
    timestamp: float
    role: Literal["assistant"] = "assistant"
    error_message: str | None = None


@dataclass(frozen=True)
class ToolResultMessage:
    """The result of a tool invocation, returned to the model."""

    tool_call_id: str
    tool_name: str
    content: ToolResultContent
    is_error: bool
    timestamp: float
    role: Literal["toolResult"] = "toolResult"
    details: Any = None


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]
"""Union of all message types."""


# ---------------------------------------------------------------------------
# Tool / Context dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    """Description of a tool the model may invoke."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class Context:
    """Full conversation context sent to the model."""

    messages: list[Message]
    system_prompt: str | None = None
    tools: list[Tool] | None = None


# ---------------------------------------------------------------------------
# Thinking budgets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThinkingBudgets:
    """Token budgets for each thinking level (token-based providers only)."""

    minimal: int | None = None
    low: int | None = None
    medium: int | None = None
    high: int | None = None


# ---------------------------------------------------------------------------
# Stream options dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamOptions:
    """Options shared by all streaming providers."""

    temperature: float | None = None
    max_tokens: int | None = None
    abort_event: asyncio.Event | None = None
    api_key: str | None = None
    transport: Transport | None = None
    cache_retention: CacheRetention | None = None
    session_id: str | None = None
    headers: dict[str, str] | None = None
    max_retry_delay_ms: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class SimpleStreamOptions(StreamOptions):
    """Extended stream options that include reasoning/thinking controls."""

    reasoning: ThinkingLevel | None = None
    thinking_budgets: ThinkingBudgets | None = None


# ---------------------------------------------------------------------------
# Model dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCost:
    """Per-million-token cost for a model."""

    input: float
    output: float
    cache_read: float
    cache_write: float


@dataclass(frozen=True)
class OpenRouterRouting:
    """OpenRouter provider routing preferences.

    Controls which upstream providers OpenRouter routes requests to.
    """

    only: list[str] | None = None
    order: list[str] | None = None


@dataclass(frozen=True)
class VercelGatewayRouting:
    """Vercel AI Gateway routing preferences.

    Controls which upstream providers the gateway routes requests to.
    """

    only: list[str] | None = None
    order: list[str] | None = None


@dataclass(frozen=True)
class OpenAICompletionsCompat:
    """Compatibility settings for OpenAI-compatible completions APIs.

    Use this to override URL-based auto-detection for custom providers.
    """

    supports_store: bool | None = None
    supports_developer_role: bool | None = None
    supports_reasoning_effort: bool | None = None
    supports_usage_in_streaming: bool | None = None
    max_tokens_field: MaxTokensField | None = None
    requires_tool_result_name: bool | None = None
    requires_assistant_after_tool_result: bool | None = None
    requires_thinking_as_text: bool | None = None
    requires_mistral_tool_ids: bool | None = None
    thinking_format: ThinkingFormat | None = None
    open_router_routing: OpenRouterRouting | None = None
    vercel_gateway_routing: VercelGatewayRouting | None = None
    supports_strict_mode: bool | None = None


@dataclass(frozen=True)
class OpenAIResponsesCompat:
    """Compatibility settings for OpenAI Responses APIs.

    Reserved for future use.
    """


ModelCompat = Union[OpenAICompletionsCompat, OpenAIResponsesCompat, None]
"""Compat settings union, depending on the model's API type."""


@dataclass(frozen=True)
class Model:
    """A model definition in the unified model system."""

    id: str
    name: str
    api: Api
    provider: Provider
    base_url: str
    reasoning: bool
    input: list[InputModality]
    cost: ModelCost
    context_window: int
    max_tokens: int
    headers: dict[str, str] | None = None
    compat: ModelCompat = None


# ---------------------------------------------------------------------------
# Assistant message event types (discriminated union for streaming)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StartEvent:
    """Emitted when the assistant message stream begins."""

    partial: AssistantMessage
    type: Literal["start"] = "start"


@dataclass(frozen=True)
class TextStartEvent:
    """Emitted when a new text content block starts."""

    content_index: int
    partial: AssistantMessage
    type: Literal["text_start"] = "text_start"


@dataclass(frozen=True)
class TextDeltaEvent:
    """Emitted for each incremental text chunk."""

    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["text_delta"] = "text_delta"


@dataclass(frozen=True)
class TextEndEvent:
    """Emitted when a text content block finishes."""

    content_index: int
    content: str
    partial: AssistantMessage
    type: Literal["text_end"] = "text_end"


@dataclass(frozen=True)
class ThinkingStartEvent:
    """Emitted when a new thinking content block starts."""

    content_index: int
    partial: AssistantMessage
    type: Literal["thinking_start"] = "thinking_start"


@dataclass(frozen=True)
class ThinkingDeltaEvent:
    """Emitted for each incremental thinking chunk."""

    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass(frozen=True)
class ThinkingEndEvent:
    """Emitted when a thinking content block finishes."""

    content_index: int
    content: str
    partial: AssistantMessage
    type: Literal["thinking_end"] = "thinking_end"


@dataclass(frozen=True)
class ToolCallStartEvent:
    """Emitted when a new tool call starts."""

    content_index: int
    partial: AssistantMessage
    type: Literal["toolcall_start"] = "toolcall_start"


@dataclass(frozen=True)
class ToolCallDeltaEvent:
    """Emitted for each incremental tool call argument chunk."""

    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["toolcall_delta"] = "toolcall_delta"


@dataclass(frozen=True)
class ToolCallEndEvent:
    """Emitted when a tool call finishes."""

    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage
    type: Literal["toolcall_end"] = "toolcall_end"


@dataclass(frozen=True)
class DoneEvent:
    """Emitted when the stream completes successfully."""

    reason: Literal["stop", "length", "toolUse"]
    message: AssistantMessage
    type: Literal["done"] = "done"


@dataclass(frozen=True)
class ErrorEvent:
    """Emitted when the stream ends due to an error or abort."""

    reason: Literal["aborted", "error"]
    error: AssistantMessage
    type: Literal["error"] = "error"


AssistantMessageEvent = Union[
    StartEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    ThinkingStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    DoneEvent,
    ErrorEvent,
]
"""Discriminated union of all streaming events."""
