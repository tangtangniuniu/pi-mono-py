from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol, Union

from pi_mono.ai.types import (
    AssistantMessageEvent,
    ImageContent,
    Message,
    Model,
    TextContent,
    ThinkingLevel,
    ToolResultMessage,
)

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable, Callable


class AgentThinkingLevel(StrEnum):
    """Agent-level thinking level (includes 'off')."""
    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


# AgentMessage is a union of standard LLM messages + custom messages
# In Python, we use a simple Union type. Custom messages can be added by apps.
AgentMessage = Union[Message, Any]  # Apps can use Any for custom message types


@dataclass
class AgentToolResult:
    """Result of a tool execution."""
    content: list[TextContent | ImageContent]
    details: Any = None


class ToolExecutor(Protocol):
    """Protocol for tool execution."""
    async def execute(
        self,
        tool_call_id: str,
        name: str,
        arguments: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Callable[[AgentToolResult], None] | None = None,
    ) -> AgentToolResult: ...


@dataclass
class AgentTool:
    """An executable tool with schema and execute function."""
    name: str
    description: str
    label: str
    parameters: dict[str, Any]  # JSON Schema
    execute: Callable[..., Awaitable[AgentToolResult]]  # async callable


@dataclass
class AgentContext:
    """Agent context with system prompt, messages, and tools."""
    system_prompt: str
    messages: list[AgentMessage]
    tools: list[AgentTool] | None = None


@dataclass
class AgentLoopConfig:
    """Configuration for the agent loop."""
    model: Model
    convert_to_llm: Callable[[list[AgentMessage]], list[Message] | Awaitable[list[Message]]]
    reasoning: ThinkingLevel | None = None
    session_id: str | None = None
    transport: str | None = None
    thinking_budgets: dict[str, int] | None = None
    max_retry_delay_ms: int | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    cache_retention: str | None = None
    headers: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None
    transform_context: Callable[[list[AgentMessage], asyncio.Event | None], Awaitable[list[AgentMessage]]] | None = None
    get_api_key: Callable[[str], Awaitable[str | None] | str | None] | None = None
    get_steering_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = None
    get_follow_up_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = None


@dataclass
class AgentState:
    """Agent state containing all configuration and conversation data."""
    system_prompt: str = ""
    model: Model | None = None
    thinking_level: AgentThinkingLevel = AgentThinkingLevel.OFF
    tools: list[AgentTool] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    is_streaming: bool = False
    stream_message: AgentMessage | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error: str | None = None


# Agent events - discriminated union using Literal type field
@dataclass(frozen=True)
class AgentStartEvent:
    type: Literal["agent_start"] = "agent_start"

@dataclass(frozen=True)
class AgentEndEvent:
    messages: list[AgentMessage]
    type: Literal["agent_end"] = "agent_end"

@dataclass(frozen=True)
class TurnStartEvent:
    type: Literal["turn_start"] = "turn_start"

@dataclass(frozen=True)
class TurnEndEvent:
    message: AgentMessage
    tool_results: list[ToolResultMessage]
    type: Literal["turn_end"] = "turn_end"

@dataclass(frozen=True)
class MessageStartEvent:
    message: AgentMessage
    type: Literal["message_start"] = "message_start"

@dataclass(frozen=True)
class MessageUpdateEvent:
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent
    type: Literal["message_update"] = "message_update"

@dataclass(frozen=True)
class MessageEndEvent:
    message: AgentMessage
    type: Literal["message_end"] = "message_end"

@dataclass(frozen=True)
class ToolExecutionStartEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    type: Literal["tool_execution_start"] = "tool_execution_start"

@dataclass(frozen=True)
class ToolExecutionUpdateEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any
    type: Literal["tool_execution_update"] = "tool_execution_update"

@dataclass(frozen=True)
class ToolExecutionEndEvent:
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool
    type: Literal["tool_execution_end"] = "tool_execution_end"

AgentEvent = Union[
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolExecutionEndEvent,
]
