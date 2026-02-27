"""Stateful AI agent wrapping the agent loop with state management.

Mirrors the TypeScript ``Agent`` class from ``packages/agent/src/agent.ts``.
The agent manages conversation state, message queuing (steering and follow-up),
and event emission while delegating the actual processing to the agent loop.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from pi_mono.ai.types import (
    AssistantMessage,
    Cost,
    ImageContent,
    Message,
    Model,
    TextContent,
    Usage,
    UserMessage,
)
from pi_mono.agent.agent_loop import agent_loop, agent_loop_continue
from pi_mono.agent.types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    AgentThinkingLevel,
    AgentTool,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnEndEvent,
)


def _default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """Keep only LLM-compatible messages (user, assistant, toolResult)."""
    return [m for m in messages if hasattr(m, "role") and m.role in ("user", "assistant", "toolResult")]


class Agent:
    """Stateful AI agent with streaming support.

    Wraps the agent loop with state management, message queuing (steering
    and follow-up), and a subscribe/emit event system for UI integration.

    Parameters
    ----------
    convert_to_llm:
        Converts AgentMessage list to LLM-compatible Message list before each
        LLM call.  Default filters to user/assistant/toolResult messages.
    transform_context:
        Optional transform applied to context before ``convert_to_llm``.
        Use for context pruning, injecting external context, etc.
    stream_fn:
        Custom stream function override.  Default uses ``stream_simple``.
    steering_mode:
        ``"one-at-a-time"`` (default) or ``"all"`` -- controls how many
        steering messages are dequeued per check.
    follow_up_mode:
        ``"one-at-a-time"`` (default) or ``"all"`` -- controls how many
        follow-up messages are dequeued per check.
    session_id:
        Optional session identifier forwarded to LLM providers for
        session-based caching.
    get_api_key:
        Resolves an API key dynamically for each LLM call.
    thinking_budgets:
        Custom token budgets for thinking levels (token-based providers).
    transport:
        Preferred transport for providers that support multiple transports.
    max_retry_delay_ms:
        Maximum delay in milliseconds to wait for server-requested retries.
    initial_state:
        Optional partial state to initialize the agent with.
    """

    def __init__(
        self,
        convert_to_llm: Callable[[list[AgentMessage]], list[Message]] | None = None,
        transform_context: Callable[[list[AgentMessage], asyncio.Event | None], Any] | None = None,
        stream_fn: Callable | None = None,
        steering_mode: str = "one-at-a-time",
        follow_up_mode: str = "one-at-a-time",
        session_id: str | None = None,
        get_api_key: Callable[[str], Any] | None = None,
        thinking_budgets: dict[str, int] | None = None,
        transport: str = "sse",
        max_retry_delay_ms: int | None = None,
        initial_state: dict[str, Any] | None = None,
    ) -> None:
        self._state = AgentState()
        if initial_state:
            for k, v in initial_state.items():
                if hasattr(self._state, k):
                    setattr(self._state, k, v)

        self._convert_to_llm = convert_to_llm or _default_convert_to_llm
        self._transform_context = transform_context
        self._stream_fn = stream_fn
        self._steering_mode = steering_mode
        self._follow_up_mode = follow_up_mode
        self._session_id = session_id
        self._get_api_key = get_api_key
        self._thinking_budgets = thinking_budgets
        self._transport = transport
        self._max_retry_delay_ms = max_retry_delay_ms

        self._listeners: list[Callable[[AgentEvent], Any]] = []
        self._abort_event: asyncio.Event | None = None
        self._steering_queue: list[AgentMessage] = []
        self._follow_up_queue: list[AgentMessage] = []
        self._running_future: asyncio.Future[None] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> AgentState:
        """The current agent state (read-only reference)."""
        return self._state

    @property
    def session_id(self) -> str | None:
        """Session ID used for provider caching."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        self._session_id = value

    @property
    def thinking_budgets(self) -> dict[str, int] | None:
        """Custom thinking budgets for token-based providers."""
        return self._thinking_budgets

    @thinking_budgets.setter
    def thinking_budgets(self, value: dict[str, int] | None) -> None:
        self._thinking_budgets = value

    @property
    def transport(self) -> str:
        """Preferred transport for providers that support multiple transports."""
        return self._transport

    @transport.setter
    def transport(self, value: str) -> None:
        self._transport = value

    @property
    def max_retry_delay_ms(self) -> int | None:
        """Maximum delay to wait for server-requested retries."""
        return self._max_retry_delay_ms

    @max_retry_delay_ms.setter
    def max_retry_delay_ms(self, value: int | None) -> None:
        self._max_retry_delay_ms = value

    # ------------------------------------------------------------------
    # Subscribe / emit
    # ------------------------------------------------------------------

    def subscribe(self, fn: Callable[[AgentEvent], Any]) -> Callable[[], None]:
        """Subscribe to agent events.  Returns an unsubscribe function."""
        self._listeners.append(fn)

        def unsub() -> None:
            if fn in self._listeners:
                self._listeners.remove(fn)

        return unsub

    def _emit(self, event: AgentEvent) -> None:
        """Emit an event to all registered listeners."""
        for listener in self._listeners:
            listener(event)

    # ------------------------------------------------------------------
    # State mutators
    # ------------------------------------------------------------------

    def set_system_prompt(self, value: str) -> None:
        """Set the system prompt."""
        self._state.system_prompt = value

    def set_model(self, model: Model) -> None:
        """Set the model to use."""
        self._state.model = model

    def set_thinking_level(self, level: AgentThinkingLevel) -> None:
        """Set the thinking level for reasoning models."""
        self._state.thinking_level = level

    def set_tools(self, tools: list[AgentTool]) -> None:
        """Set the available tools."""
        self._state.tools = tools

    def replace_messages(self, msgs: list[AgentMessage]) -> None:
        """Replace all messages (immutable copy)."""
        self._state.messages = list(msgs)

    def append_message(self, msg: AgentMessage) -> None:
        """Append a message (immutable -- creates a new list)."""
        self._state.messages = [*self._state.messages, msg]

    def clear_messages(self) -> None:
        """Remove all messages."""
        self._state.messages = []

    # ------------------------------------------------------------------
    # Steering / follow-up queuing
    # ------------------------------------------------------------------

    def steer(self, msg: AgentMessage) -> None:
        """Queue a steering message to interrupt the agent mid-run.

        Delivered after the current tool execution; remaining tools are skipped.
        """
        self._steering_queue.append(msg)

    def follow_up(self, msg: AgentMessage) -> None:
        """Queue a follow-up message for after the agent finishes.

        Delivered only when the agent has no more tool calls or steering.
        """
        self._follow_up_queue.append(msg)

    def clear_steering_queue(self) -> None:
        """Remove all queued steering messages."""
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        """Remove all queued follow-up messages."""
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        """Remove all queued steering and follow-up messages."""
        self._steering_queue.clear()
        self._follow_up_queue.clear()

    def has_queued_messages(self) -> bool:
        """Return whether any steering or follow-up messages are queued."""
        return len(self._steering_queue) > 0 or len(self._follow_up_queue) > 0

    def set_steering_mode(self, mode: str) -> None:
        """Set steering dequeue mode (``'all'`` or ``'one-at-a-time'``)."""
        self._steering_mode = mode

    def get_steering_mode(self) -> str:
        """Return the current steering dequeue mode."""
        return self._steering_mode

    def set_follow_up_mode(self, mode: str) -> None:
        """Set follow-up dequeue mode (``'all'`` or ``'one-at-a-time'``)."""
        self._follow_up_mode = mode

    def get_follow_up_mode(self) -> str:
        """Return the current follow-up dequeue mode."""
        return self._follow_up_mode

    def _dequeue_steering(self) -> list[AgentMessage]:
        """Dequeue steering messages according to the steering mode."""
        if self._steering_mode == "one-at-a-time":
            if self._steering_queue:
                first = self._steering_queue[0]
                self._steering_queue = self._steering_queue[1:]
                return [first]
            return []
        msgs = list(self._steering_queue)
        self._steering_queue = []
        return msgs

    def _dequeue_follow_up(self) -> list[AgentMessage]:
        """Dequeue follow-up messages according to the follow-up mode."""
        if self._follow_up_mode == "one-at-a-time":
            if self._follow_up_queue:
                first = self._follow_up_queue[0]
                self._follow_up_queue = self._follow_up_queue[1:]
                return [first]
            return []
        msgs = list(self._follow_up_queue)
        self._follow_up_queue = []
        return msgs

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def prompt(
        self,
        input_: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
        """Send a prompt and run the agent loop until completion.

        Parameters
        ----------
        input_:
            A string, a single AgentMessage, or a list of AgentMessages.
        images:
            Optional images to include when ``input_`` is a string.

        Raises
        ------
        RuntimeError
            If the agent is already streaming or no model is configured.
        """
        if self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing a prompt. "
                "Use steer() or follow_up() to queue messages, or wait for completion."
            )
        if self._state.model is None:
            raise RuntimeError("No model configured")

        if isinstance(input_, list):
            msgs = input_
        elif isinstance(input_, str):
            content: list[TextContent | ImageContent] = [TextContent(text=input_)]
            if images:
                content.extend(images)
            msgs: list[AgentMessage] = [UserMessage(role="user", content=content, timestamp=time.time())]
        else:
            msgs = [input_]

        await self._run_loop(msgs)

    async def continue_(self) -> None:
        """Continue from the current context.

        Used for retries and resuming queued messages.

        Raises
        ------
        RuntimeError
            If the agent is already streaming, there are no messages, or the
            last message is an assistant message with no queued messages.
        """
        if self._state.is_streaming:
            raise RuntimeError("Agent is already processing. Wait for completion before continuing.")

        messages = self._state.messages
        if not messages:
            raise RuntimeError("No messages to continue from")

        last = messages[-1]
        if hasattr(last, "role") and last.role == "assistant":
            # Try steering first, then follow-up
            queued_steering = self._dequeue_steering()
            if queued_steering:
                await self._run_loop(queued_steering, skip_initial_steering=True)
                return

            queued_follow_up = self._dequeue_follow_up()
            if queued_follow_up:
                await self._run_loop(queued_follow_up)
                return

            raise RuntimeError("Cannot continue from message role: assistant")

        await self._run_loop(None)

    def abort(self) -> None:
        """Abort the currently running agent loop."""
        if self._abort_event is not None:
            self._abort_event.set()

    async def wait_for_idle(self) -> None:
        """Wait until the agent is no longer processing."""
        if self._running_future is not None:
            await self._running_future

    def reset(self) -> None:
        """Reset the agent to its initial state (keeps model and tools)."""
        self._state.messages = []
        self._state.is_streaming = False
        self._state.stream_message = None
        self._state.pending_tool_calls = set()
        self._state.error = None
        self._steering_queue.clear()
        self._follow_up_queue.clear()

    # ------------------------------------------------------------------
    # Internal run loop
    # ------------------------------------------------------------------

    async def _run_loop(
        self,
        messages: list[AgentMessage] | None,
        skip_initial_steering: bool = False,
    ) -> None:
        """Run the agent loop, updating state and emitting events.

        If *messages* is provided, starts a new conversation turn.
        Otherwise, continues from existing context.
        """
        model = self._state.model
        if model is None:
            raise RuntimeError("No model configured")

        loop = asyncio.get_running_loop()
        self._running_future = loop.create_future()
        self._abort_event = asyncio.Event()
        self._state.is_streaming = True
        self._state.stream_message = None
        self._state.error = None

        reasoning = None if self._state.thinking_level == AgentThinkingLevel.OFF else self._state.thinking_level.value

        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=self._state.tools or None,
        )

        _skip = skip_initial_steering

        async def get_steering() -> list[AgentMessage]:
            nonlocal _skip
            if _skip:
                _skip = False
                return []
            return self._dequeue_steering()

        async def get_follow_up() -> list[AgentMessage]:
            return self._dequeue_follow_up()

        config = AgentLoopConfig(
            model=model,
            reasoning=reasoning,
            session_id=self._session_id,
            transport=self._transport,
            thinking_budgets=self._thinking_budgets,
            max_retry_delay_ms=self._max_retry_delay_ms,
            convert_to_llm=self._convert_to_llm,
            transform_context=self._transform_context,
            get_api_key=self._get_api_key,
            get_steering_messages=get_steering,
            get_follow_up_messages=get_follow_up,
        )

        partial: AgentMessage | None = None

        try:
            if messages is not None:
                event_stream = agent_loop(messages, context, config, self._abort_event, self._stream_fn)
            else:
                event_stream = agent_loop_continue(context, config, self._abort_event, self._stream_fn)

            async for event in event_stream:
                # Update internal state based on events
                if isinstance(event, MessageStartEvent):
                    partial = event.message
                    self._state.stream_message = event.message
                elif isinstance(event, MessageUpdateEvent):
                    partial = event.message
                    self._state.stream_message = event.message
                elif isinstance(event, MessageEndEvent):
                    partial = None
                    self._state.stream_message = None
                    self.append_message(event.message)
                elif isinstance(event, ToolExecutionStartEvent):
                    self._state.pending_tool_calls = self._state.pending_tool_calls | {event.tool_call_id}
                elif isinstance(event, ToolExecutionEndEvent):
                    self._state.pending_tool_calls = self._state.pending_tool_calls - {event.tool_call_id}
                elif isinstance(event, TurnEndEvent):
                    if hasattr(event.message, "error_message") and event.message.error_message:
                        self._state.error = event.message.error_message
                elif isinstance(event, AgentEndEvent):
                    self._state.is_streaming = False
                    self._state.stream_message = None

                self._emit(event)

            # Handle any remaining partial message
            if partial is not None and hasattr(partial, "role") and partial.role == "assistant":
                if hasattr(partial, "content") and partial.content:
                    has_non_empty = any(
                        (hasattr(c, "thinking") and c.thinking.strip())
                        or (hasattr(c, "text") and c.text.strip())
                        or (c.type == "toolCall" and hasattr(c, "name") and c.name.strip())
                        for c in partial.content
                    )
                    if has_non_empty:
                        self.append_message(partial)
                    elif self._abort_event and self._abort_event.is_set():
                        raise RuntimeError("Request was aborted")

        except Exception as err:
            error_msg = AssistantMessage(
                role="assistant",
                content=[TextContent(text="")],
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=Usage(
                    input=0,
                    output=0,
                    cache_read=0,
                    cache_write=0,
                    total_tokens=0,
                    cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
                ),
                stop_reason="aborted" if (self._abort_event and self._abort_event.is_set()) else "error",
                error_message=str(err),
                timestamp=time.time(),
            )
            self.append_message(error_msg)
            self._state.error = str(err)
            self._emit(AgentEndEvent(messages=[error_msg]))
        finally:
            self._state.is_streaming = False
            self._state.stream_message = None
            self._state.pending_tool_calls = set()
            self._abort_event = None
            if self._running_future is not None and not self._running_future.done():
                self._running_future.set_result(None)
            self._running_future = None
