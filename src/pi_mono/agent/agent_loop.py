"""Core agent loop that processes messages, streams LLM responses, and executes tools.

Mirrors the TypeScript ``agent-loop.ts`` from ``packages/agent/src/agent-loop.ts``.
The loop works with AgentMessage throughout and only transforms to Message[] at the
LLM call boundary via ``convert_to_llm``.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import TYPE_CHECKING, Any, Callable

from pi_mono.ai.stream import stream_simple as default_stream_fn
from pi_mono.ai.types import (
    Context,
    TextContent,
    ToolResultMessage,
)
from pi_mono.ai.utils.event_stream import EventStream
from pi_mono.ai.utils.validation import validate_tool_arguments
from pi_mono.agent.types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentStartEvent,
    AgentTool,
    AgentToolResult,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)

if TYPE_CHECKING:
    from pi_mono.ai.types import AssistantMessage, AssistantMessageEvent, Tool
    from pi_mono.ai.utils.event_stream import AssistantMessageEventStream

# Type alias for the stream function signature.
# Matches ``stream_simple(model, context, options) -> AssistantMessageEventStream``.
StreamFn = Callable[..., Any]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    abort_event: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """Start an agent loop with new prompt messages.

    The prompts are added to the context and lifecycle events are emitted for them.

    Parameters
    ----------
    prompts:
        New messages to add to the conversation.
    context:
        The current agent context (system prompt, messages, tools).
    config:
        Agent loop configuration including model and callbacks.
    abort_event:
        Optional event to signal abort.
    stream_fn:
        Optional override for the LLM stream function.

    Returns
    -------
    EventStream[AgentEvent, list[AgentMessage]]
        Async-iterable event stream whose final result is all new messages.
    """
    event_stream = _create_agent_stream()

    async def _run() -> None:
        new_messages: list[AgentMessage] = list(prompts)
        current_context = AgentContext(
            system_prompt=context.system_prompt,
            messages=[*context.messages, *prompts],
            tools=context.tools,
        )

        event_stream.push(AgentStartEvent())
        event_stream.push(TurnStartEvent())
        for prompt in prompts:
            event_stream.push(MessageStartEvent(message=prompt))
            event_stream.push(MessageEndEvent(message=prompt))

        await _run_loop(current_context, new_messages, config, abort_event, event_stream, stream_fn)

    asyncio.get_running_loop().create_task(_run())
    return event_stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    abort_event: asyncio.Event | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """Continue an agent loop from the current context without adding new messages.

    Used for retries -- the context already has user messages or tool results.

    **Important:** The last message in context must convert to a ``user`` or
    ``toolResult`` message via ``convert_to_llm``.  If it doesn't the LLM
    provider will reject the request.

    Parameters
    ----------
    context:
        The current agent context (must have at least one message).
    config:
        Agent loop configuration.
    abort_event:
        Optional abort signal.
    stream_fn:
        Optional override for the LLM stream function.

    Returns
    -------
    EventStream[AgentEvent, list[AgentMessage]]
        Async-iterable event stream.

    Raises
    ------
    ValueError
        If context has no messages or the last message is an assistant message.
    """
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")

    last_msg = context.messages[-1]
    if hasattr(last_msg, "role") and last_msg.role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    event_stream = _create_agent_stream()

    async def _run() -> None:
        new_messages: list[AgentMessage] = []
        current_context = AgentContext(
            system_prompt=context.system_prompt,
            messages=list(context.messages),
            tools=context.tools,
        )

        event_stream.push(AgentStartEvent())
        event_stream.push(TurnStartEvent())

        await _run_loop(current_context, new_messages, config, abort_event, event_stream, stream_fn)

    asyncio.get_running_loop().create_task(_run())
    return event_stream


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _create_agent_stream() -> EventStream[AgentEvent, list[AgentMessage]]:
    """Create an ``EventStream`` configured for the agent lifecycle."""
    return EventStream(
        is_complete=lambda event: isinstance(event, AgentEndEvent),
        extract_result=lambda event: event.messages if isinstance(event, AgentEndEvent) else [],
    )


async def _run_loop(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    abort_event: asyncio.Event | None,
    stream: EventStream[AgentEvent, list[AgentMessage]],
    stream_fn: StreamFn | None,
) -> None:
    """Main loop logic shared by :func:`agent_loop` and :func:`agent_loop_continue`."""
    first_turn = True

    # Check for steering messages at start (user may have typed while waiting)
    pending_messages: list[AgentMessage] = await _get_steering(config) if config.get_steering_messages else []

    # Outer loop: continues when queued follow-up messages arrive after agent would stop
    while True:
        has_more_tool_calls = True
        steering_after_tools: list[AgentMessage] | None = None

        # Inner loop: process tool calls and steering messages
        while has_more_tool_calls or pending_messages:
            if not first_turn:
                stream.push(TurnStartEvent())
            else:
                first_turn = False

            # Process pending messages (inject before next assistant response)
            if pending_messages:
                for message in pending_messages:
                    stream.push(MessageStartEvent(message=message))
                    stream.push(MessageEndEvent(message=message))
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            # Stream assistant response
            message = await _stream_assistant_response(current_context, config, abort_event, stream, stream_fn)
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                stream.push(TurnEndEvent(message=message, tool_results=[]))
                stream.push(AgentEndEvent(messages=new_messages))
                stream.end(new_messages)
                return

            # Check for tool calls
            tool_calls = [c for c in message.content if c.type == "toolCall"]
            has_more_tool_calls = len(tool_calls) > 0

            tool_results: list[ToolResultMessage] = []
            if has_more_tool_calls:
                tool_execution = await _execute_tool_calls(
                    current_context.tools,
                    message,
                    abort_event,
                    stream,
                    config.get_steering_messages,
                )
                tool_results.extend(tool_execution["tool_results"])
                steering_after_tools = tool_execution.get("steering_messages")

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            stream.push(TurnEndEvent(message=message, tool_results=tool_results))

            # Get steering messages after turn completes
            if steering_after_tools and len(steering_after_tools) > 0:
                pending_messages = steering_after_tools
                steering_after_tools = None
            else:
                pending_messages = await _get_steering(config) if config.get_steering_messages else []

        # Agent would stop here.  Check for follow-up messages.
        follow_up_messages = await _get_follow_up(config) if config.get_follow_up_messages else []
        if follow_up_messages:
            # Set as pending so inner loop processes them
            pending_messages = follow_up_messages
            continue

        # No more messages, exit
        break

    stream.push(AgentEndEvent(messages=new_messages))
    stream.end(new_messages)


async def _get_steering(config: AgentLoopConfig) -> list[AgentMessage]:
    """Call the steering message callback if configured."""
    if config.get_steering_messages is None:
        return []
    result = config.get_steering_messages()
    if inspect.isawaitable(result):
        return await result
    return result  # type: ignore[return-value]


async def _get_follow_up(config: AgentLoopConfig) -> list[AgentMessage]:
    """Call the follow-up message callback if configured."""
    if config.get_follow_up_messages is None:
        return []
    result = config.get_follow_up_messages()
    if inspect.isawaitable(result):
        return await result
    return result  # type: ignore[return-value]


async def _resolve_convert_to_llm(
    config: AgentLoopConfig,
    messages: list[AgentMessage],
) -> list:
    """Call convert_to_llm, handling both sync and async implementations."""
    result = config.convert_to_llm(messages)
    if inspect.isawaitable(result):
        return await result
    return result  # type: ignore[return-value]


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    abort_event: asyncio.Event | None,
    stream: EventStream[AgentEvent, list[AgentMessage]],
    stream_fn: StreamFn | None,
) -> AssistantMessage:
    """Stream an assistant response from the LLM.

    This is where ``AgentMessage[]`` is transformed to ``Message[]`` for the LLM.
    """
    from pi_mono.ai.types import AssistantMessage, SimpleStreamOptions, ThinkingBudgets, Tool

    # Apply context transform if configured (AgentMessage[] -> AgentMessage[])
    messages = context.messages
    if config.transform_context is not None:
        messages = await config.transform_context(messages, abort_event)

    # Convert to LLM-compatible messages (AgentMessage[] -> Message[])
    llm_messages = await _resolve_convert_to_llm(config, messages)

    # Convert AgentTool list to Tool list for LLM context
    llm_tools: list[Tool] | None = None
    if context.tools:
        llm_tools = [
            Tool(name=t.name, description=t.description, parameters=t.parameters)
            for t in context.tools
        ]

    # Build LLM context
    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=llm_tools,
    )

    fn = stream_fn or default_stream_fn

    # Resolve API key (important for expiring tokens)
    resolved_api_key: str | None = None
    if config.get_api_key is not None:
        key_result = config.get_api_key(config.model.provider)
        if inspect.isawaitable(key_result):
            resolved_api_key = await key_result
        else:
            resolved_api_key = key_result  # type: ignore[assignment]
    if resolved_api_key is None:
        resolved_api_key = config.api_key

    # Build thinking budgets
    thinking_budgets: ThinkingBudgets | None = None
    if config.thinking_budgets:
        thinking_budgets = ThinkingBudgets(
            minimal=config.thinking_budgets.get("minimal"),
            low=config.thinking_budgets.get("low"),
            medium=config.thinking_budgets.get("medium"),
            high=config.thinking_budgets.get("high"),
        )

    options = SimpleStreamOptions(
        reasoning=config.reasoning,
        thinking_budgets=thinking_budgets,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        abort_event=abort_event,
        api_key=resolved_api_key,
        transport=config.transport,
        cache_retention=config.cache_retention,
        session_id=config.session_id,
        headers=config.headers,
        max_retry_delay_ms=config.max_retry_delay_ms,
        metadata=config.metadata,
    )

    response = fn(config.model, llm_context, options)

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in response:
        if event.type == "start":
            partial_message = event.partial
            context.messages.append(partial_message)
            added_partial = True
            stream.push(MessageStartEvent(message=partial_message))

        elif event.type in (
            "text_start", "text_delta", "text_end",
            "thinking_start", "thinking_delta", "thinking_end",
            "toolcall_start", "toolcall_delta", "toolcall_end",
        ):
            if partial_message is not None:
                partial_message = event.partial
                context.messages[-1] = partial_message
                stream.push(MessageUpdateEvent(
                    assistant_message_event=event,
                    message=partial_message,
                ))

        elif event.type in ("done", "error"):
            final_message: AssistantMessage = await response.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
            if not added_partial:
                stream.push(MessageStartEvent(message=final_message))
            stream.push(MessageEndEvent(message=final_message))
            return final_message

    # Fallback: if we exhausted the iterator without a done/error event
    return await response.result()


async def _execute_tool_calls(
    tools: list[AgentTool] | None,
    assistant_message: AssistantMessage,
    abort_event: asyncio.Event | None,
    stream: EventStream[AgentEvent, list[AgentMessage]],
    get_steering_messages: object | None = None,
) -> dict:
    """Execute tool calls from an assistant message.

    Returns a dict with ``tool_results`` and optionally ``steering_messages``.
    """
    tool_calls = [c for c in assistant_message.content if c.type == "toolCall"]
    results: list[ToolResultMessage] = []
    steering_messages: list[AgentMessage] | None = None

    for index, tool_call in enumerate(tool_calls):
        tool = next((t for t in (tools or []) if t.name == tool_call.name), None)

        stream.push(ToolExecutionStartEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=tool_call.arguments,
        ))

        result: AgentToolResult
        is_error = False

        try:
            if tool is None:
                raise LookupError(f"Tool {tool_call.name} not found")

            # Build a Tool for validation
            from pi_mono.ai.types import Tool as LLMTool
            llm_tool = LLMTool(name=tool.name, description=tool.description, parameters=tool.parameters)
            validated_args = validate_tool_arguments(llm_tool, tool_call)

            def _on_update(partial_result: AgentToolResult) -> None:
                stream.push(ToolExecutionUpdateEvent(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    args=tool_call.arguments,
                    partial_result=partial_result,
                ))

            result = await tool.execute(tool_call.id, tool_call.name, validated_args, abort_event, _on_update)
        except Exception as exc:
            result = AgentToolResult(
                content=[TextContent(text=str(exc))],
            )
            is_error = True

        stream.push(ToolExecutionEndEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=result,
            is_error=is_error,
        ))

        tool_result_message = ToolResultMessage(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=result.content,
            details=result.details,
            is_error=is_error,
            timestamp=time.time(),
        )

        results.append(tool_result_message)
        stream.push(MessageStartEvent(message=tool_result_message))
        stream.push(MessageEndEvent(message=tool_result_message))

        # Check for steering messages -- skip remaining tools if user interrupted
        if get_steering_messages is not None and callable(get_steering_messages):
            steering_result = get_steering_messages()
            if inspect.isawaitable(steering_result):
                steering_result = await steering_result
            if steering_result:
                steering_messages = steering_result
                remaining_calls = tool_calls[index + 1 :]
                for skipped in remaining_calls:
                    results.append(_skip_tool_call(skipped, stream))
                break

    return {"tool_results": results, "steering_messages": steering_messages}


def _skip_tool_call(
    tool_call: object,
    stream: EventStream[AgentEvent, list[AgentMessage]],
) -> ToolResultMessage:
    """Create a skipped tool result for a tool call that was not executed."""
    result = AgentToolResult(
        content=[TextContent(text="Skipped due to queued user message.")],
    )

    stream.push(ToolExecutionStartEvent(
        tool_call_id=tool_call.id,  # type: ignore[attr-defined]
        tool_name=tool_call.name,  # type: ignore[attr-defined]
        args=tool_call.arguments,  # type: ignore[attr-defined]
    ))
    stream.push(ToolExecutionEndEvent(
        tool_call_id=tool_call.id,  # type: ignore[attr-defined]
        tool_name=tool_call.name,  # type: ignore[attr-defined]
        result=result,
        is_error=True,
    ))

    tool_result_message = ToolResultMessage(
        tool_call_id=tool_call.id,  # type: ignore[attr-defined]
        tool_name=tool_call.name,  # type: ignore[attr-defined]
        content=result.content,
        details=None,
        is_error=True,
        timestamp=time.time(),
    )

    stream.push(MessageStartEvent(message=tool_result_message))
    stream.push(MessageEndEvent(message=tool_result_message))

    return tool_result_message
