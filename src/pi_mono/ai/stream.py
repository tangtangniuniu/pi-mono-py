"""Unified streaming interface for the AI API.

Mirrors the TypeScript ``stream.ts`` module from
``packages/ai/src/stream.ts``.  All functions resolve a provider from the
registry and delegate to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pi_mono.ai.api_registry import get_api_provider
from pi_mono.ai.utils.event_stream import AssistantMessageEventStream

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pi_mono.ai.types import (
        AssistantMessage,
        AssistantMessageEvent,
        Context,
        Model,
        SimpleStreamOptions,
        StreamOptions,
    )


def _resolve_api_provider(api: str) -> object:
    """Look up the registered provider for *api* or raise."""
    # Import here to use the LLMProvider type only for the error message;
    # the registry returns the concrete provider instance.
    provider = get_api_provider(api)
    if provider is None:
        raise LookupError(f"No API provider registered for api: {api!r}")
    return provider


def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream assistant-message events for a model using provider-specific options.

    The returned :class:`AssistantMessageEventStream` can be consumed via
    ``async for`` and also exposes a :meth:`result` future for the final
    :class:`AssistantMessage`.

    Parameters
    ----------
    model:
        The resolved model definition.
    context:
        The full conversation context.
    options:
        Optional streaming configuration.

    Returns
    -------
    AssistantMessageEventStream
        An async-iterable event stream.

    Raises
    ------
    LookupError
        If no provider is registered for ``model.api``.
    """
    provider = _resolve_api_provider(model.api)
    event_stream = AssistantMessageEventStream()
    source: AsyncIterator[AssistantMessageEvent] = provider.stream(  # type: ignore[union-attr]
        model, context, options,
    )
    _pipe_iterator_to_stream(source, event_stream)
    return event_stream


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Stream assistant-message events using simplified options.

    Same as :func:`stream` but accepts :class:`SimpleStreamOptions` which
    includes reasoning-level controls.

    Parameters
    ----------
    model:
        The resolved model definition.
    context:
        The full conversation context.
    options:
        Optional simplified streaming configuration.

    Returns
    -------
    AssistantMessageEventStream
        An async-iterable event stream.

    Raises
    ------
    LookupError
        If no provider is registered for ``model.api``.
    """
    provider = _resolve_api_provider(model.api)
    event_stream = AssistantMessageEventStream()
    source: AsyncIterator[AssistantMessageEvent] = provider.stream_simple(  # type: ignore[union-attr]
        model, context, options,
    )
    _pipe_iterator_to_stream(source, event_stream)
    return event_stream


async def complete(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessage:
    """Run a full completion and return the final :class:`AssistantMessage`.

    Internally streams the response and awaits the terminal result.

    Parameters
    ----------
    model:
        The resolved model definition.
    context:
        The full conversation context.
    options:
        Optional streaming configuration.

    Returns
    -------
    AssistantMessage
        The fully assembled assistant message.

    Raises
    ------
    LookupError
        If no provider is registered for ``model.api``.
    """
    s = stream(model, context, options)
    return await s.result()


async def complete_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessage:
    """Run a simplified completion and return the final :class:`AssistantMessage`.

    Same as :func:`complete` but accepts :class:`SimpleStreamOptions`.

    Parameters
    ----------
    model:
        The resolved model definition.
    context:
        The full conversation context.
    options:
        Optional simplified streaming configuration.

    Returns
    -------
    AssistantMessage
        The fully assembled assistant message.

    Raises
    ------
    LookupError
        If no provider is registered for ``model.api``.
    """
    s = stream_simple(model, context, options)
    return await s.result()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pipe_iterator_to_stream(
    source: AsyncIterator[AssistantMessageEvent],
    target: AssistantMessageEventStream,
) -> None:
    """Spawn a background task that pipes *source* events into *target*.

    If the provider's ``stream`` / ``stream_simple`` returns an async
    iterator, we need to drive it in the background so that the caller
    receives an :class:`AssistantMessageEventStream` immediately (matching
    the synchronous-return pattern of the TypeScript implementation).
    """
    import asyncio

    async def _drain() -> None:
        try:
            async for event in source:
                target.push(event)
        except Exception as exc:
            # If the source raises, synthesise an error event so consumers
            # always see a terminal event.
            from pi_mono.ai.types import (
                AssistantMessage as _AM,
            )
            from pi_mono.ai.types import (
                Cost as _Cost,
            )
            from pi_mono.ai.types import (
                ErrorEvent as _EE,
            )
            from pi_mono.ai.types import (
                Usage as _Usage,
            )

            error_msg = _AM(
                content=[],
                api="",
                provider="",
                model="",
                usage=_Usage(
                    input=0,
                    output=0,
                    cache_read=0,
                    cache_write=0,
                    total_tokens=0,
                    cost=_Cost(
                        input=0.0,
                        output=0.0,
                        cache_read=0.0,
                        cache_write=0.0,
                        total=0.0,
                    ),
                ),
                stop_reason="error",
                timestamp=0.0,
                error_message=str(exc),
            )
            error_event = _EE(reason="error", error=error_msg)
            target.push(error_event)
        finally:
            target.end()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop -- cannot schedule the drain task.  This
        # shouldn't happen in normal usage since callers are async.
        raise RuntimeError(
            "stream() must be called from within a running asyncio event loop"
        ) from None

    _task = loop.create_task(_drain())  # noqa: RUF006
