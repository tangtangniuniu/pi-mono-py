"""Async event stream with queue-based iteration and final-result awaiting.

Mirrors the TypeScript ``EventStream`` class from
``packages/ai/src/utils/event-stream.ts``.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable, Generic, TypeVar

from pi_mono.ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
)

T = TypeVar("T")
R = TypeVar("R")


class EventStream(Generic[T, R]):
    """Async event stream that can be iterated and also awaited for a final result.

    The stream is fed via :meth:`push` and terminated via :meth:`end`.
    Consumers iterate with ``async for`` and can also call :meth:`result`
    to obtain a :class:`asyncio.Future` that resolves to the final result.

    Parameters
    ----------
    is_complete:
        Predicate that returns ``True`` when an event signals stream
        completion (e.g. a ``done`` or ``error`` event).
    extract_result:
        Callable that extracts the final result ``R`` from the terminal
        event ``T``.
    """

    def __init__(
        self,
        is_complete: Callable[[T], bool],
        extract_result: Callable[[T], R],
    ) -> None:
        self._is_complete = is_complete
        self._extract_result = extract_result

        self._queue: list[T] = []
        self._waiters: list[asyncio.Future[_IterResult[T]]] = []
        self._done: bool = False

        loop = _get_or_create_loop()
        self._final_result: asyncio.Future[R] = loop.create_future()

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    def push(self, event: T) -> None:
        """Add an event to the stream.

        If *event* satisfies the completion predicate the final result is
        resolved and no further events will be accepted.
        """
        if self._done:
            return

        if self._is_complete(event):
            self._done = True
            if not self._final_result.done():
                self._final_result.set_result(self._extract_result(event))

        # Deliver to a waiting consumer, or buffer the event.
        if self._waiters:
            waiter = self._waiters.pop(0)
            if not waiter.done():
                waiter.set_result(_IterResult(value=event, done=False))
        else:
            self._queue.append(event)

    def end(self, result: R | None = None) -> None:
        """Signal that no more events will be pushed.

        Optionally provide a *result* to resolve the final-result future
        (if it has not already been resolved by a terminal event).
        """
        self._done = True
        if result is not None and not self._final_result.done():
            self._final_result.set_result(result)

        # Wake all waiting consumers with a "done" sentinel.
        while self._waiters:
            waiter = self._waiters.pop(0)
            if not waiter.done():
                waiter.set_result(_IterResult(value=None, done=True))

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def result(self) -> asyncio.Future[R]:
        """Return a future that resolves to the final result.

        The future is resolved when a terminal event is pushed or
        :meth:`end` is called with a result value.
        """
        return self._final_result

    def __aiter__(self) -> AsyncIterator[T]:
        return self._async_iterator()

    async def _async_iterator(self) -> AsyncIterator[T]:
        """Internal async generator that yields queued events."""
        while True:
            if self._queue:
                yield self._queue.pop(0)
            elif self._done:
                return
            else:
                loop = asyncio.get_running_loop()
                waiter: asyncio.Future[_IterResult[T]] = loop.create_future()
                self._waiters.append(waiter)
                iter_result = await waiter
                if iter_result.done:
                    return
                # value is guaranteed non-None when done is False
                yield iter_result.value  # type: ignore[misc]


class _IterResult(Generic[T]):
    """Small value-holder mirroring JS ``IteratorResult``."""

    __slots__ = ("value", "done")

    def __init__(self, *, value: T | None, done: bool) -> None:
        self.value = value
        self.done = done


# ---------------------------------------------------------------------------
# AssistantMessageEventStream
# ---------------------------------------------------------------------------


class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    """Pre-configured event stream for assistant message events.

    Terminal events are ``done`` and ``error``.  The final result is the
    :class:`~pi_mono.ai.types.AssistantMessage` extracted from whichever
    terminal event arrives first.
    """

    def __init__(self) -> None:
        super().__init__(
            is_complete=_is_terminal_event,
            extract_result=_extract_assistant_message,
        )


def create_assistant_message_event_stream() -> AssistantMessageEventStream:
    """Factory function for :class:`AssistantMessageEventStream`.

    Intended for use in extensions that need to construct their own streams.
    """
    return AssistantMessageEventStream()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_terminal_event(event: AssistantMessageEvent) -> bool:
    return event.type in ("done", "error")


def _extract_assistant_message(event: AssistantMessageEvent) -> AssistantMessage:
    if event.type == "done":
        return event.message  # type: ignore[union-attr]
    if event.type == "error":
        return event.error  # type: ignore[union-attr]
    raise ValueError(f"Unexpected event type for final result: {event.type}")


def _get_or_create_loop() -> asyncio.AbstractEventLoop:
    """Return the running event loop, or create a new one if none exists.

    This is needed because ``EventStream`` instances may be constructed
    outside of an async context (e.g. at module level or in synchronous
    factory functions).
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
