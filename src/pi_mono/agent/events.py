from __future__ import annotations

from typing import Callable, Awaitable
import asyncio

from pi_mono.agent.types import AgentEvent


class EventBus:
    """Async event bus for agent lifecycle events."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[AgentEvent], Awaitable[None] | None]] = []

    def subscribe(self, callback: Callable[[AgentEvent], Awaitable[None] | None]) -> Callable[[], None]:
        """Subscribe to events. Returns unsubscribe function."""
        self._listeners.append(callback)
        def unsubscribe() -> None:
            if callback in self._listeners:
                self._listeners.remove(callback)
        return unsubscribe

    async def emit(self, event: AgentEvent) -> None:
        """Emit an event to all listeners."""
        for listener in self._listeners:
            result = listener(event)
            if asyncio.iscoroutine(result):
                await result

    def clear(self) -> None:
        """Remove all listeners."""
        self._listeners.clear()
