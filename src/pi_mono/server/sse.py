"""SSE event stream — converts Agent events to SSE format."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from pi_mono.agent.types import AgentEvent


async def event_stream_generator(
    session_id: str,
    heartbeat_interval: float = 15.0,
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted events for a given session.

    Uses Agent.subscribe() to receive events and formats them as:
        event: <type>
        data: <json>
    """
    from pi_mono.server.app import get_session_registry
    registry = get_session_registry()
    session = registry.get(session_id)
    if session is None or session.agent is None:
        yield _sse_event("error", {"error": "Session not found"})
        return

    event_queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

    def on_event(event: AgentEvent) -> None:
        event_queue.put_nowait(event)

    unsub = session.agent.subscribe(on_event)

    try:
        while True:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=heartbeat_interval)
                if event is None:
                    break
                yield _sse_event(event.type, _serialize_event(event))
                if event.type == "agent_end":
                    break
            except TimeoutError:
                yield ": heartbeat\n\n"
    finally:
        unsub()


def _sse_event(event_type: str, data: Any) -> str:
    """Format a single SSE event."""
    json_data = json.dumps(data, default=str)
    return f"event: {event_type}\ndata: {json_data}\n\n"


def _serialize_event(event: AgentEvent) -> dict[str, Any]:
    """Serialize an AgentEvent to a JSON-compatible dict."""
    result: dict[str, Any] = {"type": event.type}
    for key, value in event.__dict__.items():
        if key.startswith("_"):
            continue
        if hasattr(value, "__dict__") and not isinstance(value, str):
            # Skip complex nested objects, just note their type
            result[key] = str(type(value).__name__)
        else:
            result[key] = value
    return result
