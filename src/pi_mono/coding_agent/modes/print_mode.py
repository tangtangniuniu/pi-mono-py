from __future__ import annotations

import json
import sys
from typing import Any

from rich.console import Console

from pi_mono.agent.types import (
    AgentEndEvent, AgentEvent, AgentStartEvent,
    MessageEndEvent, MessageStartEvent, MessageUpdateEvent,
    ToolExecutionEndEvent, ToolExecutionStartEvent, TurnEndEvent, TurnStartEvent,
)
from pi_mono.coding_agent.core.agent_session import AgentSession


class PrintMode:
    """Simple text/JSON output mode for non-interactive use."""

    def __init__(self, session: AgentSession, verbose: bool = False) -> None:
        self._session = session
        self._verbose = verbose
        self._console = Console(stderr=True)

    async def run(self, prompt: str) -> None:
        """Run a single prompt and print the result."""
        async for event in self._session.send_message(prompt):
            self._handle_event(event)

    def _handle_event(self, event: AgentEvent) -> None:
        if isinstance(event, MessageUpdateEvent):
            # Stream text deltas to stdout
            msg = event.message
            if hasattr(msg, 'content') and isinstance(msg.content, list):
                for content in msg.content:
                    if hasattr(content, 'text') and hasattr(content, 'type') and content.type == "text":
                        pass  # Text is accumulated, print at end

        elif isinstance(event, MessageEndEvent):
            msg = event.message
            if hasattr(msg, 'role') and msg.role == "assistant":
                if hasattr(msg, 'content') and isinstance(msg.content, list):
                    for content in msg.content:
                        if hasattr(content, 'text') and hasattr(content, 'type') and content.type == "text":
                            print(content.text, end="")
                    print()  # Final newline

        elif isinstance(event, ToolExecutionStartEvent) and self._verbose:
            self._console.print(f"[dim]Running tool: {event.tool_name}[/dim]")

        elif isinstance(event, ToolExecutionEndEvent) and self._verbose:
            status = "error" if event.is_error else "done"
            self._console.print(f"[dim]Tool {event.tool_name}: {status}[/dim]")
