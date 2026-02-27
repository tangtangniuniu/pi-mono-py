from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from pi_mono.agent.types import (
    AgentEndEvent, AgentEvent, AgentStartEvent,
    MessageEndEvent, MessageStartEvent, MessageUpdateEvent,
    ToolExecutionEndEvent, ToolExecutionStartEvent,
    ToolExecutionUpdateEvent, TurnEndEvent, TurnStartEvent,
)
from pi_mono.coding_agent.core.agent_session import AgentSession


class InteractiveMode:
    """Interactive terminal mode using rich."""

    def __init__(self, session: AgentSession, verbose: bool = False) -> None:
        self._session = session
        self._verbose = verbose
        self._console = Console()
        self._current_text = ""

    async def run(self) -> None:
        """Main REPL loop."""
        self._console.print(Panel(
            "[bold]Pi Coding Agent[/bold]\nType your message or /help for commands. Ctrl+C to exit.",
            border_style="blue",
        ))

        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]>[/bold blue]")
            except (KeyboardInterrupt, EOFError):
                self._console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input.strip():
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                handled = await self._handle_command(user_input)
                if handled:
                    continue

            # Send message to agent
            self._current_text = ""
            try:
                async for event in self._session.send_message(user_input):
                    self._handle_event(event)
            except Exception as e:
                self._console.print(f"[red]Error: {e}[/red]")

    async def _handle_command(self, command: str) -> bool:
        cmd = command.strip().lower()

        if cmd == "/help":
            self._console.print(Panel(
                "/help     — Show this help\n"
                "/model    — Show/switch model\n"
                "/session  — Show session info\n"
                "/compact  — Compact conversation\n"
                "/clear    — Clear conversation\n"
                "/exit     — Exit",
                title="Commands",
                border_style="cyan",
            ))
            return True

        elif cmd == "/exit" or cmd == "/quit":
            raise KeyboardInterrupt

        elif cmd == "/compact":
            self._console.print("[dim]Compacting conversation...[/dim]")
            await self._session.compact()
            self._console.print("[green]Conversation compacted.[/green]")
            return True

        elif cmd == "/clear":
            if self._session.agent:
                self._session.agent.reset()
            self._console.print("[green]Conversation cleared.[/green]")
            return True

        elif cmd.startswith("/model"):
            parts = cmd.split(maxsplit=1)
            if len(parts) > 1:
                try:
                    await self._session.switch_model(parts[1])
                    self._console.print(f"[green]Switched to {parts[1]}[/green]")
                except ValueError as e:
                    self._console.print(f"[red]{e}[/red]")
            else:
                if self._session.agent and self._session.agent.state.model:
                    self._console.print(f"Current model: {self._session.agent.state.model.id}")
            return True

        return False

    def _handle_event(self, event: AgentEvent) -> None:
        if isinstance(event, MessageUpdateEvent):
            msg = event.message
            if hasattr(msg, 'content') and isinstance(msg.content, list):
                # Print streaming text
                ae = event.assistant_message_event
                if hasattr(ae, 'type') and ae.type == "text_delta" and hasattr(ae, 'delta'):
                    self._console.print(ae.delta, end="")
                    self._current_text += ae.delta

        elif isinstance(event, MessageEndEvent):
            msg = event.message
            if hasattr(msg, 'role') and msg.role == "assistant":
                if self._current_text:
                    print()  # Newline after streaming
                    self._current_text = ""

        elif isinstance(event, ToolExecutionStartEvent):
            self._console.print(f"\n[yellow]⚡ {event.tool_name}[/yellow]", end="")
            if self._verbose:
                self._console.print(f" [dim]{event.args}[/dim]")
            else:
                print()

        elif isinstance(event, ToolExecutionEndEvent):
            if event.is_error:
                self._console.print(f"[red]  ✗ Error[/red]")
            elif self._verbose:
                self._console.print(f"[green]  ✓ Done[/green]")
