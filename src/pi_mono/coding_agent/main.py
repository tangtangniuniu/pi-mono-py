from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table

from pi_mono.coding_agent.config import ensure_dirs
from pi_mono.coding_agent.core.agent_session import AgentSession
from pi_mono.coding_agent.core.model_registry import ModelRegistry
from pi_mono.coding_agent.core.session_manager import SessionManager
from pi_mono.coding_agent.core.settings_manager import SettingsManager

console = Console()


def _create_session() -> AgentSession:
    """Create a new agent session with all dependencies."""
    settings_manager = SettingsManager()
    session_manager = SessionManager()
    model_registry = ModelRegistry()

    return AgentSession(
        settings_manager=settings_manager,
        session_manager=session_manager,
        model_registry=model_registry,
    )


async def run_chat(
    model: str | None = None,
    prompt: str | None = None,
    session_id: str | None = None,
    print_mode: bool = False,
    thinking_level: str | None = None,
    verbose: bool = False,
) -> None:
    """Run the chat command."""
    ensure_dirs()
    session = _create_session()

    try:
        await session.start(
            session_id=session_id,
            model=model,
            thinking_level=thinking_level,
        )

        if prompt:
            # Non-interactive: print mode
            from pi_mono.coding_agent.modes.print_mode import PrintMode
            mode = PrintMode(session, verbose=verbose)
            await mode.run(prompt)
        elif print_mode:
            # Read from stdin
            from pi_mono.coding_agent.modes.print_mode import PrintMode
            mode = PrintMode(session, verbose=verbose)
            stdin_content = sys.stdin.read()
            if stdin_content.strip():
                await mode.run(stdin_content)
        else:
            # Interactive mode
            from pi_mono.coding_agent.modes.interactive import InteractiveMode
            mode = InteractiveMode(session, verbose=verbose)
            await mode.run()
    finally:
        await session.close()


async def run_list_sessions() -> None:
    """List saved sessions."""
    ensure_dirs()
    manager = SessionManager()
    sessions = await manager.list_sessions()

    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Messages", justify="right")
    table.add_column("Model")

    for s in sessions:
        table.add_row(
            s.session_id,
            s.name or "",
            str(s.message_count),
            s.model or "",
        )

    console.print(table)


async def run_list_models(provider: str | None = None, search: str | None = None) -> None:
    """List available models."""
    registry = ModelRegistry()

    if search:
        models = registry.search_models(search)
    else:
        models = registry.list_models(provider)

    if not models:
        console.print("[dim]No models found.[/dim]")
        return

    table = Table(title="Models")
    table.add_column("ID", style="cyan")
    table.add_column("Provider")
    table.add_column("Name")
    table.add_column("Context", justify="right")

    for m in models:
        table.add_row(
            m.id,
            m.provider,
            m.name,
            f"{m.context_window:,}",
        )

    console.print(table)


def main() -> None:
    """CLI entry point."""
    from pi_mono.coding_agent.cli.args import cli
    cli()


if __name__ == "__main__":
    main()
