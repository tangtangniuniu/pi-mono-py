from __future__ import annotations

import asyncio
import sys
from typing import Any

import click

from pi_mono import __version__


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="pi")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Pi — AI coding agent."""
    if ctx.invoked_subcommand is None:
        # Default to chat
        ctx.invoke(chat)


@cli.command()
@click.option("--model", "-m", help="LLM model to use.")
@click.option("--prompt", "-p", help="Initial prompt (non-interactive mode).")
@click.option("--session", "-s", help="Resume session by ID.")
@click.option("--print", "print_mode", is_flag=True, help="Print mode (no TUI).")
@click.option(
    "--thinking",
    type=click.Choice(["off", "minimal", "low", "medium", "high", "xhigh"]),
    default=None,
    help="Thinking/reasoning level.",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
def chat(
    model: str | None = None,
    prompt: str | None = None,
    session: str | None = None,
    print_mode: bool = False,
    thinking: str | None = None,
    verbose: bool = False,
) -> None:
    """Start an interactive chat session."""
    from pi_mono.coding_agent.main import run_chat

    asyncio.run(run_chat(
        model=model,
        prompt=prompt,
        session_id=session,
        print_mode=print_mode,
        thinking_level=thinking,
        verbose=verbose,
    ))


@cli.command()
def sessions() -> None:
    """List saved sessions."""
    from pi_mono.coding_agent.main import run_list_sessions
    asyncio.run(run_list_sessions())


@cli.command()
@click.option("--provider", "-p", help="Filter by provider.")
@click.option("--search", "-s", help="Search query.")
def models(provider: str | None = None, search: str | None = None) -> None:
    """List available models."""
    from pi_mono.coding_agent.main import run_list_models
    asyncio.run(run_list_models(provider=provider, search=search))
