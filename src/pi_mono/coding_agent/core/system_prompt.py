from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any

from pi_mono.agent.types import AgentTool
from pi_mono.ai.types import Tool


def build_system_prompt(
    tools: list[AgentTool] | None = None,
    context: dict[str, str] | None = None,
) -> str:
    """Build the system prompt for the coding agent."""
    parts: list[str] = []

    parts.append("You are an AI coding assistant. You help users with software engineering tasks.")
    parts.append("")

    # Environment info
    parts.append("# Environment")
    cwd = (context or {}).get("cwd", os.getcwd())
    parts.append(f"Working directory: {cwd}")
    parts.append(f"Platform: {platform.system().lower()}")
    parts.append(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")

    # Git info
    git_branch = (context or {}).get("git_branch")
    if git_branch:
        parts.append(f"Git branch: {git_branch}")

    is_git = (context or {}).get("is_git_repo")
    if is_git:
        parts.append(f"Git repo: {is_git}")

    parts.append("")

    # Tool descriptions
    if tools:
        parts.append("# Available Tools")
        parts.append("")
        for tool in tools:
            parts.append(f"## {tool.name}")
            parts.append(tool.description)
            parts.append("")

    # Guidelines
    parts.append("# Guidelines")
    parts.append("- Read files before modifying them")
    parts.append("- Prefer editing existing files over creating new ones")
    parts.append("- Write clean, readable code")
    parts.append("- Handle errors explicitly")
    parts.append("- Keep changes minimal and focused")
    parts.append("")

    return "\n".join(parts)


def get_git_context() -> dict[str, str]:
    """Get git context for the current directory."""
    context: dict[str, str] = {}

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=5,
        )
        context["is_git_repo"] = "yes" if result.returncode == 0 else "no"
    except Exception:
        context["is_git_repo"] = "no"

    if context.get("is_git_repo") == "yes":
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                context["git_branch"] = result.stdout.strip()
        except Exception:
            pass

    return context
