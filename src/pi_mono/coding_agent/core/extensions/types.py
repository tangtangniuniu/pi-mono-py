"""ExtensionAPI — the interface extensions use to register capabilities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pi_mono.agent.types import AgentTool


class ExtensionAPI:
    """API object passed to extension's create_extension(api) function.

    Extensions use this to register tools, prompts, and event handlers.
    """

    def __init__(self, source_id: str) -> None:
        self._source_id = source_id
        self._tools: list[AgentTool] = []
        self._prompts: dict[str, str] = {}
        self._event_handlers: dict[str, list[Callable[..., Any]]] = defaultdict(list)

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def tools(self) -> list[AgentTool]:
        return list(self._tools)

    @property
    def prompts(self) -> dict[str, str]:
        return dict(self._prompts)

    @property
    def event_handlers(self) -> dict[str, list[Callable[..., Any]]]:
        return dict(self._event_handlers)

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        execute: Callable[..., Any],
    ) -> None:
        """Register a custom tool that will be available to the Agent."""
        tool = AgentTool(
            name=name,
            description=description,
            label=name.replace("_", " ").title(),
            parameters=parameters,
            execute=execute,
        )
        self._tools.append(tool)

    def register_prompt(self, name: str, path: str) -> None:
        """Register a prompt template from a file path."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        self._prompts[name] = p.read_text(encoding="utf-8")

    def on_message_end(self, callback: Callable[..., Any]) -> None:
        """Subscribe to message_end events."""
        self._event_handlers["message_end"].append(callback)

    def on_turn_end(self, callback: Callable[..., Any]) -> None:
        """Subscribe to turn_end events."""
        self._event_handlers["turn_end"].append(callback)
