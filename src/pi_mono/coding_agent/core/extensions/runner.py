"""ExtensionRunner — manages extension lifecycle, tool injection, and event dispatch."""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pi_mono.agent.types import AgentTool
    from pi_mono.coding_agent.core.extensions.types import ExtensionAPI

logger = logging.getLogger(__name__)


class ExtensionRunner:
    """Manages loaded extensions: collects tools, dispatches events, supports unloading."""

    def __init__(self, builtin_tool_names: set[str] | None = None) -> None:
        self._extensions: dict[str, ExtensionAPI] = {}
        self._builtin_tool_names = builtin_tool_names or set()

    def add_extension(self, source_id: str, api: ExtensionAPI) -> None:
        """Register a loaded extension."""
        self._extensions[source_id] = api

    def remove_extension(self, source_id: str) -> None:
        """Unload an extension — removes all its tools and event handlers."""
        self._extensions.pop(source_id, None)

    def get_all_tools(self) -> list[AgentTool]:
        """Collect tools from all extensions, excluding name conflicts with built-ins."""
        tools: list[AgentTool] = []
        for source_id, api in self._extensions.items():
            for tool in api.tools:
                if tool.name in self._builtin_tool_names:
                    logger.warning(
                        "Extension %s tool '%s' conflicts with built-in — skipped",
                        source_id, tool.name,
                    )
                    continue
                tools.append(tool)
        return tools

    async def dispatch(self, event_type: str, event: Any) -> None:
        """Dispatch an event to all registered handlers of the given type."""
        for api in self._extensions.values():
            handlers = api.event_handlers.get(event_type, [])
            for handler in handlers:
                try:
                    if inspect.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception:
                    logger.warning(
                        "Error in event handler for %s", event_type, exc_info=True,
                    )
