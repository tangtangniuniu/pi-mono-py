from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pi_mono.agent.types import AgentTool
from pi_mono.coding_agent.config import EXTENSIONS_DIR


class Extension(Protocol):
    """Extension protocol."""
    name: str

    def get_tools(self) -> list[AgentTool]: ...


@dataclass
class LoadedExtension:
    """A loaded extension."""
    name: str
    tools: list[AgentTool]


class ExtensionLoader:
    """Loads Python extensions from ~/.pi/extensions/."""

    def __init__(self, extensions_dir: Path | None = None) -> None:
        self._dir = extensions_dir or EXTENSIONS_DIR

    def load_all(self) -> list[LoadedExtension]:
        extensions: list[LoadedExtension] = []
        if not self._dir.exists():
            return extensions

        import importlib.util

        for path in sorted(self._dir.glob("*.py")):
            try:
                spec = importlib.util.spec_from_file_location(path.stem, path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if hasattr(module, "create_extension"):
                        ext = module.create_extension()
                        tools = ext.get_tools() if hasattr(ext, 'get_tools') else []
                        extensions.append(LoadedExtension(
                            name=getattr(ext, 'name', path.stem),
                            tools=tools,
                        ))
            except Exception:
                continue

        return extensions
