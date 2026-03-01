"""ExtensionLoader — dynamically loads Python extension modules."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pi_mono.coding_agent.config import EXTENSIONS_DIR
from pi_mono.coding_agent.core.extensions.types import ExtensionAPI

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LoadedExtension:
    """Result of loading a single extension module."""
    source_id: str
    api: ExtensionAPI


class ExtensionLoader:
    """Loads Python extensions from a directory.

    Each extension module must export a `create_extension(api: ExtensionAPI) -> None`
    function. The loader creates an ExtensionAPI for each module and calls
    create_extension with it.
    """

    def __init__(self, extensions_dir: Path | None = None) -> None:
        self._dir = extensions_dir or EXTENSIONS_DIR

    def load_all(self) -> list[LoadedExtension]:
        """Load all .py extensions from the configured directory."""
        results: list[LoadedExtension] = []

        if not self._dir.exists():
            return results

        for path in sorted(self._dir.glob("*.py")):
            try:
                spec = importlib.util.spec_from_file_location(path.stem, path)
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if not hasattr(module, "create_extension"):
                    logger.debug("Skipping %s: no create_extension function", path)
                    continue

                api = ExtensionAPI(source_id=path.stem)
                module.create_extension(api)
                results.append(LoadedExtension(source_id=path.stem, api=api))

            except Exception:
                logger.warning("Failed to load extension %s", path, exc_info=True)
                continue

        return results
