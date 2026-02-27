from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from pi_mono.ai.types import ThinkingLevel
from pi_mono.coding_agent.config import DEFAULT_COMPACT_THRESHOLD, DEFAULT_MAX_TURNS, DEFAULT_MODEL, SETTINGS_FILE


class Settings(BaseModel):
    """User configuration settings."""
    default_model: str = DEFAULT_MODEL
    thinking_level: str = "off"
    max_turns: int = DEFAULT_MAX_TURNS
    auto_compact: bool = True
    compact_threshold: int = DEFAULT_COMPACT_THRESHOLD
    theme: str = "default"
    verbose: bool = False
    # Provider-specific settings
    custom_api_urls: dict[str, str] = Field(default_factory=dict)
    custom_headers: dict[str, dict[str, str]] = Field(default_factory=dict)


class SettingsManager:
    """Manages user settings with JSON file persistence."""

    def __init__(self, settings_file: Path | None = None) -> None:
        self._settings_file = settings_file or SETTINGS_FILE
        self._settings: Settings | None = None

    async def load(self) -> Settings:
        if self._settings is not None:
            return self._settings

        if self._settings_file.exists():
            try:
                data = json.loads(self._settings_file.read_text(encoding="utf-8"))
                self._settings = Settings.model_validate(data)
            except Exception:
                self._settings = Settings()
        else:
            self._settings = Settings()

        return self._settings

    async def save(self, settings: Settings) -> None:
        self._settings = settings
        self._settings_file.parent.mkdir(parents=True, exist_ok=True)
        self._settings_file.write_text(
            settings.model_dump_json(indent=2),
            encoding="utf-8",
        )

    async def update(self, **kwargs: Any) -> Settings:
        settings = await self.load()
        updated = settings.model_copy(update=kwargs)
        await self.save(updated)
        return updated

    async def reset(self) -> Settings:
        settings = Settings()
        await self.save(settings)
        return settings
