from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field

from pi_mono.ai.env_api_keys import register_env_api_key
from pi_mono.ai.providers import register_default_providers
from pi_mono.coding_agent.config import DEFAULT_COMPACT_THRESHOLD, DEFAULT_MAX_TURNS, DEFAULT_MODEL, SETTINGS_FILE
from pi_mono.config.dotenv import get_runtime_model_config, runtime_model_to_model

if TYPE_CHECKING:
    from pi_mono.ai.types import Model
    from pi_mono.coding_agent.core.model_registry import ModelRegistry


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
    runtime_model_id: str | None = None

    model_config = {"extra": "ignore"}


class SettingsManager:
    """Manages user settings with YAML file persistence."""

    def __init__(self, settings_file: Path | None = None, project_dir: Path | None = None) -> None:
        self._settings_file = settings_file or SETTINGS_FILE
        self._project_dir = project_dir
        self._settings: Settings | None = None
        self._runtime_model: Model | None = None

    def _load_runtime_model(self) -> Model | None:
        register_default_providers()
        runtime_config = get_runtime_model_config(self._project_dir)
        if runtime_config is None:
            self._runtime_model = None
            return None

        register_env_api_key(runtime_config.provider, runtime_config.api_key_env)
        self._runtime_model = runtime_model_to_model(runtime_config)
        return self._runtime_model

    @property
    def runtime_model(self) -> Model | None:
        return self._runtime_model

    def bootstrap_runtime_model(self, registry: ModelRegistry) -> Model | None:
        runtime_model = self._load_runtime_model()
        if runtime_model is not None:
            registry.register_custom_model(runtime_model)
        return runtime_model

    async def load(self) -> Settings:
        if self._settings is not None:
            return self._settings

        explicit_default = False
        if self._settings_file.exists():
            try:
                text = self._settings_file.read_text(encoding="utf-8")
                data = yaml.safe_load(text)
                if data is None:
                    data = {}
                explicit_default = "default_model" in data
                self._settings = Settings.model_validate(data)
            except Exception:
                self._settings = Settings()
        else:
            self._settings = Settings()

        runtime_model = self._load_runtime_model()
        if runtime_model is not None:
            runtime_model_id = runtime_model.id
            default_model = self._settings.default_model
            if explicit_default and default_model != DEFAULT_MODEL:
                self._settings = self._settings.model_copy(update={"runtime_model_id": runtime_model_id})
            else:
                self._settings = self._settings.model_copy(
                    update={"default_model": default_model, "runtime_model_id": runtime_model_id}
                )

        return self._settings

    async def save(self, settings: Settings) -> None:
        self._settings = settings
        self._settings_file.parent.mkdir(parents=True, exist_ok=True)
        data = settings.model_dump()
        self._settings_file.write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True),
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
