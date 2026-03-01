"""Settings Pydantic models for pi-mono configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """REST server configuration."""

    host: str = "127.0.0.1"
    port: int = 8080
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    model_config = {"extra": "ignore"}


class CustomModelConfig(BaseModel):
    """Custom model entry in settings."""

    id: str
    provider: str
    base_url: str
    name: str | None = None
    api: str = "openai-completions"
    context_window: int = 128_000
    max_tokens: int = 4096

    model_config = {"extra": "ignore"}


class ModelsConfig(BaseModel):
    """Model selection and custom model definitions."""

    default: str = "claude-sonnet-4-20250514"
    thinking_level: str = "off"
    max_turns: int = 100
    custom: list[CustomModelConfig] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


class ExtensionsConfig(BaseModel):
    """Extension system configuration."""

    enabled: bool = True
    directories: list[str] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


class Settings(BaseModel):
    """Top-level settings model — the single source of truth for configuration."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    extensions: ExtensionsConfig = Field(default_factory=ExtensionsConfig)

    auto_compact: bool = True
    compact_threshold: int = 80_000
    theme: str = "default"
    verbose: bool = False

    model_config = {"extra": "ignore"}
