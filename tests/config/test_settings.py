"""Tests for pi_mono.config.settings — Settings Pydantic models."""

from __future__ import annotations

import pytest

from pi_mono.config.settings import (
    CustomModelConfig,
    ExtensionsConfig,
    ModelsConfig,
    ServerConfig,
    Settings,
)


class TestServerConfig:
    def test_defaults(self) -> None:
        cfg = ServerConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8080
        assert cfg.cors_origins == ["*"]

    def test_custom_values(self) -> None:
        cfg = ServerConfig(host="0.0.0.0", port=3000, cors_origins=["http://localhost:3000"])
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 3000


class TestModelsConfig:
    def test_defaults(self) -> None:
        cfg = ModelsConfig()
        assert cfg.default == "claude-sonnet-4-20250514"
        assert cfg.thinking_level == "off"
        assert cfg.max_turns == 100
        assert cfg.custom == []

    def test_custom_models(self) -> None:
        custom = CustomModelConfig(
            id="local-llama",
            provider="ollama",
            base_url="http://localhost:11434/v1",
        )
        cfg = ModelsConfig(custom=[custom])
        assert len(cfg.custom) == 1
        assert cfg.custom[0].id == "local-llama"


class TestExtensionsConfig:
    def test_defaults(self) -> None:
        cfg = ExtensionsConfig()
        assert cfg.enabled is True
        assert cfg.directories == []


class TestSettings:
    def test_defaults(self) -> None:
        s = Settings()
        assert isinstance(s.server, ServerConfig)
        assert isinstance(s.models, ModelsConfig)
        assert isinstance(s.extensions, ExtensionsConfig)
        assert s.auto_compact is True
        assert s.compact_threshold == 80_000
        assert s.theme == "default"
        assert s.verbose is False

    def test_nested_override(self) -> None:
        s = Settings(server=ServerConfig(port=9999))
        assert s.server.port == 9999

    def test_from_dict(self) -> None:
        data = {
            "server": {"port": 4000},
            "models": {"default": "gpt-4o", "thinking_level": "high"},
            "verbose": True,
        }
        s = Settings.model_validate(data)
        assert s.server.port == 4000
        assert s.models.default == "gpt-4o"
        assert s.verbose is True

    def test_unknown_fields_ignored(self) -> None:
        data = {"unknown_field": "value", "verbose": True}
        s = Settings.model_validate(data)
        assert s.verbose is True

    def test_invalid_type_raises(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            Settings.model_validate({"models": {"max_turns": "not_a_number"}})

    def test_immutable_copy(self) -> None:
        s1 = Settings()
        s2 = s1.model_copy(update={"verbose": True})
        assert s1.verbose is False
        assert s2.verbose is True
