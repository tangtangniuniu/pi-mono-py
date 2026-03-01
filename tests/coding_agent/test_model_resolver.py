"""Tests for pi_mono.coding_agent.core.model_resolver."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pi_mono.coding_agent.core.model_registry import ModelRegistry
from pi_mono.coding_agent.core.model_resolver import ModelResolver
from pi_mono.coding_agent.core.settings_manager import Settings
from tests.conftest import make_model


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry()


@pytest.fixture
def resolver(registry: ModelRegistry) -> ModelResolver:
    return ModelResolver(registry)


class TestResolve:
    def test_requested_model_takes_priority(self, registry: ModelRegistry, resolver: ModelResolver) -> None:
        model = make_model(model_id="requested-model")
        registry.register_custom_model(model)
        settings = Settings(default_model="different-model")

        result = resolver.resolve("requested-model", settings)
        assert result.id == "requested-model"

    def test_falls_back_to_settings_default(self, registry: ModelRegistry, resolver: ModelResolver) -> None:
        model = make_model(model_id="settings-default")
        registry.register_custom_model(model)
        settings = Settings(default_model="settings-default")

        result = resolver.resolve(None, settings)
        assert result.id == "settings-default"

    @patch("pi_mono.coding_agent.core.model_resolver.DEFAULT_MODEL", "fallback-model")
    def test_falls_back_to_default_constant(self, registry: ModelRegistry, resolver: ModelResolver) -> None:
        model = make_model(model_id="fallback-model")
        registry.register_custom_model(model)
        settings = Settings(default_model="")

        result = resolver.resolve(None, settings)
        assert result.id == "fallback-model"

    def test_not_found_raises_value_error(self, resolver: ModelResolver) -> None:
        settings = Settings(default_model="nonexistent")
        with pytest.raises(ValueError, match="Model not found"):
            resolver.resolve(None, settings)

    @patch("pi_mono.coding_agent.core.model_registry.get_model")
    @patch("pi_mono.coding_agent.core.model_registry.get_providers", return_value=[])
    def test_tries_provider_prefixes(self, mock_providers, mock_get_model, registry: ModelRegistry) -> None:
        """When direct lookup fails, tries with common provider prefixes."""
        model = make_model(model_id="gpt-4", provider="openai")
        # First call (direct) returns None; second call (with provider prefix) returns model
        mock_get_model.side_effect = [None, None, model]

        resolver = ModelResolver(registry)
        settings = Settings(default_model="gpt-4")
        result = resolver.resolve(None, settings)
        assert result.id == "gpt-4"
