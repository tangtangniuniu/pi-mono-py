"""Tests for pi_mono.coding_agent.core.model_registry."""

from __future__ import annotations

from unittest.mock import patch

from pi_mono.coding_agent.core.model_registry import ModelRegistry
from tests.conftest import make_model


class TestRegisterCustomModel:
    def test_register_and_retrieve(self) -> None:
        reg = ModelRegistry()
        model = make_model(model_id="custom-model")
        reg.register_custom_model(model)
        assert reg.get_model("custom-model") is model

    def test_custom_model_overrides_builtin(self) -> None:
        """Custom models take priority over built-in catalog."""
        custom = make_model(model_id="builtin-id", provider="custom")
        reg = ModelRegistry()
        reg.register_custom_model(custom)
        result = reg.get_model("builtin-id")
        assert result is custom


class TestGetModel:
    def test_get_nonexistent_returns_none(self) -> None:
        reg = ModelRegistry()
        assert reg.get_model("nonexistent") is None

    @patch("pi_mono.coding_agent.core.model_registry.get_providers", return_value=["openai"])
    @patch("pi_mono.coding_agent.core.model_registry.get_model")
    def test_get_searches_all_providers(self, mock_get_model, mock_providers) -> None:
        builtin = make_model(model_id="gpt-4", provider="openai")
        mock_get_model.return_value = builtin

        reg = ModelRegistry()
        result = reg.get_model("gpt-4")
        assert result is builtin

    @patch("pi_mono.coding_agent.core.model_registry.get_model")
    def test_get_with_provider_filter(self, mock_get_model) -> None:
        builtin = make_model(model_id="gpt-4", provider="openai")
        mock_get_model.return_value = builtin

        reg = ModelRegistry()
        result = reg.get_model("gpt-4", provider="openai")
        mock_get_model.assert_called_once_with("openai", "gpt-4")
        assert result is builtin


class TestListModels:
    @patch("pi_mono.coding_agent.core.model_registry.get_providers", return_value=[])
    def test_list_empty_with_no_providers(self, mock_providers) -> None:
        reg = ModelRegistry()
        models = reg.list_models()
        assert models == []

    @patch("pi_mono.coding_agent.core.model_registry.get_providers", return_value=["openai"])
    @patch("pi_mono.coding_agent.core.model_registry.get_models")
    def test_list_includes_custom_models(self, mock_get_models, mock_providers) -> None:
        mock_get_models.return_value = [make_model(model_id="gpt-4")]
        custom = make_model(model_id="my-custom")
        reg = ModelRegistry()
        reg.register_custom_model(custom)

        models = reg.list_models()
        ids = [m.id for m in models]
        assert "gpt-4" in ids
        assert "my-custom" in ids

    @patch("pi_mono.coding_agent.core.model_registry.get_models")
    def test_list_with_provider_filter(self, mock_get_models) -> None:
        mock_get_models.return_value = [make_model(model_id="gpt-4")]
        reg = ModelRegistry()
        reg.list_models(provider="openai")
        mock_get_models.assert_called_once_with("openai")


class TestSearchModels:
    @patch("pi_mono.coding_agent.core.model_registry.get_providers", return_value=[])
    def test_search_by_id(self, mock_providers) -> None:
        reg = ModelRegistry()
        reg.register_custom_model(make_model(model_id="claude-sonnet-4"))
        reg.register_custom_model(make_model(model_id="gpt-4o"))

        results = reg.search_models("claude")
        assert len(results) == 1
        assert results[0].id == "claude-sonnet-4"

    @patch("pi_mono.coding_agent.core.model_registry.get_providers", return_value=[])
    def test_search_case_insensitive(self, mock_providers) -> None:
        reg = ModelRegistry()
        reg.register_custom_model(make_model(model_id="GPT-4o"))

        results = reg.search_models("gpt")
        assert len(results) == 1

    @patch("pi_mono.coding_agent.core.model_registry.get_providers", return_value=[])
    def test_search_no_results(self, mock_providers) -> None:
        reg = ModelRegistry()
        results = reg.search_models("nonexistent")
        assert results == []
