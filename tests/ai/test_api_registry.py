"""Tests for pi_mono.ai.api_registry."""

from __future__ import annotations

import pytest

from pi_mono.ai.api_registry import (
    clear_api_providers,
    get_api_provider,
    get_api_providers,
    register_api_provider,
    unregister_api_providers,
)
from tests.conftest import MockLLMProvider


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Ensure a clean registry before and after each test."""
    clear_api_providers()
    yield  # type: ignore[misc]
    clear_api_providers()


class TestRegisterAndLookup:
    def test_register_and_get(self) -> None:
        provider = MockLLMProvider()
        register_api_provider(provider)

        result = get_api_provider("openai-completions")
        assert result is provider

    def test_get_nonexistent_returns_none(self) -> None:
        assert get_api_provider("nonexistent") is None

    def test_register_replaces_existing(self) -> None:
        p1 = MockLLMProvider(response_text="first")
        p2 = MockLLMProvider(response_text="second")
        register_api_provider(p1)
        register_api_provider(p2)

        result = get_api_provider("openai-completions")
        assert result is p2

    def test_get_all_providers(self) -> None:
        p1 = MockLLMProvider()
        register_api_provider(p1)

        providers = get_api_providers()
        assert len(providers) == 1
        assert providers[0] is p1


class TestUnregisterBySource:
    def test_unregister_by_source_id(self) -> None:
        provider = MockLLMProvider()
        register_api_provider(provider, source_id="ext-a")

        unregister_api_providers("ext-a")
        assert get_api_provider("openai-completions") is None

    def test_unregister_preserves_other_sources(self) -> None:
        p1 = MockLLMProvider()
        p1._api = "api-a"
        p2 = MockLLMProvider()
        p2._api = "api-b"

        register_api_provider(p1, source_id="ext-a")
        register_api_provider(p2, source_id="ext-b")

        unregister_api_providers("ext-a")
        assert get_api_provider("api-a") is None
        assert get_api_provider("api-b") is p2

    def test_unregister_nonexistent_source_is_noop(self) -> None:
        provider = MockLLMProvider()
        register_api_provider(provider)
        unregister_api_providers("nonexistent")
        assert get_api_provider("openai-completions") is provider


class TestClear:
    def test_clear_removes_all(self) -> None:
        register_api_provider(MockLLMProvider())
        clear_api_providers()
        assert get_api_providers() == []
