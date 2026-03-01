"""Tests for pi_mono.ai.env_api_keys."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pi_mono.ai.env_api_keys import get_env_api_key

if TYPE_CHECKING:
    import pytest


class TestGetEnvApiKey:
    def test_known_provider_with_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        assert get_env_api_key("openai") == "sk-test-123"

    def test_known_provider_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert get_env_api_key("openai") is None

    def test_unknown_provider_returns_none(self) -> None:
        assert get_env_api_key("unknown-provider") is None

    def test_first_matching_var_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
        assert get_env_api_key("google") == "google-key"

    def test_fallback_to_second_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
        assert get_env_api_key("google") == "gemini-key"

    def test_empty_string_is_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "")
        assert get_env_api_key("openai") is None
