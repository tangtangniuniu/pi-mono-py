"""Tests for pi_mono.server — session registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pi_mono.ai.api_registry import clear_api_providers, register_api_provider
from pi_mono.coding_agent.core.model_registry import ModelRegistry
from pi_mono.coding_agent.core.session_manager import SessionManager
from pi_mono.coding_agent.core.settings_manager import SettingsManager
from pi_mono.server.session_registry import SessionRegistry
from tests.conftest import MockLLMProvider, make_model

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def registry(tmp_path: Path) -> SessionRegistry:
    clear_api_providers()
    provider = MockLLMProvider(response_text="test")
    register_api_provider(provider)

    model_reg = ModelRegistry()
    model_reg.register_custom_model(make_model(model_id="claude-sonnet-4-20250514"))

    reg = SessionRegistry(
        settings_manager=SettingsManager(settings_file=tmp_path / "settings.yaml"),
        session_manager=SessionManager(sessions_dir=tmp_path / "sessions"),
        model_registry=model_reg,
    )
    yield reg  # type: ignore[misc]
    clear_api_providers()


class TestSessionRegistryCreate:
    async def test_create_returns_session(self, registry: SessionRegistry) -> None:
        session = await registry.create()
        assert session.session_id is not None
        assert session.agent is not None

    async def test_create_registers_session(self, registry: SessionRegistry) -> None:
        session = await registry.create()
        assert registry.get(session.session_id) is session


class TestSessionRegistryGet:
    async def test_get_nonexistent_returns_none(self, registry: SessionRegistry) -> None:
        assert registry.get("nonexistent") is None

    async def test_get_existing(self, registry: SessionRegistry) -> None:
        session = await registry.create()
        found = registry.get(session.session_id)
        assert found is session


class TestSessionRegistryDelete:
    async def test_delete_removes_session(self, registry: SessionRegistry) -> None:
        session = await registry.create()
        sid = session.session_id
        result = await registry.delete(sid)
        assert result is True
        assert registry.get(sid) is None

    async def test_delete_nonexistent_returns_false(self, registry: SessionRegistry) -> None:
        result = await registry.delete("nonexistent")
        assert result is False


class TestSessionRegistryList:
    async def test_list_empty(self, registry: SessionRegistry) -> None:
        assert registry.list_sessions() == []

    async def test_list_multiple(self, registry: SessionRegistry) -> None:
        await registry.create()
        await registry.create()
        assert len(registry.list_sessions()) == 2
