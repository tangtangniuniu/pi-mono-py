"""Tests for pi_mono.coding_agent.core.agent_session — integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pi_mono.ai.api_registry import clear_api_providers, register_api_provider
from pi_mono.coding_agent.core.agent_session import AgentSession
from pi_mono.coding_agent.core.compaction import NoCompaction
from pi_mono.coding_agent.core.model_registry import ModelRegistry
from pi_mono.coding_agent.core.session_manager import SessionManager
from pi_mono.coding_agent.core.settings_manager import SettingsManager
from tests.conftest import MockLLMProvider, make_model

if TYPE_CHECKING:
    from pathlib import Path

    from pi_mono.agent.types import AgentEvent


@pytest.fixture
def mock_registry() -> ModelRegistry:
    reg = ModelRegistry()
    reg.register_custom_model(make_model(model_id="claude-sonnet-4-20250514"))
    reg.register_custom_model(make_model(model_id="test-model"))
    return reg


@pytest.fixture
def agent_session(tmp_path: Path, mock_registry: ModelRegistry) -> AgentSession:
    clear_api_providers()
    provider = MockLLMProvider(response_text="Test response")
    register_api_provider(provider)

    settings_mgr = SettingsManager(settings_file=tmp_path / "settings.json")
    session_mgr = SessionManager(sessions_dir=tmp_path / "sessions")

    session = AgentSession(
        settings_manager=settings_mgr,
        session_manager=session_mgr,
        model_registry=mock_registry,
        compaction=NoCompaction(),
    )
    yield session  # type: ignore[misc]
    clear_api_providers()


class TestStart:
    async def test_start_creates_session(self, agent_session: AgentSession) -> None:
        await agent_session.start()
        assert agent_session.session_id is not None
        assert agent_session.agent is not None

    async def test_start_with_existing_session_id(self, agent_session: AgentSession) -> None:
        await agent_session.start(session_id="existing-123")
        assert agent_session.session_id == "existing-123"

    async def test_start_configures_agent_model(self, agent_session: AgentSession) -> None:
        await agent_session.start()
        assert agent_session.agent is not None
        assert agent_session.agent.state.model is not None

    async def test_start_with_custom_model(self, agent_session: AgentSession) -> None:
        await agent_session.start(model="test-model")
        assert agent_session.agent is not None
        assert agent_session.agent.state.model.id == "test-model"

    async def test_start_configures_tools(self, agent_session: AgentSession) -> None:
        await agent_session.start()
        assert len(agent_session.agent.state.tools) == 7


class TestSendMessage:
    async def test_send_message_without_start_raises(self, agent_session: AgentSession) -> None:
        with pytest.raises(RuntimeError, match="Session not started"):
            async for _ in agent_session.send_message("hi"):
                pass

    async def test_send_message_yields_events(self, agent_session: AgentSession) -> None:
        await agent_session.start()
        events: list[AgentEvent] = []
        async for event in agent_session.send_message("hello"):
            events.append(event)

        assert len(events) > 0
        # Should have at least agent_start and agent_end
        types = [e.type for e in events]
        assert "agent_start" in types
        assert "agent_end" in types


class TestSwitchModel:
    async def test_switch_model_without_start_raises(self, agent_session: AgentSession) -> None:
        with pytest.raises(RuntimeError, match="Session not started"):
            await agent_session.switch_model("test-model")

    async def test_switch_model_updates_agent(self, agent_session: AgentSession) -> None:
        await agent_session.start()
        await agent_session.switch_model("test-model")
        assert agent_session.agent.state.model.id == "test-model"


class TestCompact:
    async def test_compact_without_start_raises(self, agent_session: AgentSession) -> None:
        with pytest.raises(RuntimeError, match="Session not started"):
            await agent_session.compact()

    async def test_compact_with_no_messages_is_noop(self, agent_session: AgentSession) -> None:
        await agent_session.start()
        await agent_session.compact()
        assert agent_session.agent.state.messages == []


class TestClose:
    async def test_close_clears_session(self, agent_session: AgentSession) -> None:
        await agent_session.start()
        assert agent_session.agent is not None
        await agent_session.close()
        assert agent_session.agent is None
        assert agent_session.session_id is None

    async def test_close_without_start_is_noop(self, agent_session: AgentSession) -> None:
        await agent_session.close()  # should not raise
