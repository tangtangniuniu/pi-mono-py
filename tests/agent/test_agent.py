"""Tests for pi_mono.agent — Agent class, state, events, and queuing."""

from __future__ import annotations

import asyncio

import pytest

from pi_mono.agent.agent import Agent
from pi_mono.agent.types import (
    AgentEvent,
    AgentStartEvent,
    AgentThinkingLevel,
    AgentTool,
)
from pi_mono.ai.types import UserMessage
from tests.conftest import MockLLMProvider, make_model


@pytest.fixture
def agent_with_mock() -> Agent:
    """Create an Agent with a mock provider registered."""
    from pi_mono.ai.api_registry import clear_api_providers, register_api_provider

    clear_api_providers()
    provider = MockLLMProvider(response_text="Agent response")
    register_api_provider(provider)

    agent = Agent()
    agent.set_model(make_model())
    agent.set_system_prompt("You are a test agent.")

    yield agent  # type: ignore[misc]
    clear_api_providers()


class TestAgentState:
    def test_initial_state(self) -> None:
        agent = Agent()
        assert agent.state.system_prompt == ""
        assert agent.state.model is None
        assert agent.state.thinking_level == AgentThinkingLevel.OFF
        assert agent.state.tools == []
        assert agent.state.messages == []
        assert agent.state.is_streaming is False

    def test_set_model(self) -> None:
        agent = Agent()
        model = make_model()
        agent.set_model(model)
        assert agent.state.model is model

    def test_set_system_prompt(self) -> None:
        agent = Agent()
        agent.set_system_prompt("Hello")
        assert agent.state.system_prompt == "Hello"

    def test_set_thinking_level(self) -> None:
        agent = Agent()
        agent.set_thinking_level(AgentThinkingLevel.HIGH)
        assert agent.state.thinking_level == AgentThinkingLevel.HIGH

    def test_set_tools(self, mock_tool: AgentTool) -> None:
        agent = Agent()
        agent.set_tools([mock_tool])
        assert len(agent.state.tools) == 1
        assert agent.state.tools[0].name == "mock_tool"


class TestImmutableMessages:
    def test_append_creates_new_list(self) -> None:
        agent = Agent()
        original = agent.state.messages
        msg = UserMessage(content="hi", timestamp=1.0)
        agent.append_message(msg)

        assert len(agent.state.messages) == 1
        assert original == []  # original unchanged

    def test_replace_messages(self) -> None:
        agent = Agent()
        msg = UserMessage(content="hi", timestamp=1.0)
        agent.replace_messages([msg])
        assert len(agent.state.messages) == 1

    def test_clear_messages(self) -> None:
        agent = Agent()
        agent.append_message(UserMessage(content="hi", timestamp=1.0))
        agent.clear_messages()
        assert agent.state.messages == []


class TestSubscribeEmit:
    def test_subscribe_receives_events(self) -> None:
        agent = Agent()
        events: list[AgentEvent] = []
        agent.subscribe(events.append)
        agent._emit(AgentStartEvent())
        assert len(events) == 1
        assert events[0].type == "agent_start"

    def test_unsubscribe(self) -> None:
        agent = Agent()
        events: list[AgentEvent] = []
        unsub = agent.subscribe(events.append)
        agent._emit(AgentStartEvent())
        unsub()
        agent._emit(AgentStartEvent())
        assert len(events) == 1  # only the first event


class TestPrompt:
    async def test_prompt_requires_model(self) -> None:
        agent = Agent()
        with pytest.raises(RuntimeError, match="No model configured"):
            await agent.prompt("hi")

    async def test_prompt_string(self, agent_with_mock: Agent) -> None:
        events: list[AgentEvent] = []
        agent_with_mock.subscribe(events.append)
        await agent_with_mock.prompt("hello")

        assert len(agent_with_mock.state.messages) >= 1
        assert agent_with_mock.state.is_streaming is False

    async def test_prompt_rejects_while_streaming(self, agent_with_mock: Agent) -> None:
        agent_with_mock._state.is_streaming = True
        with pytest.raises(RuntimeError, match="already processing"):
            await agent_with_mock.prompt("hi")


class TestAbort:
    def test_abort_sets_event(self) -> None:
        agent = Agent()
        agent._abort_event = asyncio.Event()
        agent.abort()
        assert agent._abort_event.is_set()

    def test_abort_without_event_is_noop(self) -> None:
        agent = Agent()
        agent.abort()  # should not raise


class TestSteeringFollowUp:
    def test_steer_queues_message(self) -> None:
        agent = Agent()
        msg = UserMessage(content="steer!", timestamp=1.0)
        agent.steer(msg)
        assert agent.has_queued_messages()

    def test_follow_up_queues_message(self) -> None:
        agent = Agent()
        msg = UserMessage(content="follow!", timestamp=1.0)
        agent.follow_up(msg)
        assert agent.has_queued_messages()

    def test_dequeue_steering_one_at_a_time(self) -> None:
        agent = Agent()
        m1 = UserMessage(content="1", timestamp=1.0)
        m2 = UserMessage(content="2", timestamp=2.0)
        agent.steer(m1)
        agent.steer(m2)

        result = agent._dequeue_steering()
        assert len(result) == 1
        assert result[0] is m1

        result2 = agent._dequeue_steering()
        assert len(result2) == 1
        assert result2[0] is m2

    def test_dequeue_steering_all(self) -> None:
        agent = Agent()
        agent.set_steering_mode("all")
        m1 = UserMessage(content="1", timestamp=1.0)
        m2 = UserMessage(content="2", timestamp=2.0)
        agent.steer(m1)
        agent.steer(m2)

        result = agent._dequeue_steering()
        assert len(result) == 2

    def test_clear_all_queues(self) -> None:
        agent = Agent()
        agent.steer(UserMessage(content="s", timestamp=1.0))
        agent.follow_up(UserMessage(content="f", timestamp=1.0))
        agent.clear_all_queues()
        assert not agent.has_queued_messages()


class TestReset:
    def test_reset_clears_state(self) -> None:
        agent = Agent()
        agent.set_model(make_model())
        agent.append_message(UserMessage(content="hi", timestamp=1.0))
        agent.steer(UserMessage(content="s", timestamp=1.0))
        agent._state.error = "some error"

        agent.reset()

        assert agent.state.messages == []
        assert agent.state.error is None
        assert not agent.has_queued_messages()
        assert agent.state.model is not None  # model preserved
