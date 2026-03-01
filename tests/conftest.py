"""Shared test fixtures for pi-mono test suite."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pytest

from pi_mono.agent.types import AgentTool, AgentToolResult
from pi_mono.ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    Cost,
    DoneEvent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    Usage,
)

if TYPE_CHECKING:
    import asyncio
    from collections.abc import AsyncIterator
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_usage(
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> Usage:
    """Create a minimal Usage object for tests."""
    return Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=0,
        cache_write=0,
        total_tokens=input_tokens + output_tokens,
        cost=Cost(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0, total=0.0),
    )


def make_assistant_message(
    text: str = "Hello!",
    model: str = "test-model",
) -> AssistantMessage:
    """Create a minimal AssistantMessage for tests."""
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="openai-completions",
        provider="openai",
        model=model,
        usage=make_usage(),
        stop_reason="stop",
        timestamp=time.time(),
    )


def make_model(
    model_id: str = "test-model",
    provider: str = "openai",
    base_url: str = "https://api.openai.com/v1",
) -> Model:
    """Create a minimal Model for tests."""
    return Model(
        id=model_id,
        name=model_id,
        api="openai-completions",
        provider=provider,
        base_url=base_url,
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0),
        context_window=128000,
        max_tokens=4096,
    )


# ---------------------------------------------------------------------------
# Mock LLM Provider
# ---------------------------------------------------------------------------

class MockLLMProvider:
    """A mock LLM provider that yields pre-configured events."""

    def __init__(self, response_text: str = "Mock response") -> None:
        self._response_text = response_text
        self._api = "openai-completions"
        self.call_count = 0

    @property
    def api(self) -> str:
        return self._api

    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        async for event in self._generate_events(model):
            yield event

    async def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        async for event in self._generate_events(model):
            yield event

    async def _generate_events(
        self, model: Model,
    ) -> AsyncIterator[AssistantMessageEvent]:
        self.call_count += 1
        msg = make_assistant_message(text="", model=model.id)

        yield StartEvent(partial=msg)
        yield TextStartEvent(content_index=0, partial=msg)
        yield TextDeltaEvent(content_index=0, delta=self._response_text, partial=msg)

        final_msg = make_assistant_message(text=self._response_text, model=model.id)
        yield TextEndEvent(content_index=0, content=self._response_text, partial=final_msg)
        yield DoneEvent(reason="stop", message=final_msg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    """Provide a temporary config directory for tests."""
    return tmp_path


@pytest.fixture
def tmp_project_dir(tmp_path: Path) -> Path:
    """Provide a temporary project directory with .pi structure."""
    pi_dir = tmp_path / ".pi"
    pi_dir.mkdir()
    (pi_dir / "extensions").mkdir()
    (pi_dir / "prompts").mkdir()
    return tmp_path


@pytest.fixture
def test_model() -> Model:
    """Provide a test Model instance."""
    return make_model()


@pytest.fixture
def mock_provider() -> MockLLMProvider:
    """Provide a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def mock_tool() -> AgentTool:
    """Provide a mock AgentTool."""
    async def execute(
        tool_call_id: str,
        name: str,
        arguments: dict[str, Any],
        signal: asyncio.Event | None = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        return AgentToolResult(
            content=[TextContent(text=f"Result of {name}: {arguments}")],
        )

    return AgentTool(
        name="mock_tool",
        description="A mock tool for testing",
        label="Mock",
        parameters={
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Test input"},
            },
            "required": ["input"],
        },
        execute=execute,
    )
