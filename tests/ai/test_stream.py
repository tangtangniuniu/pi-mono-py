"""Tests for pi_mono.ai.stream — unified streaming interface."""

from __future__ import annotations

import pytest

from pi_mono.ai.api_registry import clear_api_providers, register_api_provider
from pi_mono.ai.stream import complete, stream, stream_simple
from pi_mono.ai.types import (
    Context,
    UserMessage,
)
from tests.conftest import MockLLMProvider, make_model


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    clear_api_providers()
    yield  # type: ignore[misc]
    clear_api_providers()


class TestStream:
    async def test_stream_yields_events(self) -> None:
        provider = MockLLMProvider(response_text="Hello world")
        register_api_provider(provider)

        model = make_model()
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1.0)])
        event_stream = stream(model, ctx)

        events = []
        async for event in event_stream:
            events.append(event)

        assert len(events) == 5  # start, text_start, text_delta, text_end, done
        assert events[0].type == "start"
        assert events[-1].type == "done"

    async def test_stream_result(self) -> None:
        provider = MockLLMProvider(response_text="result text")
        register_api_provider(provider)

        model = make_model()
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1.0)])
        event_stream = stream(model, ctx)

        result = await event_stream.result()
        assert result.content[0].text == "result text"  # type: ignore[union-attr]

    async def test_stream_no_provider_raises(self) -> None:
        model = make_model()
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1.0)])
        with pytest.raises(LookupError, match="No API provider"):
            stream(model, ctx)


class TestStreamSimple:
    async def test_stream_simple_yields_events(self) -> None:
        provider = MockLLMProvider(response_text="simple response")
        register_api_provider(provider)

        model = make_model()
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1.0)])
        event_stream = stream_simple(model, ctx)

        events = []
        async for event in event_stream:
            events.append(event)

        assert len(events) == 5
        assert events[2].type == "text_delta"


class TestComplete:
    async def test_complete_returns_message(self) -> None:
        provider = MockLLMProvider(response_text="complete response")
        register_api_provider(provider)

        model = make_model()
        ctx = Context(messages=[UserMessage(content="hi", timestamp=1.0)])
        msg = await complete(model, ctx)

        assert msg.content[0].text == "complete response"  # type: ignore[union-attr]
        assert msg.stop_reason == "stop"
