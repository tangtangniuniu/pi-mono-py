"""Tests for pi_mono.ai.types — immutable content, message, and event types."""

from __future__ import annotations

import pytest

from pi_mono.ai.types import (
    AssistantMessage,
    Context,
    Cost,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


class TestContentTypes:
    def test_text_content_creation(self) -> None:
        tc = TextContent(text="hello")
        assert tc.text == "hello"
        assert tc.type == "text"

    def test_text_content_is_frozen(self) -> None:
        tc = TextContent(text="hello")
        with pytest.raises(AttributeError):
            tc.text = "world"  # type: ignore[misc]

    def test_thinking_content_creation(self) -> None:
        tc = ThinkingContent(thinking="hmm")
        assert tc.thinking == "hmm"
        assert tc.type == "thinking"

    def test_image_content_creation(self) -> None:
        ic = ImageContent(data="base64data", mime_type="image/png")
        assert ic.data == "base64data"
        assert ic.type == "image"

    def test_tool_call_creation(self) -> None:
        tc = ToolCall(id="call_1", name="bash", arguments={"cmd": "ls"})
        assert tc.name == "bash"
        assert tc.arguments == {"cmd": "ls"}
        assert tc.type == "toolCall"

    def test_tool_call_is_frozen(self) -> None:
        tc = ToolCall(id="call_1", name="bash", arguments={})
        with pytest.raises(AttributeError):
            tc.name = "other"  # type: ignore[misc]


class TestMessageTypes:
    def test_user_message(self) -> None:
        msg = UserMessage(content="hello", timestamp=1.0)
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_user_message_is_frozen(self) -> None:
        msg = UserMessage(content="hello", timestamp=1.0)
        with pytest.raises(AttributeError):
            msg.content = "world"  # type: ignore[misc]

    def test_assistant_message(self) -> None:
        usage = Usage(
            input=10, output=5, cache_read=0, cache_write=0,
            total_tokens=15,
            cost=Cost(input=0.1, output=0.05, cache_read=0.0, cache_write=0.0, total=0.15),
        )
        msg = AssistantMessage(
            content=[TextContent(text="hi")],
            api="openai-completions",
            provider="openai",
            model="gpt-4o",
            usage=usage,
            stop_reason="stop",
            timestamp=1.0,
        )
        assert msg.role == "assistant"
        assert len(msg.content) == 1

    def test_tool_result_message(self) -> None:
        msg = ToolResultMessage(
            tool_call_id="call_1",
            tool_name="bash",
            content=[TextContent(text="output")],
            is_error=False,
            timestamp=1.0,
        )
        assert msg.role == "toolResult"
        assert msg.tool_name == "bash"


class TestEventTypes:
    def test_start_event_type_discriminator(self) -> None:
        from tests.conftest import make_assistant_message
        msg = make_assistant_message()
        ev = StartEvent(partial=msg)
        assert ev.type == "start"

    def test_text_delta_event(self) -> None:
        from tests.conftest import make_assistant_message
        msg = make_assistant_message()
        ev = TextDeltaEvent(content_index=0, delta="hi", partial=msg)
        assert ev.type == "text_delta"
        assert ev.delta == "hi"

    def test_done_event(self) -> None:
        from tests.conftest import make_assistant_message
        msg = make_assistant_message()
        ev = DoneEvent(reason="stop", message=msg)
        assert ev.type == "done"

    def test_error_event(self) -> None:
        from tests.conftest import make_assistant_message
        msg = make_assistant_message()
        ev = ErrorEvent(reason="error", error=msg)
        assert ev.type == "error"


class TestModelAndContext:
    def test_model_creation(self) -> None:
        from tests.conftest import make_model
        model = make_model(model_id="gpt-4o", provider="openai")
        assert model.id == "gpt-4o"
        assert model.provider == "openai"

    def test_context_creation(self) -> None:
        ctx = Context(
            messages=[UserMessage(content="hi", timestamp=1.0)],
            system_prompt="You are helpful.",
        )
        assert ctx.system_prompt == "You are helpful."
        assert len(ctx.messages) == 1
