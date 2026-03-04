"""Tests for pi_mono.ai.providers.openai_compat."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pi_mono.ai.providers.openai_compat import (
    OpenAICompletionsProvider,
    _block_to_content,
    _build_params,
    _convert_messages,
    _convert_tools,
    _detect_compat,
    _get_compat,
    _has_tool_history,
    _map_stop_reason,
    _MutableBlock,
    _MutableUsage,
    _normalize_mistral_tool_id,
    _parse_streaming_json,
    _stream_completions,
)
from pi_mono.ai.types import (
    AssistantMessage,
    Context,
    Cost,
    Model,
    ModelCost,
    OpenAICompletionsCompat,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from tests.conftest import make_usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model(
    provider: str = "openai",
    base_url: str = "https://api.openai.com/v1",
    reasoning: bool = False,
    input_: list[str] | None = None,
    compat: OpenAICompletionsCompat | None = None,
) -> Model:
    return Model(
        id="gpt-4o",
        name="GPT-4o",
        api="openai-completions",
        provider=provider,
        base_url=base_url,
        reasoning=reasoning,
        input=input_ or ["text"],
        cost=ModelCost(input=5.0, output=15.0, cache_read=2.5, cache_write=0.0),
        context_window=128000,
        max_tokens=4096,
        compat=compat,
    )


# ===========================================================================
# _parse_streaming_json
# ===========================================================================


class TestParseStreamingJson:
    def test_empty_string_returns_empty_dict(self) -> None:
        assert _parse_streaming_json("") == {}

    def test_none_returns_empty_dict(self) -> None:
        assert _parse_streaming_json(None) == {}

    def test_valid_json(self) -> None:
        assert _parse_streaming_json('{"key": "value"}') == {"key": "value"}

    def test_truncated_json_recovers(self) -> None:
        result = _parse_streaming_json('{"name": "test"')
        assert result == {"name": "test"}

    def test_nested_truncated_json(self) -> None:
        result = _parse_streaming_json('{"a": {"b": "c"')
        assert result == {"a": {"b": "c"}}

    def test_truncated_array_recovers(self) -> None:
        result = _parse_streaming_json('{"items": [1, 2, 3')
        assert result == {"items": [1, 2, 3]}

    def test_unrecoverable_returns_empty(self) -> None:
        assert _parse_streaming_json("not json at all") == {}

    def test_whitespace_only(self) -> None:
        assert _parse_streaming_json("   ") == {}


# ===========================================================================
# _normalize_mistral_tool_id
# ===========================================================================


class TestNormalizeMistralToolId:
    def test_short_id_padded(self) -> None:
        result = _normalize_mistral_tool_id("abc")
        assert len(result) == 9
        assert result.startswith("abc")
        assert result.isalnum()

    def test_long_id_truncated(self) -> None:
        result = _normalize_mistral_tool_id("abcdefghijklmnop")
        assert len(result) == 9
        assert result == "abcdefghi"

    def test_exact_nine_unchanged(self) -> None:
        result = _normalize_mistral_tool_id("abcdefghi")
        assert result == "abcdefghi"

    def test_non_alnum_stripped(self) -> None:
        result = _normalize_mistral_tool_id("a-b_c!d@e")
        # Non-alnum chars removed: "abcde" then padded to 9
        assert len(result) == 9
        assert result[:5] == "abcde"
        assert result.isalnum()


# ===========================================================================
# _has_tool_history
# ===========================================================================


class TestHasToolHistory:
    def test_empty_messages(self) -> None:
        assert _has_tool_history([]) is False

    def test_user_message_only(self) -> None:
        msg = UserMessage(
            role="user",
            content=[TextContent(text="Hello")],
            timestamp=time.time(),
        )
        assert _has_tool_history([msg]) is False

    def test_with_tool_result(self) -> None:
        msg = ToolResultMessage(
            role="toolResult",
            tool_call_id="tc1",
            tool_name="bash",
            content=[TextContent(text="result")],
            is_error=False,
            timestamp=time.time(),
        )
        assert _has_tool_history([msg]) is True

    def test_with_assistant_tool_call(self) -> None:
        msg = AssistantMessage(
            content=[
                ToolCall(id="tc1", name="bash", arguments={"command": "ls"}),
            ],
            api="openai-completions",
            provider="openai",
            model="gpt-4o",
            usage=make_usage(),
            stop_reason="toolUse",
            timestamp=time.time(),
        )
        assert _has_tool_history([msg]) is True

    def test_with_text_only_assistant(self) -> None:
        msg = AssistantMessage(
            content=[TextContent(text="Hello")],
            api="openai-completions",
            provider="openai",
            model="gpt-4o",
            usage=make_usage(),
            stop_reason="stop",
            timestamp=time.time(),
        )
        assert _has_tool_history([msg]) is False


# ===========================================================================
# _map_stop_reason
# ===========================================================================


class TestMapStopReason:
    def test_none_returns_stop(self) -> None:
        assert _map_stop_reason(None) == "stop"

    def test_stop(self) -> None:
        assert _map_stop_reason("stop") == "stop"

    def test_length(self) -> None:
        assert _map_stop_reason("length") == "length"

    def test_tool_calls(self) -> None:
        assert _map_stop_reason("tool_calls") == "toolUse"

    def test_function_call(self) -> None:
        assert _map_stop_reason("function_call") == "toolUse"

    def test_content_filter(self) -> None:
        assert _map_stop_reason("content_filter") == "error"

    def test_unknown_defaults_to_stop(self) -> None:
        assert _map_stop_reason("whatever") == "stop"


# ===========================================================================
# _detect_compat
# ===========================================================================


class TestDetectCompat:
    def test_openai_defaults(self) -> None:
        model = _model(provider="openai")
        compat = _detect_compat(model)
        assert compat.supports_store is True
        assert compat.supports_developer_role is True
        assert compat.max_tokens_field == "max_completion_tokens"
        assert compat.requires_mistral_tool_ids is False
        assert compat.thinking_format == "openai"

    def test_mistral_detection(self) -> None:
        model = _model(provider="mistral", base_url="https://api.mistral.ai/v1")
        compat = _detect_compat(model)
        assert compat.supports_store is False
        assert compat.supports_developer_role is False
        assert compat.max_tokens_field == "max_tokens"
        assert compat.requires_tool_result_name is True
        assert compat.requires_mistral_tool_ids is True
        assert compat.requires_thinking_as_text is True

    def test_xai_detection(self) -> None:
        model = _model(provider="xai", base_url="https://api.x.ai/v1")
        compat = _detect_compat(model)
        assert compat.supports_store is False
        assert compat.supports_reasoning_effort is False

    def test_zai_detection(self) -> None:
        model = _model(provider="zai", base_url="https://api.z.ai/v1")
        compat = _detect_compat(model)
        assert compat.thinking_format == "zai"
        assert compat.supports_reasoning_effort is False

    def test_cerebras_detection(self) -> None:
        model = _model(provider="cerebras", base_url="https://api.cerebras.ai/v1")
        compat = _detect_compat(model)
        assert compat.supports_store is False
        assert compat.supports_developer_role is False

    def test_deepseek_via_url(self) -> None:
        model = _model(provider="custom", base_url="https://api.deepseek.com/v1")
        compat = _detect_compat(model)
        assert compat.supports_store is False

    def test_chutes_uses_max_tokens(self) -> None:
        model = _model(provider="custom", base_url="https://api.chutes.ai/v1")
        compat = _detect_compat(model)
        assert compat.max_tokens_field == "max_tokens"


class TestGetCompat:
    def test_no_compat_uses_detected(self) -> None:
        model = _model(provider="openai")
        compat = _get_compat(model)
        assert compat.supports_store is True

    def test_compat_overrides_detected(self) -> None:
        override = OpenAICompletionsCompat(
            supports_store=False,
            max_tokens_field="max_tokens",
        )
        model = _model(provider="openai", compat=override)
        compat = _get_compat(model)
        assert compat.supports_store is False
        assert compat.max_tokens_field == "max_tokens"
        # Non-overridden fields use detected defaults
        assert compat.supports_developer_role is True

    def test_none_compat_fields_use_detected(self) -> None:
        override = OpenAICompletionsCompat(supports_store=None)
        model = _model(provider="openai", compat=override)
        compat = _get_compat(model)
        assert compat.supports_store is True


# ===========================================================================
# _convert_messages
# ===========================================================================


class TestConvertMessages:
    def test_simple_user_message(self) -> None:
        model = _model()
        ctx = Context(
            system_prompt="You are helpful.",
            messages=[
                UserMessage(
                    role="user",
                    content=[TextContent(text="Hello")],
                    timestamp=time.time(),
                ),
            ],
            tools=[],
        )
        compat = _detect_compat(model)
        result = _convert_messages(model, ctx, compat)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."
        assert result[1]["role"] == "user"
        assert result[1]["content"] == [{"type": "text", "text": "Hello"}]

    def test_developer_role_for_reasoning_model(self) -> None:
        model = _model(reasoning=True)
        ctx = Context(
            system_prompt="System",
            messages=[],
            tools=[],
        )
        compat = _detect_compat(model)
        result = _convert_messages(model, ctx, compat)
        assert result[0]["role"] == "developer"

    def test_assistant_message_with_text(self) -> None:
        model = _model()
        ctx = Context(
            system_prompt=None,
            messages=[
                AssistantMessage(
                    content=[TextContent(text="Hi there")],
                    api="openai-completions",
                    provider="openai",
                    model="gpt-4o",
                    usage=make_usage(),
                    stop_reason="stop",
                    timestamp=time.time(),
                ),
            ],
            tools=[],
        )
        compat = _detect_compat(model)
        result = _convert_messages(model, ctx, compat)
        assert result[0]["role"] == "assistant"

    def test_tool_result_message(self) -> None:
        model = _model()
        ctx = Context(
            system_prompt=None,
            messages=[
                AssistantMessage(
                    content=[
                        ToolCall(id="tc1", name="bash", arguments={"command": "ls"}),
                    ],
                    api="openai-completions",
                    provider="openai",
                    model="gpt-4o",
                    usage=make_usage(),
                    stop_reason="toolUse",
                    timestamp=time.time(),
                ),
                ToolResultMessage(
                    role="toolResult",
                    tool_call_id="tc1",
                    tool_name="bash",
                    content=[TextContent(text="file1.py")],
                    is_error=False,
                    timestamp=time.time(),
                ),
            ],
            tools=[],
        )
        compat = _detect_compat(model)
        result = _convert_messages(model, ctx, compat)
        assert result[0]["role"] == "assistant"
        assert result[0]["tool_calls"][0]["id"] == "tc1"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "tc1"

    def test_empty_assistant_skipped(self) -> None:
        model = _model()
        ctx = Context(
            system_prompt=None,
            messages=[
                AssistantMessage(
                    content=[TextContent(text="")],
                    api="openai-completions",
                    provider="openai",
                    model="gpt-4o",
                    usage=make_usage(),
                    stop_reason="stop",
                    timestamp=time.time(),
                ),
            ],
            tools=[],
        )
        compat = _detect_compat(model)
        result = _convert_messages(model, ctx, compat)
        assert len(result) == 0


class TestConvertTools:
    def test_converts_tool_definitions(self) -> None:
        tools = [
            Tool(
                name="bash",
                description="Run bash commands",
                parameters={"type": "object", "properties": {"command": {"type": "string"}}},
            ),
        ]
        compat = _detect_compat(_model())
        result = _convert_tools(tools, compat)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "bash"
        assert result[0]["function"]["description"] == "Run bash commands"

    def test_strict_mode_flag(self) -> None:
        tools = [
            Tool(name="t", description="d", parameters={}),
        ]
        compat = _detect_compat(_model())
        result = _convert_tools(tools, compat)
        assert result[0]["function"]["strict"] is False


# ===========================================================================
# _MutableBlock / _MutableUsage / _block_to_content
# ===========================================================================


class TestMutableBlock:
    def test_block_to_text_content(self) -> None:
        block = _MutableBlock(type="text", text="hello")
        content = _block_to_content(block)
        assert isinstance(content, TextContent)
        assert content.text == "hello"

    def test_block_to_thinking_content(self) -> None:
        block = _MutableBlock(type="thinking", thinking="reasoning here", thinking_signature="reasoning_content")
        content = _block_to_content(block)
        assert isinstance(content, ThinkingContent)
        assert content.thinking == "reasoning here"
        assert content.thinking_signature == "reasoning_content"

    def test_block_to_tool_call(self) -> None:
        block = _MutableBlock(
            type="toolCall",
            id="tc1",
            name="bash",
            arguments={"command": "ls"},
        )
        content = _block_to_content(block)
        assert isinstance(content, ToolCall)
        assert content.id == "tc1"
        assert content.name == "bash"
        assert content.arguments == {"command": "ls"}


class TestMutableUsage:
    def test_to_usage_defaults(self) -> None:
        mu = _MutableUsage()
        u = mu.to_usage()
        assert u.input == 0
        assert u.output == 0
        assert u.cost.total == 0.0

    def test_to_usage_with_values(self) -> None:
        mu = _MutableUsage(input=100, output=50, total_tokens=150)
        u = mu.to_usage()
        assert u.input == 100
        assert u.output == 50
        assert u.total_tokens == 150

    def test_to_usage_with_custom_cost(self) -> None:
        mu = _MutableUsage(input=100, output=50)
        cost = Cost(input=0.01, output=0.02, cache_read=0.0, cache_write=0.0, total=0.03)
        u = mu.to_usage(cost=cost)
        assert u.cost.total == 0.03


# ===========================================================================
# _build_params
# ===========================================================================


class TestBuildParams:
    def test_basic_params(self) -> None:
        model = _model()
        ctx = Context(
            system_prompt="System",
            messages=[
                UserMessage(
                    role="user",
                    content=[TextContent(text="Hello")],
                    timestamp=time.time(),
                ),
            ],
            tools=[],
        )
        params = _build_params(model, ctx, None)
        assert params["model"] == "gpt-4o"
        assert params["stream"] is True
        assert params["stream_options"] == {"include_usage": True}
        assert params["store"] is False

    def test_tools_included(self) -> None:
        model = _model()
        tools = [
            Tool(name="bash", description="Run bash", parameters={"type": "object"}),
        ]
        ctx = Context(system_prompt=None, messages=[], tools=tools)
        params = _build_params(model, ctx, None)
        assert "tools" in params
        assert len(params["tools"]) == 1

    def test_empty_tools_with_history(self) -> None:
        model = _model()
        ctx = Context(
            system_prompt=None,
            messages=[
                ToolResultMessage(
                    role="toolResult",
                    tool_call_id="tc1",
                    tool_name="bash",
                    content=[TextContent(text="result")],
                    is_error=False,
                    timestamp=time.time(),
                ),
            ],
            tools=[],
        )
        params = _build_params(model, ctx, None)
        assert params["tools"] == []


# ===========================================================================
# _stream_completions (integration-style, mocking OpenAI client)
# ===========================================================================


class TestStreamCompletions:
    @pytest.mark.asyncio
    async def test_text_streaming(self) -> None:
        """Test basic text streaming produces correct events."""
        model = _model()
        ctx = Context(
            system_prompt="System",
            messages=[
                UserMessage(
                    role="user",
                    content=[TextContent(text="Hello")],
                    timestamp=time.time(),
                ),
            ],
            tools=[],
        )

        # Build mock chunks
        chunk1 = MagicMock()
        chunk1.usage = None
        chunk1.choices = [
            MagicMock(
                finish_reason=None,
                delta=MagicMock(
                    content="Hello ",
                    tool_calls=None,
                    reasoning_content=None,
                    reasoning=None,
                    reasoning_text=None,
                    reasoning_details=None,
                ),
            )
        ]

        chunk2 = MagicMock()
        chunk2.usage = None
        chunk2.choices = [
            MagicMock(
                finish_reason=None,
                delta=MagicMock(
                    content="world!",
                    tool_calls=None,
                    reasoning_content=None,
                    reasoning=None,
                    reasoning_text=None,
                    reasoning_details=None,
                ),
            )
        ]

        chunk_done = MagicMock()
        chunk_done.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        )
        chunk_done.choices = [
            MagicMock(
                finish_reason="stop",
                delta=MagicMock(
                    content=None,
                    tool_calls=None,
                    reasoning_content=None,
                    reasoning=None,
                    reasoning_text=None,
                    reasoning_details=None,
                ),
            )
        ]

        async def mock_stream() -> Any:
            for c in [chunk1, chunk2, chunk_done]:
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        with (
            patch("pi_mono.ai.providers.openai_compat._create_client", return_value=mock_client),
            patch("pi_mono.ai.providers.openai_compat.get_env_api_key", return_value="test-key"),
        ):
            events = []
            async for event in _stream_completions(model, ctx):
                events.append(event)

        event_types = [e.type for e in events]
        assert "start" in event_types
        assert "text_start" in event_types
        assert "text_delta" in event_types
        assert "text_end" in event_types
        assert "done" in event_types

    @pytest.mark.asyncio
    async def test_error_yields_error_event(self) -> None:
        """Test that exceptions produce an ErrorEvent."""
        model = _model()
        ctx = Context(
            system_prompt=None,
            messages=[
                UserMessage(
                    role="user",
                    content=[TextContent(text="Hello")],
                    timestamp=time.time(),
                ),
            ],
            tools=[],
        )

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API error"))

        with (
            patch("pi_mono.ai.providers.openai_compat._create_client", return_value=mock_client),
            patch("pi_mono.ai.providers.openai_compat.get_env_api_key", return_value="test-key"),
        ):
            events = []
            async for event in _stream_completions(model, ctx):
                events.append(event)

        assert len(events) == 1
        assert events[0].type == "error"
        assert "API error" in events[0].error.error_message


# ===========================================================================
# OpenAICompletionsProvider
# ===========================================================================


class TestOpenAICompletionsProvider:
    def test_api_property(self) -> None:
        provider = OpenAICompletionsProvider()
        assert provider.api == "openai-completions"

    @pytest.mark.asyncio
    async def test_stream_delegates(self) -> None:
        """Test that stream() delegates to _stream_completions."""
        provider = OpenAICompletionsProvider()
        model = _model()
        ctx = Context(
            system_prompt=None,
            messages=[
                UserMessage(
                    role="user",
                    content=[TextContent(text="Hi")],
                    timestamp=time.time(),
                ),
            ],
            tools=[],
        )

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("test"))

        with (
            patch("pi_mono.ai.providers.openai_compat._create_client", return_value=mock_client),
            patch("pi_mono.ai.providers.openai_compat.get_env_api_key", return_value="test-key"),
        ):
            events = []
            async for event in provider.stream(model, ctx):
                events.append(event)

        assert len(events) >= 1
        assert events[-1].type == "error"

    @pytest.mark.asyncio
    async def test_stream_simple_requires_api_key(self) -> None:
        """Test that stream_simple raises when no API key is found."""
        provider = OpenAICompletionsProvider()
        model = _model()
        ctx = Context(system_prompt=None, messages=[], tools=[])

        with (
            patch("pi_mono.ai.providers.openai_compat.get_env_api_key", return_value=None),
            pytest.raises(ValueError, match="No API key"),
        ):
            async for _ in provider.stream_simple(model, ctx):
                pass


# ===========================================================================
# Cost calculation integration
# ===========================================================================


class TestCostCalculation:
    def test_usage_cost_in_done_event(self) -> None:
        """Verify that cost is calculated from model pricing and usage."""
        from pi_mono.ai.models import calculate_cost

        model = _model()
        usage = Usage(
            input=1000,
            output=500,
            cache_read=200,
            cache_write=0,
            total_tokens=1700,
            cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
        )
        cost = calculate_cost(model, usage)
        # input: 5.0/1M * 1000 = 0.005
        # output: 15.0/1M * 500 = 0.0075
        # cache_read: 2.5/1M * 200 = 0.0005
        expected_total = 0.005 + 0.0075 + 0.0005
        assert abs(cost.total - expected_total) < 1e-10
