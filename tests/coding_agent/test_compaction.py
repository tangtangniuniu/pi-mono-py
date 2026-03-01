"""Tests for pi_mono.coding_agent.core.compaction — compaction strategies."""

from __future__ import annotations

import time

from pi_mono.ai.types import UserMessage
from pi_mono.coding_agent.core.compaction import NoCompaction, SummaryCompaction
from tests.conftest import make_assistant_message, make_model


def _user_msg(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=time.time())


class TestNoCompaction:
    async def test_returns_messages_unchanged(self) -> None:
        model = make_model()
        msgs = [_user_msg(f"msg{i}") for i in range(10)]
        strategy = NoCompaction()
        result = await strategy.compact(msgs, model, "system")
        assert result is msgs
        assert len(result) == 10


class TestSummaryCompaction:
    async def test_short_messages_unchanged(self) -> None:
        """Messages <= 4 should not be compacted."""
        model = make_model()
        msgs = [_user_msg("hi"), _user_msg("there")]
        strategy = SummaryCompaction()
        result = await strategy.compact(msgs, model, "system")
        assert result is msgs

    async def test_exactly_four_messages_unchanged(self) -> None:
        model = make_model()
        msgs = [_user_msg(f"msg{i}") for i in range(4)]
        strategy = SummaryCompaction()
        result = await strategy.compact(msgs, model, "system")
        assert result is msgs

    async def test_fallback_keeps_recent_without_stream_fn(self) -> None:
        """Without a stream function, only recent messages are kept."""
        model = make_model()
        msgs = [_user_msg(f"msg{i}") for i in range(8)]
        strategy = SummaryCompaction(stream_fn=None)
        result = await strategy.compact(msgs, model, "system")
        assert len(result) == 4
        # Should keep the last 4 messages
        assert result[0].content == "msg4"
        assert result[3].content == "msg7"

    async def test_with_stream_fn_produces_summary(self) -> None:
        """With a working stream function, summary is prepended to recent."""
        from pi_mono.ai.types import DoneEvent

        model = make_model()
        summary_msg = make_assistant_message(text="Summary of conversation", model=model.id)

        async def mock_stream_fn(m, ctx, opts):
            yield DoneEvent(reason="stop", message=summary_msg)

        msgs = [_user_msg(f"msg{i}") for i in range(8)]
        strategy = SummaryCompaction(stream_fn=mock_stream_fn)
        result = await strategy.compact(msgs, model, "system")

        # Should have 1 summary + 4 recent = 5
        assert len(result) == 5
        assert "summary" in result[0].content.lower() or "Summary" in result[0].content

    async def test_stream_fn_error_falls_back(self) -> None:
        """If stream function throws, falls back to keeping recent only."""
        model = make_model()

        async def failing_stream_fn(m, ctx, opts):
            raise RuntimeError("LLM unavailable")
            yield

        msgs = [_user_msg(f"msg{i}") for i in range(8)]
        strategy = SummaryCompaction(stream_fn=failing_stream_fn)
        result = await strategy.compact(msgs, model, "system")
        assert len(result) == 4

    async def test_preserves_recent_messages_order(self) -> None:
        """Recent messages should maintain their original order."""
        model = make_model()
        msgs = [_user_msg(f"msg{i}") for i in range(6)]
        strategy = SummaryCompaction()
        result = await strategy.compact(msgs, model, "system")
        contents = [m.content for m in result]
        assert contents == ["msg2", "msg3", "msg4", "msg5"]
