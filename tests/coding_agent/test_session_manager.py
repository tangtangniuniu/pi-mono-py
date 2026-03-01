"""Tests for pi_mono.coding_agent.core.session_manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pi_mono.coding_agent.core.session_manager import (
    MessageEntry,
    MetadataEntry,
    SessionManager,
    ToolResultEntry,
    ToolUseEntry,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def session_mgr(tmp_path: Path) -> SessionManager:
    return SessionManager(sessions_dir=tmp_path / "sessions")


class TestCreateSession:
    async def test_create_returns_id(self, session_mgr: SessionManager) -> None:
        sid = await session_mgr.create_session()
        assert len(sid) == 12
        assert sid.isalnum()

    async def test_create_with_name(self, session_mgr: SessionManager) -> None:
        sid = await session_mgr.create_session(name="test-session")
        entries = await session_mgr.load_session(sid)
        assert len(entries) == 1
        assert isinstance(entries[0], MetadataEntry)
        assert entries[0].value == "test-session"

    async def test_create_creates_file(self, session_mgr: SessionManager) -> None:
        sid = await session_mgr.create_session()
        path = session_mgr._session_path(sid)
        assert path.exists()


class TestLoadSession:
    async def test_load_nonexistent_raises(self, session_mgr: SessionManager) -> None:
        with pytest.raises(FileNotFoundError):
            await session_mgr.load_session("nonexistent")

    async def test_load_returns_entries(self, session_mgr: SessionManager) -> None:
        sid = await session_mgr.create_session()
        msg = MessageEntry(
            type="message",
            timestamp=1.0,
            message={"role": "user", "content": "hi"},
        )
        await session_mgr.append_entry(sid, msg)
        entries = await session_mgr.load_session(sid)
        # 1 metadata + 1 message
        assert len(entries) == 2
        assert isinstance(entries[1], MessageEntry)


class TestAppendEntry:
    async def test_append_multiple(self, session_mgr: SessionManager) -> None:
        sid = await session_mgr.create_session()
        for i in range(3):
            await session_mgr.append_entry(
                sid,
                MessageEntry(type="message", timestamp=float(i), message={"i": i}),
            )
        entries = await session_mgr.load_session(sid)
        assert len(entries) == 4  # 1 metadata + 3 messages

    async def test_append_tool_entries(self, session_mgr: SessionManager) -> None:
        sid = await session_mgr.create_session()
        await session_mgr.append_entry(
            sid,
            ToolUseEntry(
                type="tool_use", timestamp=1.0,
                tool_name="bash", tool_call_id="c1", arguments={"cmd": "ls"},
            ),
        )
        await session_mgr.append_entry(
            sid,
            ToolResultEntry(
                type="tool_result", timestamp=2.0,
                tool_call_id="c1", content="output", is_error=False,
            ),
        )
        entries = await session_mgr.load_session(sid)
        assert isinstance(entries[1], ToolUseEntry)
        assert isinstance(entries[2], ToolResultEntry)


class TestListSessions:
    async def test_list_empty(self, session_mgr: SessionManager) -> None:
        summaries = await session_mgr.list_sessions()
        assert summaries == []

    async def test_list_multiple(self, session_mgr: SessionManager) -> None:
        await session_mgr.create_session(name="s1")
        await session_mgr.create_session(name="s2")
        summaries = await session_mgr.list_sessions()
        assert len(summaries) == 2


class TestDeleteSession:
    async def test_delete_removes_file(self, session_mgr: SessionManager) -> None:
        sid = await session_mgr.create_session()
        await session_mgr.delete_session(sid)
        path = session_mgr._session_path(sid)
        assert not path.exists()

    async def test_delete_nonexistent_is_noop(self, session_mgr: SessionManager) -> None:
        await session_mgr.delete_session("nonexistent")  # should not raise
