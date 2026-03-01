from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from pi_mono.coding_agent.config import SESSIONS_DIR

if TYPE_CHECKING:
    from pathlib import Path


class SessionEntry(BaseModel):
    """Base session entry."""
    type: str
    timestamp: float


class MessageEntry(SessionEntry):
    """A message in the session."""
    type: Literal["message"] = "message"
    message: dict[str, Any]


class ToolUseEntry(SessionEntry):
    """A tool use in the session."""
    type: Literal["tool_use"] = "tool_use"
    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any]


class ToolResultEntry(SessionEntry):
    """A tool result in the session."""
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: str
    is_error: bool = False


class MetadataEntry(SessionEntry):
    """Session metadata."""
    type: Literal["metadata"] = "metadata"
    key: str
    value: Any


class SessionSummary(BaseModel):
    """Summary of a session for listing."""
    session_id: str
    name: str | None = None
    created_at: float
    updated_at: float
    message_count: int
    model: str | None = None


class SessionManager:
    """JSONL session persistence manager."""

    def __init__(self, sessions_dir: Path | None = None) -> None:
        self._sessions_dir = sessions_dir or SESSIONS_DIR
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        return self._sessions_dir / f"{session_id}.jsonl"

    async def create_session(self, name: str | None = None) -> str:
        session_id = uuid.uuid4().hex[:12]
        path = self._session_path(session_id)

        metadata = MetadataEntry(
            type="metadata",
            timestamp=time.time(),
            key="name",
            value=name or session_id,
        )
        path.write_text(metadata.model_dump_json() + "\n", encoding="utf-8")
        return session_id

    async def load_session(self, session_id: str) -> list[SessionEntry]:
        path = self._session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        entries: list[SessionEntry] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            entry_type = data.get("type")
            if entry_type == "message":
                entries.append(MessageEntry.model_validate(data))
            elif entry_type == "tool_use":
                entries.append(ToolUseEntry.model_validate(data))
            elif entry_type == "tool_result":
                entries.append(ToolResultEntry.model_validate(data))
            elif entry_type == "metadata":
                entries.append(MetadataEntry.model_validate(data))
            else:
                entries.append(SessionEntry.model_validate(data))
        return entries

    async def append_entry(self, session_id: str, entry: SessionEntry) -> None:
        path = self._session_path(session_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

    async def list_sessions(self) -> list[SessionSummary]:
        summaries: list[SessionSummary] = []
        for path in sorted(self._sessions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            session_id = path.stem
            try:
                entries = await self.load_session(session_id)
                name = None
                model = None
                msg_count = 0
                created_at = 0.0
                updated_at = 0.0

                for entry in entries:
                    if created_at == 0.0:
                        created_at = entry.timestamp
                    updated_at = entry.timestamp
                    if isinstance(entry, MetadataEntry) and entry.key == "name":
                        name = str(entry.value)
                    elif isinstance(entry, MetadataEntry) and entry.key == "model":
                        model = str(entry.value)
                    elif isinstance(entry, MessageEntry):
                        msg_count += 1

                summaries.append(SessionSummary(
                    session_id=session_id,
                    name=name,
                    created_at=created_at,
                    updated_at=updated_at,
                    message_count=msg_count,
                    model=model,
                ))
            except Exception:
                continue
        return summaries

    async def delete_session(self, session_id: str) -> None:
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()

    async def fork_session(self, session_id: str, at_entry: int | None = None) -> str:
        entries = await self.load_session(session_id)
        if at_entry is not None:
            entries = entries[:at_entry]

        new_id = await self.create_session(name=f"fork-{session_id[:8]}")
        for entry in entries:
            await self.append_entry(new_id, entry)
        return new_id
