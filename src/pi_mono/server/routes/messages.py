"""Message interaction routes — send, steer, abort, get history."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from pi_mono.ai.types import UserMessage
from pi_mono.server.schemas import SendMessageRequest, SteerRequest
from pi_mono.server.session_registry import SessionRegistry

router = APIRouter(prefix="/api/sessions/{session_id}", tags=["messages"])


def _get_registry() -> SessionRegistry:
    from pi_mono.server.app import get_session_registry
    return get_session_registry()


def _get_session_or_404(session_id: str) -> Any:
    registry = _get_registry()
    session = registry.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/messages", status_code=202)
async def send_message(session_id: str, req: SendMessageRequest) -> dict[str, str]:
    session = _get_session_or_404(session_id)
    if session.agent and session.agent.state.is_streaming:
        raise HTTPException(status_code=409, detail="Agent is busy")

    # Start processing in background
    async def _process() -> None:
        async for _ in session.send_message(req.content):
            pass

    _task = asyncio.create_task(_process())  # noqa: RUF006
    return {"status": "accepted"}


@router.post("/steer", status_code=202)
async def steer_message(session_id: str, req: SteerRequest) -> dict[str, str]:
    session = _get_session_or_404(session_id)
    if session.agent is None:
        raise HTTPException(status_code=400, detail="Agent not initialized")

    msg = UserMessage(content=req.content, timestamp=time.time())
    session.agent.steer(msg)
    return {"status": "accepted"}


@router.post("/abort", status_code=200)
async def abort_session(session_id: str) -> dict[str, str]:
    session = _get_session_or_404(session_id)
    if session.agent:
        session.agent.abort()
    return {"status": "ok"}


@router.get("/messages")
async def get_messages(session_id: str) -> list[dict[str, str]]:
    session = _get_session_or_404(session_id)
    if session.agent is None:
        return []

    messages: list[dict[str, str]] = []
    for msg in session.agent.state.messages:
        entry: dict[str, str] = {"role": getattr(msg, "role", "unknown")}
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            entry["content"] = content
        elif isinstance(content, list):
            texts = [c.text for c in content if hasattr(c, "text")]
            entry["content"] = " ".join(texts)
        else:
            entry["content"] = str(content) if content else ""
        messages.append(entry)
    return messages
