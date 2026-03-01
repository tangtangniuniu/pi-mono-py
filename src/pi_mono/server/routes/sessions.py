"""Session CRUD routes — POST/GET/DELETE /api/sessions."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from pi_mono.server.schemas import (
    CreateSessionRequest,
    SessionDetailResponse,
    SessionResponse,
    SessionSummary,
)
from pi_mono.server.session_registry import SessionRegistry

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _get_registry() -> SessionRegistry:
    """Lazy import to avoid circular dependencies."""
    from pi_mono.server.app import get_session_registry
    return get_session_registry()


@router.post("", status_code=201, response_model=SessionResponse)
async def create_session(req: CreateSessionRequest | None = None) -> SessionResponse:
    registry = _get_registry()
    body = req or CreateSessionRequest()
    session = await registry.create(model=body.model, thinking_level=body.thinking_level)
    model_id = None
    if session.agent and session.agent.state.model:
        model_id = session.agent.state.model.id
    sid = session.session_id or ""
    return SessionResponse(session_id=sid, model=model_id)


@router.get("", response_model=list[SessionSummary])
async def list_sessions() -> list[SessionSummary]:
    registry = _get_registry()
    result = []
    for sid in registry.list_sessions():
        session = registry.get(sid)
        msg_count = len(session.agent.state.messages) if session and session.agent else 0
        result.append(SessionSummary(session_id=sid, message_count=msg_count))
    return result


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: str) -> SessionDetailResponse:
    registry = _get_registry()
    session = registry.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    agent = session.agent
    return SessionDetailResponse(
        session_id=session_id,
        model=agent.state.model.id if agent and agent.state.model else None,
        thinking_level=agent.state.thinking_level.value if agent else "off",
        is_streaming=agent.state.is_streaming if agent else False,
        message_count=len(agent.state.messages) if agent else 0,
    )


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str) -> None:
    registry = _get_registry()
    found = await registry.delete(session_id)
    if not found:
        raise HTTPException(status_code=404, detail="Session not found")
