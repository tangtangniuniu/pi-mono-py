"""Model management routes — list models, switch model, set thinking."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from pi_mono.coding_agent.core.model_registry import ModelRegistry
from pi_mono.server.schemas import ModelInfo, SetThinkingRequest, SwitchModelRequest
from pi_mono.server.session_registry import SessionRegistry

router = APIRouter(tags=["models"])


def _get_registry() -> tuple[SessionRegistry, ModelRegistry]:
    from pi_mono.server.app import get_model_registry, get_session_registry
    return get_session_registry(), get_model_registry()


@router.get("/api/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    _, model_reg = _get_registry()
    models = model_reg.list_models()
    return [ModelInfo(id=m.id, name=m.name, provider=m.provider) for m in models]


@router.put("/api/sessions/{session_id}/model")
async def switch_model(session_id: str, req: SwitchModelRequest) -> dict[str, str]:
    session_reg, _ = _get_registry()
    session = session_reg.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    await session.switch_model(req.model_id)
    return {"status": "ok", "model": req.model_id}


@router.put("/api/sessions/{session_id}/thinking")
async def set_thinking(session_id: str, req: SetThinkingRequest) -> dict[str, str]:
    session_reg, _ = _get_registry()
    session = session_reg.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.agent is None:
        raise HTTPException(status_code=400, detail="Agent not initialized")

    from pi_mono.agent.types import AgentThinkingLevel
    level = AgentThinkingLevel(req.level) if req.level != "off" else AgentThinkingLevel.OFF
    session.agent.set_thinking_level(level)
    return {"status": "ok", "thinking_level": req.level}
