"""Pydantic request/response schemas for the REST API."""

from __future__ import annotations

from pydantic import BaseModel

# --- Session schemas ---

class CreateSessionRequest(BaseModel):
    model: str | None = None
    thinking_level: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    model: str | None = None


class SessionDetailResponse(BaseModel):
    session_id: str
    model: str | None = None
    thinking_level: str = "off"
    is_streaming: bool = False
    message_count: int = 0


class SessionSummary(BaseModel):
    session_id: str
    created_at: float | None = None
    message_count: int = 0


# --- Message schemas ---

class SendMessageRequest(BaseModel):
    content: str


class SteerRequest(BaseModel):
    content: str


# --- Model schemas ---

class ModelInfo(BaseModel):
    id: str
    name: str
    provider: str


class SwitchModelRequest(BaseModel):
    model_id: str


class SetThinkingRequest(BaseModel):
    level: str


# --- Settings schemas ---

class UpdateSettingsRequest(BaseModel):
    """Partial settings update — any field can be provided."""
    model_config = {"extra": "allow"}


# --- Health ---

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


# --- Error ---

class ErrorResponse(BaseModel):
    error: str
