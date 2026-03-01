"""Tests for pi_mono.server.schemas — request/response models."""

from __future__ import annotations

from pi_mono.server.schemas import (
    CreateSessionRequest,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    SendMessageRequest,
    SessionDetailResponse,
    SessionResponse,
    SessionSummary,
    SwitchModelRequest,
    UpdateSettingsRequest,
)


class TestSchemas:
    def test_create_session_request_defaults(self) -> None:
        req = CreateSessionRequest()
        assert req.model is None
        assert req.thinking_level is None

    def test_create_session_request_with_values(self) -> None:
        req = CreateSessionRequest(model="gpt-4o", thinking_level="high")
        assert req.model == "gpt-4o"

    def test_session_response(self) -> None:
        resp = SessionResponse(session_id="abc123", model="gpt-4o")
        assert resp.session_id == "abc123"

    def test_session_detail_response(self) -> None:
        resp = SessionDetailResponse(
            session_id="abc", model="gpt-4o",
            thinking_level="medium", is_streaming=True, message_count=5,
        )
        assert resp.is_streaming is True

    def test_session_summary(self) -> None:
        s = SessionSummary(session_id="abc", message_count=10)
        assert s.message_count == 10

    def test_send_message_request(self) -> None:
        req = SendMessageRequest(content="hello")
        assert req.content == "hello"

    def test_model_info(self) -> None:
        info = ModelInfo(id="gpt-4o", name="GPT-4o", provider="openai")
        data = info.model_dump()
        assert data["id"] == "gpt-4o"

    def test_switch_model_request(self) -> None:
        req = SwitchModelRequest(model_id="gpt-4o-mini")
        assert req.model_id == "gpt-4o-mini"

    def test_health_response(self) -> None:
        resp = HealthResponse()
        assert resp.status == "ok"

    def test_error_response(self) -> None:
        err = ErrorResponse(error="Not found")
        assert err.error == "Not found"

    def test_update_settings_allows_extra(self) -> None:
        req = UpdateSettingsRequest.model_validate({"verbose": True, "theme": "dark"})
        data = req.model_dump()
        assert data["verbose"] is True
