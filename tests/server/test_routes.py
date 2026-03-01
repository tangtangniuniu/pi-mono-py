"""Integration tests for REST API routes using FastAPI TestClient."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from pi_mono.ai.api_registry import clear_api_providers, register_api_provider
from pi_mono.server.app import (
    create_app,
)
from tests.conftest import MockLLMProvider, make_model

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    clear_api_providers()
    provider = MockLLMProvider(response_text="API response")
    register_api_provider(provider)

    app = create_app(
        settings_file=tmp_path / "settings.yaml",
        sessions_dir=tmp_path / "sessions",
    )

    # Register a test model so session creation can resolve it
    from pi_mono.server.app import _model_registry as reg
    if reg is not None:
        reg.register_custom_model(make_model(model_id="claude-sonnet-4-20250514"))
        reg.register_custom_model(make_model(model_id="test-model"))

    client = TestClient(app)
    yield client  # type: ignore[misc]
    clear_api_providers()


class TestHealthEndpoint:
    def test_health_check(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestSessionRoutes:
    def test_create_session(self, client: TestClient) -> None:
        resp = client.post("/api/sessions", json={})
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data

    def test_list_sessions_empty(self, client: TestClient) -> None:
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_sessions_after_create(self, client: TestClient) -> None:
        client.post("/api/sessions", json={})
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_get_session_detail(self, client: TestClient) -> None:
        create_resp = client.post("/api/sessions", json={})
        sid = create_resp.json()["session_id"]
        resp = client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == sid

    def test_get_session_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/sessions/nonexistent")
        assert resp.status_code == 404

    def test_delete_session(self, client: TestClient) -> None:
        create_resp = client.post("/api/sessions", json={})
        sid = create_resp.json()["session_id"]
        del_resp = client.delete(f"/api/sessions/{sid}")
        assert del_resp.status_code == 204

        # Verify it's gone
        resp = client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 404

    def test_delete_session_not_found(self, client: TestClient) -> None:
        resp = client.delete("/api/sessions/nonexistent")
        assert resp.status_code == 404


class TestMessageRoutes:
    def test_send_message(self, client: TestClient) -> None:
        create_resp = client.post("/api/sessions", json={})
        sid = create_resp.json()["session_id"]
        resp = client.post(f"/api/sessions/{sid}/messages", json={"content": "hello"})
        assert resp.status_code == 202

    def test_send_message_session_not_found(self, client: TestClient) -> None:
        resp = client.post("/api/sessions/nonexistent/messages", json={"content": "hi"})
        assert resp.status_code == 404

    def test_steer_message(self, client: TestClient) -> None:
        create_resp = client.post("/api/sessions", json={})
        sid = create_resp.json()["session_id"]
        resp = client.post(f"/api/sessions/{sid}/steer", json={"content": "change direction"})
        assert resp.status_code == 202

    def test_abort_session(self, client: TestClient) -> None:
        create_resp = client.post("/api/sessions", json={})
        sid = create_resp.json()["session_id"]
        resp = client.post(f"/api/sessions/{sid}/abort")
        assert resp.status_code == 200

    def test_abort_idle_session(self, client: TestClient) -> None:
        create_resp = client.post("/api/sessions", json={})
        sid = create_resp.json()["session_id"]
        # Abort when idle should still return 200 (idempotent)
        resp = client.post(f"/api/sessions/{sid}/abort")
        assert resp.status_code == 200

    def test_get_messages_empty(self, client: TestClient) -> None:
        create_resp = client.post("/api/sessions", json={})
        sid = create_resp.json()["session_id"]
        resp = client.get(f"/api/sessions/{sid}/messages")
        assert resp.status_code == 200
        assert resp.json() == []


class TestModelRoutes:
    def test_list_models(self, client: TestClient) -> None:
        resp = client.get("/api/models")
        assert resp.status_code == 200
        models = resp.json()
        assert len(models) >= 1
        assert any(m["id"] == "test-model" for m in models)

    def test_switch_model(self, client: TestClient) -> None:
        create_resp = client.post("/api/sessions", json={})
        sid = create_resp.json()["session_id"]
        resp = client.put(f"/api/sessions/{sid}/model", json={"model_id": "test-model"})
        assert resp.status_code == 200

    def test_switch_model_not_found(self, client: TestClient) -> None:
        resp = client.put("/api/sessions/nonexistent/model", json={"model_id": "test-model"})
        assert resp.status_code == 404


class TestSettingsRoutes:
    def test_get_settings(self, client: TestClient) -> None:
        resp = client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "default_model" in data
        # custom_headers should be excluded
        assert "custom_headers" not in data

    def test_update_settings(self, client: TestClient) -> None:
        resp = client.put("/api/settings", json={"verbose": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["verbose"] is True
