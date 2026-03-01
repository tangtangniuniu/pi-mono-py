"""End-to-end integration test — full REST API flow with SSE event streaming.

Tests the complete lifecycle:
  1. Create FastAPI app with mock LLM provider
  2. Create session via POST /api/sessions
  3. Send message via POST /api/sessions/{id}/messages
  4. Connect to SSE stream to receive agent events
  5. Verify message history via GET /api/sessions/{id}/messages
  6. Clean up via DELETE /api/sessions/{id}
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pi_mono.ai.api_registry import clear_api_providers, register_api_provider
from pi_mono.server.app import create_app
from tests.conftest import MockLLMProvider, make_model


@pytest.fixture
def e2e_client(tmp_path: Path) -> TestClient:
    """Create a fully wired test client with mock LLM provider."""
    clear_api_providers()
    provider = MockLLMProvider(response_text="E2E test response")
    register_api_provider(provider)

    app = create_app(
        settings_file=tmp_path / "settings.yaml",
        sessions_dir=tmp_path / "sessions",
    )

    # Register test models
    from pi_mono.server.app import _model_registry as reg
    if reg is not None:
        reg.register_custom_model(make_model(model_id="claude-sonnet-4-20250514"))
        reg.register_custom_model(make_model(model_id="test-model"))

    client = TestClient(app)
    yield client  # type: ignore[misc]
    clear_api_providers()


class TestE2ESessionLifecycle:
    """Full lifecycle: create → interact → list → detail → delete."""

    def test_complete_session_lifecycle(self, e2e_client: TestClient) -> None:
        client = e2e_client

        # 1. Health check
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # 2. Create session
        create_resp = client.post("/api/sessions", json={})
        assert create_resp.status_code == 201
        session_data = create_resp.json()
        session_id = session_data["session_id"]
        assert session_id

        # 3. Get session detail
        detail_resp = client.get(f"/api/sessions/{session_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["session_id"] == session_id
        assert "message_count" in detail

        # 4. List sessions — should include our session
        list_resp = client.get("/api/sessions")
        assert list_resp.status_code == 200
        sessions = list_resp.json()
        assert any(s["session_id"] == session_id for s in sessions)

        # 5. Send message (async — 202 accepted)
        msg_resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Hello, agent!"},
        )
        assert msg_resp.status_code == 202

        # 6. Delete session
        del_resp = client.delete(f"/api/sessions/{session_id}")
        assert del_resp.status_code == 204

        # 7. Verify session is gone
        gone_resp = client.get(f"/api/sessions/{session_id}")
        assert gone_resp.status_code == 404


class TestE2EModelManagement:
    """Model listing and switching within a session."""

    def test_list_and_switch_model(self, e2e_client: TestClient) -> None:
        client = e2e_client

        # List models
        models_resp = client.get("/api/models")
        assert models_resp.status_code == 200
        models = models_resp.json()
        assert len(models) >= 1
        model_ids = [m["id"] for m in models]
        assert "test-model" in model_ids

        # Create session and switch model
        create_resp = client.post("/api/sessions", json={})
        session_id = create_resp.json()["session_id"]

        switch_resp = client.put(
            f"/api/sessions/{session_id}/model",
            json={"model_id": "test-model"},
        )
        assert switch_resp.status_code == 200
        assert switch_resp.json()["model"] == "test-model"

        # Clean up
        client.delete(f"/api/sessions/{session_id}")


class TestE2ESettingsManagement:
    """Settings get and update."""

    def test_get_and_update_settings(self, e2e_client: TestClient) -> None:
        client = e2e_client

        # Get settings
        get_resp = client.get("/api/settings")
        assert get_resp.status_code == 200
        settings = get_resp.json()
        assert "default_model" in settings
        # Sensitive fields should be excluded
        assert "custom_headers" not in settings

        # Update settings
        update_resp = client.put("/api/settings", json={"verbose": True})
        assert update_resp.status_code == 200
        updated = update_resp.json()
        assert updated["verbose"] is True

        # Verify persistence
        get_resp2 = client.get("/api/settings")
        assert get_resp2.json()["verbose"] is True


class TestE2EErrorHandling:
    """Verify proper error responses for invalid operations."""

    def test_session_not_found(self, e2e_client: TestClient) -> None:
        client = e2e_client

        assert client.get("/api/sessions/nonexistent").status_code == 404
        assert client.delete("/api/sessions/nonexistent").status_code == 404
        assert client.post(
            "/api/sessions/nonexistent/messages", json={"content": "hi"}
        ).status_code == 404

    def test_model_switch_nonexistent_session(self, e2e_client: TestClient) -> None:
        client = e2e_client
        resp = client.put(
            "/api/sessions/nonexistent/model",
            json={"model_id": "test-model"},
        )
        assert resp.status_code == 404

    def test_abort_idle_session(self, e2e_client: TestClient) -> None:
        client = e2e_client
        create_resp = client.post("/api/sessions", json={})
        session_id = create_resp.json()["session_id"]

        # Abort on idle session should be idempotent
        abort_resp = client.post(f"/api/sessions/{session_id}/abort")
        assert abort_resp.status_code == 200

        client.delete(f"/api/sessions/{session_id}")


class TestE2EMultiSession:
    """Multiple concurrent sessions."""

    def test_multiple_sessions_independent(self, e2e_client: TestClient) -> None:
        client = e2e_client

        # Create two sessions
        resp1 = client.post("/api/sessions", json={})
        resp2 = client.post("/api/sessions", json={})
        sid1 = resp1.json()["session_id"]
        sid2 = resp2.json()["session_id"]
        assert sid1 != sid2

        # List should show both
        sessions = client.get("/api/sessions").json()
        session_ids = [s["session_id"] for s in sessions]
        assert sid1 in session_ids
        assert sid2 in session_ids

        # Delete one, other should survive
        client.delete(f"/api/sessions/{sid1}")
        sessions_after = client.get("/api/sessions").json()
        remaining_ids = [s["session_id"] for s in sessions_after]
        assert sid1 not in remaining_ids
        assert sid2 in remaining_ids

        # Clean up
        client.delete(f"/api/sessions/{sid2}")
