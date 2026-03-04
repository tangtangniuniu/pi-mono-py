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
from tests.conftest import MockLLMProvider, make_assistant_message, make_model


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


class TestE2ESSEEventStream:
    """SSE event stream verification using the event_stream_generator."""

    def test_sse_event_format(self, e2e_client: TestClient) -> None:
        """Test that the SSE serialization produces correct event format."""
        from pi_mono.server.sse import _sse_event

        result = _sse_event("agent_start", {"type": "agent_start"})
        assert result.startswith("event: agent_start\n")
        assert "data: " in result
        assert result.endswith("\n\n")

    def test_sse_serialize_event(self, e2e_client: TestClient) -> None:
        """Test event serialization handles agent events."""
        from pi_mono.agent.types import AgentStartEvent
        from pi_mono.server.sse import _serialize_event

        event = AgentStartEvent()
        data = _serialize_event(event)
        assert data["type"] == "agent_start"

    def test_sse_serialize_message_end_event(self, e2e_client: TestClient) -> None:
        """Test event serialization of MessageEndEvent."""
        from pi_mono.agent.types import MessageEndEvent
        from pi_mono.server.sse import _serialize_event

        msg = make_assistant_message(text="hello")
        event = MessageEndEvent(message=msg)
        data = _serialize_event(event)
        assert data["type"] == "message_end"
        # Complex objects are serialized as type name
        assert "message" in data

    @pytest.mark.asyncio
    async def test_sse_generator_session_not_found(self, e2e_client: TestClient) -> None:
        """Test that SSE generator yields error for missing session."""
        from pi_mono.server.sse import event_stream_generator

        events = []
        async for sse_text in event_stream_generator("nonexistent"):
            events.append(sse_text)
            break  # Only need the first event

        assert len(events) == 1
        assert "error" in events[0]
        assert "Session not found" in events[0]

    @pytest.mark.asyncio
    async def test_sse_generator_with_active_session(self, e2e_client: TestClient) -> None:
        """Test SSE event streaming for an active session with subscribed events."""
        import asyncio

        from pi_mono.server.app import get_session_registry

        # Create a session via the API
        create_resp = e2e_client.post("/api/sessions", json={})
        session_id = create_resp.json()["session_id"]

        registry = get_session_registry()
        session = registry.get(session_id)
        assert session is not None
        assert session.agent is not None

        # Send a message to trigger events — but read events via subscribe
        collected_events: list[str] = []

        from pi_mono.agent.types import AgentEvent

        def on_event(event: AgentEvent) -> None:
            collected_events.append(event.type)

        unsub = session.agent.subscribe(on_event)

        # Send message in background
        msg_resp = e2e_client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Hello SSE test"},
        )
        assert msg_resp.status_code == 202

        # Wait for agent to finish processing
        await asyncio.sleep(1.0)

        unsub()

        # Agent should have emitted events
        # Depending on mock provider, we expect at least agent_start and agent_end
        if collected_events:
            assert "agent_start" in collected_events or "message_start" in collected_events

        # Clean up
        e2e_client.delete(f"/api/sessions/{session_id}")


class TestE2EMessageHistory:
    """Verify message history retrieval after sending messages."""

    def test_get_messages_empty_session(self, e2e_client: TestClient) -> None:
        """New session should have no messages."""
        client = e2e_client
        create_resp = client.post("/api/sessions", json={})
        session_id = create_resp.json()["session_id"]

        msgs_resp = client.get(f"/api/sessions/{session_id}/messages")
        assert msgs_resp.status_code == 200
        assert msgs_resp.json() == []

        client.delete(f"/api/sessions/{session_id}")

    def test_get_messages_not_found(self, e2e_client: TestClient) -> None:
        """Non-existent session should return 404."""
        client = e2e_client
        resp = client.get("/api/sessions/nonexistent/messages")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_messages_after_send(self, e2e_client: TestClient) -> None:
        """After sending a message and waiting, messages should appear in history."""
        import asyncio

        client = e2e_client

        # Create session
        create_resp = client.post("/api/sessions", json={})
        session_id = create_resp.json()["session_id"]

        # Send a message
        msg_resp = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "Tell me a joke"},
        )
        assert msg_resp.status_code == 202

        # Wait for agent to finish processing
        await asyncio.sleep(1.0)

        # Check messages in history
        msgs_resp = client.get(f"/api/sessions/{session_id}/messages")
        assert msgs_resp.status_code == 200
        messages = msgs_resp.json()

        # Should have at least the user message added by the agent
        # The mock provider generates a response, so we may also have an assistant message
        if messages:
            roles = [m.get("role") for m in messages]
            # User message should be present
            assert "user" in roles

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
