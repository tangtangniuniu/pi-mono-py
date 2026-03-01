"""In-process session registry — maps session IDs to AgentSession instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pi_mono.coding_agent.core.agent_session import AgentSession

if TYPE_CHECKING:
    from pi_mono.coding_agent.core.model_registry import ModelRegistry
    from pi_mono.coding_agent.core.session_manager import SessionManager
    from pi_mono.coding_agent.core.settings_manager import SettingsManager


class SessionRegistry:
    """Thread-safe in-process registry for active AgentSession instances."""

    def __init__(
        self,
        settings_manager: SettingsManager,
        session_manager: SessionManager,
        model_registry: ModelRegistry,
    ) -> None:
        self._settings_manager = settings_manager
        self._session_manager = session_manager
        self._model_registry = model_registry
        self._sessions: dict[str, AgentSession] = {}

    async def create(
        self,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> AgentSession:
        """Create a new AgentSession, start it, and register it."""
        session = AgentSession(
            settings_manager=self._settings_manager,
            session_manager=self._session_manager,
            model_registry=self._model_registry,
        )
        await session.start(model=model, thinking_level=thinking_level)
        session_id = session.session_id
        if session_id is None:
            raise RuntimeError("Session failed to start — no session ID")
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> AgentSession | None:
        """Look up a session by ID. Returns None if not found."""
        return self._sessions.get(session_id)

    async def delete(self, session_id: str) -> bool:
        """Close and remove a session. Returns True if found, False otherwise."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        await session.close()
        return True

    def list_sessions(self) -> list[str]:
        """Return all active session IDs."""
        return list(self._sessions.keys())

    async def close_all(self) -> None:
        """Close all active sessions."""
        for session_id in list(self._sessions.keys()):
            await self.delete(session_id)
