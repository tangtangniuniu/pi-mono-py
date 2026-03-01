"""FastAPI application assembly — routes, CORS, health check, dependency wiring."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pi_mono.coding_agent.core.model_registry import ModelRegistry
from pi_mono.coding_agent.core.session_manager import SessionManager
from pi_mono.coding_agent.core.settings_manager import SettingsManager
from pi_mono.server.schemas import HealthResponse
from pi_mono.server.session_registry import SessionRegistry

if TYPE_CHECKING:
    from pathlib import Path

# Module-level singletons (set during app creation)
_session_registry: SessionRegistry | None = None
_model_registry: ModelRegistry | None = None
_settings_manager: SettingsManager | None = None


def get_session_registry() -> SessionRegistry:
    assert _session_registry is not None, "App not initialized"
    return _session_registry


def get_model_registry() -> ModelRegistry:
    assert _model_registry is not None, "App not initialized"
    return _model_registry


def get_settings_manager() -> SettingsManager:
    assert _settings_manager is not None, "App not initialized"
    return _settings_manager


def create_app(
    settings_file: Path | None = None,
    sessions_dir: Path | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    global _session_registry, _model_registry, _settings_manager

    _settings_manager = SettingsManager(settings_file=settings_file)
    session_manager = SessionManager(sessions_dir=sessions_dir)
    _model_registry = ModelRegistry()
    _session_registry = SessionRegistry(
        settings_manager=_settings_manager,
        session_manager=session_manager,
        model_registry=_model_registry,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        yield
        if _session_registry:
            await _session_registry.close_all()

    app = FastAPI(title="Pi Agent Server", version="0.1.0", lifespan=lifespan)

    # CORS
    origins = cors_origins or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check
    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    # Register routes
    from pi_mono.server.routes.messages import router as messages_router
    from pi_mono.server.routes.models import router as models_router
    from pi_mono.server.routes.sessions import router as sessions_router
    from pi_mono.server.routes.settings import router as settings_router

    app.include_router(sessions_router)
    app.include_router(messages_router)
    app.include_router(models_router)
    app.include_router(settings_router)

    return app
