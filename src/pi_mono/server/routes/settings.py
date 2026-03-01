"""Settings management routes — GET/PUT /api/settings."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from pi_mono.coding_agent.core.settings_manager import SettingsManager
from pi_mono.server.schemas import UpdateSettingsRequest

router = APIRouter(prefix="/api/settings", tags=["settings"])


def _get_settings_manager() -> SettingsManager:
    from pi_mono.server.app import get_settings_manager
    return get_settings_manager()


@router.get("")
async def get_settings() -> dict[str, Any]:
    mgr = _get_settings_manager()
    settings = await mgr.load()
    # Exclude potentially sensitive fields
    data = settings.model_dump()
    data.pop("custom_headers", None)
    return data


@router.put("")
async def update_settings(req: UpdateSettingsRequest) -> dict[str, Any]:
    mgr = _get_settings_manager()
    kwargs = req.model_dump(exclude_unset=True)
    updated = await mgr.update(**kwargs)
    return updated.model_dump()
