from __future__ import annotations

import json
from pathlib import Path

from pi_mono.coding_agent.config import CREDENTIALS_DIR


class AuthStorage:
    """API key and credential storage."""

    def __init__(self, credentials_dir: Path | None = None) -> None:
        self._dir = credentials_dir or CREDENTIALS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, provider: str) -> Path:
        return self._dir / f"{provider}.json"

    async def get_api_key(self, provider: str) -> str | None:
        path = self._key_path(provider)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("api_key")
        except Exception:
            return None

    async def set_api_key(self, provider: str, key: str) -> None:
        path = self._key_path(provider)
        path.write_text(
            json.dumps({"api_key": key, "provider": provider}),
            encoding="utf-8",
        )
        # Set restrictive permissions
        path.chmod(0o600)

    async def delete_api_key(self, provider: str) -> None:
        path = self._key_path(provider)
        if path.exists():
            path.unlink()

    async def list_providers(self) -> list[str]:
        providers = []
        for path in self._dir.glob("*.json"):
            providers.append(path.stem)
        return sorted(providers)
