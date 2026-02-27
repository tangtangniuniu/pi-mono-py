from __future__ import annotations

from pathlib import Path

APP_NAME = "pi"
CONFIG_DIR = Path.home() / ".pi" / "agent"
SESSIONS_DIR = CONFIG_DIR / "sessions"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
CREDENTIALS_DIR = CONFIG_DIR / "credentials"
EXTENSIONS_DIR = Path.home() / ".pi" / "extensions"
SKILLS_DIR = CONFIG_DIR / "skills"

DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TURNS = 100
DEFAULT_COMPACT_THRESHOLD = 80_000  # tokens

def ensure_dirs() -> None:
    """Create required directories if they don't exist."""
    for d in (CONFIG_DIR, SESSIONS_DIR, CREDENTIALS_DIR):
        d.mkdir(parents=True, exist_ok=True)
