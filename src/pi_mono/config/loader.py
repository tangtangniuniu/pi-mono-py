"""YAML configuration loader with multi-level priority merging.

Priority (highest wins):
  1. Environment variables (PI_* prefix)
  2. Project-level .pi/settings.yaml
  3. User-level ~/.pi/settings.yaml
  4. Built-in defaults
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import ValidationError

from pi_mono.config.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path

# Environment variable mappings: env_var -> (dotted.path, type_converter)
_ENV_OVERRIDES: dict[str, tuple[str, type]] = {
    "PI_DEFAULT_MODEL": ("models.default", str),
    "PI_THINKING_LEVEL": ("models.thinking_level", str),
    "PI_MAX_TURNS": ("models.max_turns", int),
    "PI_SERVER_HOST": ("server.host", str),
    "PI_SERVER_PORT": ("server.port", int),
    "PI_VERBOSE": ("verbose", bool),
    "PI_THEME": ("theme", str),
    "PI_AUTO_COMPACT": ("auto_compact", bool),
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_nested(data: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set a value at a dotted path in a nested dict, creating intermediaries."""
    parts = dotted_path.split(".")
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _parse_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
        return data
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to the config dict."""
    result = dict(data)
    for env_var, (dotted_path, type_conv) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value is not None:
            converted: Any
            if type_conv is bool:
                converted = _parse_bool(value)
            elif type_conv is int:
                converted = int(value)
            else:
                converted = value
            _set_nested(result, dotted_path, converted)
    return result


async def load_settings(
    project_dir: Path | None = None,
    user_dir: Path | None = None,
) -> Settings:
    """Load settings from YAML files with priority merging.

    Priority: env vars > project .pi/settings.yaml > user ~/.pi/settings.yaml > defaults.
    """
    merged: dict[str, Any] = {}

    # 1. User-level settings (lowest priority file)
    if user_dir is not None:
        user_file = user_dir / ".pi" / "settings.yaml"
        if user_file.exists():
            merged = _deep_merge(merged, _load_yaml_file(user_file))

    # 2. Project-level settings (overrides user)
    if project_dir is not None:
        project_file = project_dir / ".pi" / "settings.yaml"
        if project_file.exists():
            merged = _deep_merge(merged, _load_yaml_file(project_file))

    # 3. Environment variable overrides (highest priority)
    merged = _apply_env_overrides(merged)

    # 4. Validate and return
    try:
        return Settings.model_validate(merged)
    except ValidationError as e:
        raise ValueError(f"Invalid settings: {e}") from e
