from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from pi_mono.ai.types import Model, ModelCost
from pi_mono.config.settings import CustomModelConfig

_RUNTIME_MODEL_PREFIX = "PI_RUNTIME_MODEL_"
_RUNTIME_MODEL_REQUIRED_KEYS = {
    "PROVIDER",
    "BASE_URL",
    "ID",
}


@dataclass(frozen=True)
class RuntimeModelConfig:
    provider: str
    base_url: str
    model_id: str
    api_key_env: str
    api: str = "openai-completions"
    name: str | None = None
    context_window: int = 128_000
    max_tokens: int = 4096


def load_dotenv(path: Path) -> dict[str, str]:
    """Parse a simple .env file into key/value pairs."""
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env in {path}:{lineno}: expected KEY=VALUE")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid .env in {path}:{lineno}: empty key")
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def apply_dotenv_to_environ(values: dict[str, str]) -> None:
    """Export .env values into os.environ without overriding explicit env vars."""
    for key, value in values.items():
        os.environ.setdefault(key, value)


def get_runtime_model_config(project_dir: Path | None) -> RuntimeModelConfig | None:
    """Return runtime model configuration from project-local .env when present."""
    values: dict[str, str] = {}
    if project_dir is not None:
        dotenv_path = project_dir / ".env"
        values = load_dotenv(dotenv_path)
        if values:
            apply_dotenv_to_environ(values)

    runtime_values = {
        key.removeprefix(_RUNTIME_MODEL_PREFIX): value
        for key, value in os.environ.items()
        if key.startswith(_RUNTIME_MODEL_PREFIX)
    }
    if not runtime_values:
        return None

    missing = sorted(key for key in _RUNTIME_MODEL_REQUIRED_KEYS if not runtime_values.get(key))
    if missing:
        joined = ", ".join(f"{_RUNTIME_MODEL_PREFIX}{key}" for key in missing)
        raise ValueError(f"Invalid .env runtime model config: missing {joined}")

    provider = runtime_values["PROVIDER"].strip()
    api_key_env = runtime_values.get("API_KEY_ENV", default_api_key_env(provider)).strip()
    name = runtime_values.get("NAME") or None
    api = runtime_values.get("API", "openai-completions").strip() or "openai-completions"

    try:
        context_window = int(runtime_values.get("CONTEXT_WINDOW", "128000"))
        max_tokens = int(runtime_values.get("MAX_TOKENS", "4096"))
    except ValueError as exc:
        raise ValueError("Invalid .env runtime model config: CONTEXT_WINDOW and MAX_TOKENS must be integers") from exc

    if context_window <= 0 or max_tokens <= 0:
        raise ValueError("Invalid .env runtime model config: CONTEXT_WINDOW and MAX_TOKENS must be positive")

    if api_key_env not in values and api_key_env not in os.environ:
        raise ValueError(
            "Invalid .env runtime model config: "
            f"{api_key_env} must be set in .env or the environment for provider {provider}"
        )

    return RuntimeModelConfig(
        provider=provider,
        base_url=runtime_values["BASE_URL"].strip(),
        model_id=runtime_values["ID"].strip(),
        api_key_env=api_key_env,
        api=api,
        name=name,
        context_window=context_window,
        max_tokens=max_tokens,
    )


def default_api_key_env(provider: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", provider.strip().upper()).strip("_")
    return f"{normalized}_API_KEY"


def runtime_model_to_custom_config(config: RuntimeModelConfig) -> CustomModelConfig:
    return CustomModelConfig(
        id=config.model_id,
        provider=config.provider,
        base_url=config.base_url,
        name=config.name,
        api=config.api,
        context_window=config.context_window,
        max_tokens=config.max_tokens,
        api_key_env=config.api_key_env,
    )


def runtime_model_to_model(config: RuntimeModelConfig) -> Model:
    return Model(
        id=config.model_id,
        name=config.name or config.model_id,
        api=config.api,
        provider=config.provider,
        base_url=config.base_url,
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0),
        context_window=config.context_window,
        max_tokens=config.max_tokens,
    )
