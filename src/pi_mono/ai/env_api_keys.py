from __future__ import annotations

import os

# Provider name -> ordered list of environment variable names to probe.
# The first non-empty value wins.
_ENV_KEY_MAP: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "cerebras": ["CEREBRAS_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "huggingface": ["HF_TOKEN", "HUGGINGFACE_API_KEY"],
}


def get_env_api_key(provider: str) -> str | None:
    """Return the first non-empty API key found in the environment for *provider*.

    Returns ``None`` when the provider is unknown or no matching variable is
    set.
    """
    for var in _ENV_KEY_MAP.get(provider, []):
        val = os.environ.get(var)
        if val:
            return val
    return None
