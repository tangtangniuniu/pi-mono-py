from __future__ import annotations

from pi_mono.ai.api_registry import get_api_provider, register_api_provider
from pi_mono.ai.providers.openai_compat import OpenAICompletionsProvider


def register_default_providers() -> None:
    """Register built-in API providers required by the runtime.

    The registration is idempotent so repeated bootstrap calls are safe.
    """
    if get_api_provider("openai-completions") is None:
        register_api_provider(OpenAICompletionsProvider())


__all__ = ["register_default_providers", "OpenAICompletionsProvider"]
