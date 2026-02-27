from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pi_mono.ai.providers.base import LLMProvider

_registry: dict[str, _RegisteredProvider] = {}


@dataclass(frozen=True)
class _RegisteredProvider:
    """Internal wrapper that pairs a provider with an optional source identifier."""

    provider: LLMProvider
    source_id: str | None = None


def register_api_provider(
    provider: LLMProvider,
    source_id: str | None = None,
) -> None:
    """Register an LLM provider, keyed by its ``api`` attribute.

    If a provider with the same ``api`` value already exists it will be
    replaced silently.
    """
    _registry[provider.api] = _RegisteredProvider(
        provider=provider,
        source_id=source_id,
    )


def get_api_provider(api: str) -> LLMProvider | None:
    """Return the provider registered under *api*, or ``None``."""
    entry = _registry.get(api)
    return entry.provider if entry else None


def get_api_providers() -> list[LLMProvider]:
    """Return every registered provider (insertion order)."""
    return [entry.provider for entry in _registry.values()]


def unregister_api_providers(source_id: str) -> None:
    """Remove all providers whose *source_id* matches the given value."""
    keys_to_remove = [
        key
        for key, entry in _registry.items()
        if entry.source_id == source_id
    ]
    for key in keys_to_remove:
        del _registry[key]


def clear_api_providers() -> None:
    """Remove every registered provider."""
    _registry.clear()
