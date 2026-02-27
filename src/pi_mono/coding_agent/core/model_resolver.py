from __future__ import annotations

from pi_mono.ai.types import Model
from pi_mono.coding_agent.config import DEFAULT_MODEL
from pi_mono.coding_agent.core.model_registry import ModelRegistry
from pi_mono.coding_agent.core.settings_manager import Settings


class ModelResolver:
    """Resolves which model to use based on settings and request."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def resolve(self, requested: str | None, settings: Settings) -> Model:
        model_id = requested or settings.default_model or DEFAULT_MODEL

        model = self._registry.get_model(model_id)
        if model is not None:
            return model

        # Try with common provider prefixes
        for provider in ["anthropic", "openai", "google"]:
            model = self._registry.get_model(model_id, provider)
            if model is not None:
                return model

        raise ValueError(f"Model not found: {model_id}. Use 'models' command to list available models.")
