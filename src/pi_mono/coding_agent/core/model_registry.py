from __future__ import annotations

from pi_mono.ai.models import get_model, get_models, get_providers
from pi_mono.ai.types import Model


class ModelRegistry:
    """Model discovery — merges built-in catalog and user-defined models."""

    def __init__(self) -> None:
        self._custom_models: dict[str, Model] = {}

    def register_custom_model(self, model: Model) -> None:
        self._custom_models[model.id] = model

    def get_model(self, model_id: str, provider: str | None = None) -> Model | None:
        # Check custom models first
        if model_id in self._custom_models:
            return self._custom_models[model_id]

        # Search built-in catalog
        if provider:
            return get_model(provider, model_id)

        # Search all providers
        for p in get_providers():
            model = get_model(p, model_id)
            if model is not None:
                return model
        return None

    def list_models(self, provider: str | None = None) -> list[Model]:
        models: list[Model] = []
        if provider:
            models = get_models(provider)
        else:
            for p in get_providers():
                models.extend(get_models(p))
        # Add custom models
        models.extend(self._custom_models.values())
        return models

    def search_models(self, query: str) -> list[Model]:
        query_lower = query.lower()
        results: list[Model] = []
        for model in self.list_models():
            if query_lower in model.id.lower() or query_lower in model.name.lower():
                results.append(model)
        return results
