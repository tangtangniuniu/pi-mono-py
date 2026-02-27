from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pi_mono.ai.types import Cost, Model, Usage

# provider -> model_id -> Model
_model_registry: dict[str, dict[str, Model]] = {}


def register_models(provider: str, models: dict[str, Model]) -> None:
    """Register a batch of models under *provider*.

    Replaces any previously registered models for the same provider.
    """
    _model_registry[provider] = dict(models)


def get_model(provider: str, model_id: str) -> Model | None:
    """Look up a single model by *provider* and *model_id*."""
    provider_models = _model_registry.get(provider)
    if provider_models is None:
        return None
    return provider_models.get(model_id)


def get_providers() -> list[str]:
    """Return the names of all providers that have registered models."""
    return list(_model_registry.keys())


def get_models(provider: str) -> list[Model]:
    """Return every model registered under *provider*."""
    models = _model_registry.get(provider)
    return list(models.values()) if models else []


def calculate_cost(model: Model, usage: Usage) -> Cost:
    """Calculate dollar cost from model pricing and token usage.

    Pricing values in ``model.cost`` are expressed in dollars per million
    tokens.  The returned :class:`Cost` fields are in absolute dollars.
    """
    from pi_mono.ai.types import Cost as _Cost

    input_cost = (model.cost.input / 1_000_000) * usage.input_tokens
    output_cost = (model.cost.output / 1_000_000) * usage.output_tokens
    cache_read_cost = (model.cost.cache_read / 1_000_000) * usage.cache_read_tokens
    cache_write_cost = (model.cost.cache_write / 1_000_000) * usage.cache_write_tokens
    total = input_cost + output_cost + cache_read_cost + cache_write_cost

    return _Cost(
        input=input_cost,
        output=output_cost,
        cache_read=cache_read_cost,
        cache_write=cache_write_cost,
        total=total,
    )


def supports_xhigh(model: Model) -> bool:
    """Return whether *model* supports the ``xhigh`` reasoning tier."""
    if "gpt-5.2" in model.id or "gpt-5.3" in model.id:
        return True
    if model.api == "anthropic-messages":
        return "opus-4-6" in model.id or "opus-4.6" in model.id
    return False


def models_are_equal(a: Model | None, b: Model | None) -> bool:
    """Shallow equality check for two :class:`Model` instances."""
    if a is None or b is None:
        return False
    return a.id == b.id and a.provider == b.provider
