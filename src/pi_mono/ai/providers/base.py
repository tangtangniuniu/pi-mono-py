"""LLM provider protocol definition.

Mirrors the TypeScript ``ApiProvider`` interface from
``packages/ai/src/api-registry.ts``.  Concrete provider implementations
(Anthropic, OpenAI, etc.) should satisfy this protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pi_mono.ai.types import (
        AssistantMessageEvent,
        Context,
        Model,
        SimpleStreamOptions,
        StreamOptions,
    )


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that every LLM provider must satisfy.

    A provider is keyed by its :pyattr:`api` identifier (e.g.
    ``"anthropic-messages"``) and exposes two streaming entry-points:

    * :pymeth:`stream` -- full-featured streaming with provider-specific
      options.
    * :pymeth:`stream_simple` -- simplified streaming that accepts a
      :class:`~pi_mono.ai.types.SimpleStreamOptions` (reasoning level,
      etc.).

    Both methods return an async iterator of
    :class:`~pi_mono.ai.types.AssistantMessageEvent` instances.
    """

    @property
    def api(self) -> str:
        """Return the API identifier this provider handles.

        Examples: ``"anthropic-messages"``, ``"openai-completions"``.
        """
        ...

    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """Stream assistant-message events using provider-specific options.

        Parameters
        ----------
        model:
            The resolved model definition (contains id, base_url, etc.).
        context:
            The full conversation context (system prompt, messages, tools).
        options:
            Optional streaming configuration (temperature, max_tokens, ...).

        Returns
        -------
        AsyncIterator[AssistantMessageEvent]
            An async iterator that yields streaming events and can be
            consumed with ``async for``.
        """
        ...

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """Stream assistant-message events with simplified options.

        This entry-point accepts :class:`SimpleStreamOptions` which adds
        reasoning-level control on top of the base stream options.

        Parameters
        ----------
        model:
            The resolved model definition.
        context:
            The full conversation context.
        options:
            Optional simplified streaming configuration (includes
            ``reasoning`` level).

        Returns
        -------
        AsyncIterator[AssistantMessageEvent]
            An async iterator that yields streaming events.
        """
        ...
