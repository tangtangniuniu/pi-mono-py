from __future__ import annotations

from typing import Any, Protocol

from pi_mono.ai.types import AssistantMessage, Context, Message, Model, SimpleStreamOptions


class CompactionStrategy(Protocol):
    """Protocol for context compaction strategies."""

    async def compact(
        self,
        messages: list[Message],
        model: Model,
        system_prompt: str,
    ) -> list[Message]: ...


class SummaryCompaction:
    """Default compaction — uses LLM to summarize old messages."""

    def __init__(self, stream_fn: Any = None) -> None:
        self._stream_fn = stream_fn

    async def compact(
        self,
        messages: list[Message],
        model: Model,
        system_prompt: str,
    ) -> list[Message]:
        if len(messages) <= 4:
            return messages

        # Keep the last few messages intact
        keep_recent = 4
        old_messages = messages[:-keep_recent]
        recent_messages = messages[-keep_recent:]

        # Build summary request
        import time

        from pi_mono.ai.types import UserMessage

        summary_prompt = (
            "Summarize the following conversation concisely. "
            "Focus on key decisions, code changes, and important context. "
            "Format as a bulleted list."
        )

        # Serialize old messages to text for summarization
        old_text_parts: list[str] = []
        for msg in old_messages:
            if hasattr(msg, 'role'):
                role = msg.role
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, str):
                        old_text_parts.append(f"{role}: {msg.content}")
                    elif isinstance(msg.content, list):
                        texts = [c.text for c in msg.content if hasattr(c, 'text')]
                        if texts:
                            old_text_parts.append(f"{role}: {' '.join(texts)}")

        old_text = "\n".join(old_text_parts)
        if not old_text.strip():
            return messages

        # If we have a stream function, use it to generate summary
        if self._stream_fn is not None:
            try:
                summary_context = Context(
                    system_prompt="You are a conversation summarizer. Be concise and factual.",
                    messages=[
                        UserMessage(
                            role="user",
                            content=f"{summary_prompt}\n\n{old_text}",
                            timestamp=time.time(),
                        )
                    ],
                )

                result: AssistantMessage | None = None
                async for event in self._stream_fn(model, summary_context, SimpleStreamOptions()):
                    if hasattr(event, 'type') and event.type in ("done", "error"):
                        if event.type == "done" and hasattr(event, 'message'):
                            result = event.message
                        break

                if result and result.content:
                    summary_text = ""
                    for c in result.content:
                        if hasattr(c, 'text'):
                            summary_text += c.text

                    if summary_text:
                        summary_msg = UserMessage(
                            role="user",
                            content=f"[Conversation summary]\n{summary_text}",
                            timestamp=time.time(),
                        )
                        return [summary_msg, *recent_messages]
            except Exception:
                pass  # Fall through to simple truncation

        # Fallback: just keep recent messages
        return recent_messages


class NoCompaction:
    """No-op compaction strategy."""

    async def compact(
        self,
        messages: list[Message],
        model: Model,
        system_prompt: str,
    ) -> list[Message]:
        return messages
