from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from pi_mono.agent.agent import Agent
from pi_mono.agent.types import (
    AgentEvent,
    AgentThinkingLevel,
    MessageEndEvent,
)
from pi_mono.coding_agent.core.compaction import CompactionStrategy, SummaryCompaction
from pi_mono.coding_agent.core.extensions.loader import ExtensionLoader
from pi_mono.coding_agent.core.extensions.runner import ExtensionRunner
from pi_mono.coding_agent.core.model_resolver import ModelResolver
from pi_mono.coding_agent.core.session_manager import MessageEntry, SessionManager
from pi_mono.coding_agent.core.tools import create_all_tools

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from pi_mono.coding_agent.core.model_registry import ModelRegistry
    from pi_mono.coding_agent.core.settings_manager import Settings, SettingsManager


class AgentSession:
    """Session lifecycle management — the core hub."""

    def __init__(
        self,
        settings_manager: SettingsManager,
        session_manager: SessionManager,
        model_registry: ModelRegistry,
        compaction: CompactionStrategy | None = None,
        stream_fn: Any = None,
    ) -> None:
        self._settings_manager = settings_manager
        self._session_manager = session_manager
        self._model_registry = model_registry
        self._model_resolver = ModelResolver(model_registry)
        self._compaction = compaction or SummaryCompaction(stream_fn=stream_fn)
        self._stream_fn = stream_fn
        self._extension_runner: ExtensionRunner | None = None

        self._agent: Agent | None = None
        self._session_id: str | None = None
        self._settings: Settings | None = None

    @property
    def agent(self) -> Agent | None:
        return self._agent

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def start(
        self,
        session_id: str | None = None,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> None:
        self._settings = await self._settings_manager.load()

        # Resolve model
        resolved_model = self._model_resolver.resolve(model, self._settings)

        # Create or resume session
        if session_id:
            self._session_id = session_id
        else:
            self._session_id = await self._session_manager.create_session()

        # Determine thinking level
        tl = thinking_level or self._settings.thinking_level
        agent_thinking = AgentThinkingLevel(tl) if tl != "off" else AgentThinkingLevel.OFF

        # Create tools
        tools = create_all_tools()
        builtin_names = {t.name for t in tools}

        # Load extensions
        loader = ExtensionLoader()
        loaded = loader.load_all()
        self._extension_runner = ExtensionRunner(builtin_tool_names=builtin_names)
        for ext in loaded:
            self._extension_runner.add_extension(ext.source_id, ext.api)

        # Merge extension tools
        ext_tools = self._extension_runner.get_all_tools()
        all_tools = tools + ext_tools

        # Create agent
        self._agent = Agent(
            stream_fn=self._stream_fn,
            steering_mode="one-at-a-time",
            follow_up_mode="one-at-a-time",
            session_id=self._session_id,
        )
        self._agent.set_model(resolved_model)
        self._agent.set_thinking_level(agent_thinking)
        self._agent.set_tools(all_tools)

    async def send_message(self, content: str) -> AsyncGenerator[AgentEvent, None]:
        if self._agent is None:
            raise RuntimeError("Session not started")

        events: list[AgentEvent] = []

        def on_event(event: AgentEvent) -> None:
            events.append(event)

        unsub = self._agent.subscribe(on_event)

        try:
            # Run prompt in background
            task = asyncio.create_task(self._agent.prompt(content))

            # Yield events as they arrive
            last_idx = 0
            while not task.done() or last_idx < len(events):
                if last_idx < len(events):
                    yield events[last_idx]

                    # Persist message_end events
                    event = events[last_idx]
                    if isinstance(event, MessageEndEvent) and self._session_id:
                        await self._session_manager.append_entry(
                            self._session_id,
                            MessageEntry(
                                timestamp=time.time(),
                                message=_serialize_message(event.message),
                            ),
                        )
                    last_idx += 1
                else:
                    await asyncio.sleep(0.01)

            # Drain remaining events
            while last_idx < len(events):
                yield events[last_idx]
                last_idx += 1

            await task
        finally:
            unsub()

    async def switch_model(self, model_id: str) -> None:
        if self._agent is None:
            raise RuntimeError("Session not started")
        if self._settings is None:
            self._settings = await self._settings_manager.load()

        resolved = self._model_resolver.resolve(model_id, self._settings)
        self._agent.set_model(resolved)

    async def compact(self) -> None:
        if self._agent is None:
            raise RuntimeError("Session not started")

        messages = self._agent.state.messages
        model = self._agent.state.model
        if model is None:
            return

        compacted = await self._compaction.compact(
            messages=[m for m in messages if hasattr(m, 'role')],
            model=model,
            system_prompt=self._agent.state.system_prompt,
        )
        self._agent.replace_messages(compacted)

    async def fork(self) -> str:
        if self._session_id is None:
            raise RuntimeError("No active session")
        return await self._session_manager.fork_session(self._session_id)

    async def close(self) -> None:
        if self._agent is not None:
            self._agent.abort()
            await self._agent.wait_for_idle()
        self._agent = None
        self._session_id = None


def _serialize_message(msg: Any) -> dict[str, Any]:
    """Serialize an AgentMessage to dict for JSONL storage."""
    if hasattr(msg, '__dict__'):
        result: dict[str, Any] = {}
        for k, v in msg.__dict__.items():
            if not k.startswith('_'):
                if hasattr(v, '__dict__'):
                    result[k] = _serialize_message(v)
                elif isinstance(v, list):
                    result[k] = [_serialize_message(item) if hasattr(item, '__dict__') else item for item in v]
                else:
                    result[k] = v
        return result
    return {"value": str(msg)}
