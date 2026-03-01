"""Tests for extension system — ExtensionAPI, ExtensionLoader, ExtensionRunner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from pi_mono.agent.types import AgentToolResult
from pi_mono.ai.types import TextContent

if TYPE_CHECKING:
    from pathlib import Path


class TestExtensionAPI:
    """Tests for ExtensionAPI — the interface extensions use to register capabilities."""

    def test_register_tool(self) -> None:
        from pi_mono.coding_agent.core.extensions.types import ExtensionAPI

        api = ExtensionAPI(source_id="test_ext")

        async def handler(tool_call_id: str, params: dict) -> AgentToolResult:
            return AgentToolResult(content=[TextContent(text="ok")])

        api.register_tool(
            name="my_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            execute=handler,
        )
        assert len(api.tools) == 1
        assert api.tools[0].name == "my_tool"

    def test_register_prompt(self, tmp_path: Path) -> None:
        from pi_mono.coding_agent.core.extensions.types import ExtensionAPI

        api = ExtensionAPI(source_id="test_ext")
        prompt_file = tmp_path / "review.md"
        prompt_file.write_text("# Review Prompt\nCheck code quality.")

        api.register_prompt("review", path=str(prompt_file))
        assert "review" in api.prompts
        assert "Review Prompt" in api.prompts["review"]

    def test_register_prompt_missing_file_raises(self) -> None:
        from pi_mono.coding_agent.core.extensions.types import ExtensionAPI

        api = ExtensionAPI(source_id="test_ext")
        with pytest.raises(FileNotFoundError):
            api.register_prompt("missing", path="/nonexistent/path.md")

    def test_on_message_end(self) -> None:
        from pi_mono.coding_agent.core.extensions.types import ExtensionAPI

        api = ExtensionAPI(source_id="test_ext")
        callback = MagicMock()
        api.on_message_end(callback)
        assert len(api.event_handlers["message_end"]) == 1

    def test_on_turn_end(self) -> None:
        from pi_mono.coding_agent.core.extensions.types import ExtensionAPI

        api = ExtensionAPI(source_id="test_ext")
        callback = MagicMock()
        api.on_turn_end(callback)
        assert len(api.event_handlers["turn_end"]) == 1

    def test_source_id_tracking(self) -> None:
        from pi_mono.coding_agent.core.extensions.types import ExtensionAPI

        api = ExtensionAPI(source_id="my_ext")
        assert api.source_id == "my_ext"


class TestExtensionLoader:
    """Tests for ExtensionLoader — dynamically loads .py extensions from directory."""

    def test_load_successful_extension(self, tmp_path: Path) -> None:
        from pi_mono.coding_agent.core.extensions.loader import ExtensionLoader

        ext_dir = tmp_path / "extensions"
        ext_dir.mkdir()
        (ext_dir / "hello.py").write_text(
            "def create_extension(api):\n"
            "    api.register_tool(\n"
            "        name='hello_tool',\n"
            "        description='Says hello',\n"
            "        parameters={'type': 'object', 'properties': {}},\n"
            "        execute=lambda *a: None,\n"
            "    )\n"
        )

        loader = ExtensionLoader(extensions_dir=ext_dir)
        results = loader.load_all()
        assert len(results) == 1
        assert results[0].source_id == "hello"
        assert len(results[0].api.tools) == 1

    def test_load_extension_with_error(self, tmp_path: Path) -> None:
        from pi_mono.coding_agent.core.extensions.loader import ExtensionLoader

        ext_dir = tmp_path / "extensions"
        ext_dir.mkdir()
        (ext_dir / "bad.py").write_text("raise RuntimeError('broken')\n")

        loader = ExtensionLoader(extensions_dir=ext_dir)
        results = loader.load_all()
        assert len(results) == 0  # error is logged, not propagated

    def test_load_no_directory(self, tmp_path: Path) -> None:
        from pi_mono.coding_agent.core.extensions.loader import ExtensionLoader

        loader = ExtensionLoader(extensions_dir=tmp_path / "nonexistent")
        results = loader.load_all()
        assert results == []

    def test_load_skips_non_py_files(self, tmp_path: Path) -> None:
        from pi_mono.coding_agent.core.extensions.loader import ExtensionLoader

        ext_dir = tmp_path / "extensions"
        ext_dir.mkdir()
        (ext_dir / "readme.txt").write_text("not a python file")
        (ext_dir / "valid.py").write_text(
            "def create_extension(api): pass\n"
        )

        loader = ExtensionLoader(extensions_dir=ext_dir)
        results = loader.load_all()
        assert len(results) == 1

    def test_load_skips_module_without_create_extension(self, tmp_path: Path) -> None:
        from pi_mono.coding_agent.core.extensions.loader import ExtensionLoader

        ext_dir = tmp_path / "extensions"
        ext_dir.mkdir()
        (ext_dir / "no_entry.py").write_text("x = 42\n")

        loader = ExtensionLoader(extensions_dir=ext_dir)
        results = loader.load_all()
        assert len(results) == 0


class TestExtensionRunner:
    """Tests for ExtensionRunner — manages lifecycle, tool injection, event dispatch."""

    def test_collect_tools_from_extensions(self, tmp_path: Path) -> None:
        from pi_mono.coding_agent.core.extensions.runner import ExtensionRunner
        from pi_mono.coding_agent.core.extensions.types import ExtensionAPI

        api = ExtensionAPI(source_id="ext1")
        api.register_tool(
            name="ext_tool",
            description="Extension tool",
            parameters={"type": "object", "properties": {}},
            execute=MagicMock(),
        )

        runner = ExtensionRunner()
        runner.add_extension("ext1", api)
        tools = runner.get_all_tools()
        assert any(t.name == "ext_tool" for t in tools)

    def test_tool_name_conflict_with_builtin(self) -> None:
        from pi_mono.coding_agent.core.extensions.runner import ExtensionRunner
        from pi_mono.coding_agent.core.extensions.types import ExtensionAPI

        api = ExtensionAPI(source_id="ext1")
        api.register_tool(
            name="bash",  # conflicts with built-in
            description="Override bash",
            parameters={"type": "object", "properties": {}},
            execute=MagicMock(),
        )

        runner = ExtensionRunner(builtin_tool_names={"bash", "read", "write"})
        runner.add_extension("ext1", api)
        tools = runner.get_all_tools()
        # Extension tool should NOT override built-in
        assert not any(t.name == "bash" for t in tools)

    async def test_dispatch_message_end_event(self) -> None:
        from pi_mono.coding_agent.core.extensions.runner import ExtensionRunner
        from pi_mono.coding_agent.core.extensions.types import ExtensionAPI

        api = ExtensionAPI(source_id="ext1")
        received: list[Any] = []
        api.on_message_end(lambda event: received.append(event))

        runner = ExtensionRunner()
        runner.add_extension("ext1", api)

        await runner.dispatch("message_end", {"type": "message_end"})
        assert len(received) == 1

    async def test_dispatch_async_callback(self) -> None:
        from pi_mono.coding_agent.core.extensions.runner import ExtensionRunner
        from pi_mono.coding_agent.core.extensions.types import ExtensionAPI

        api = ExtensionAPI(source_id="ext1")
        received: list[Any] = []

        async def async_handler(event: Any) -> None:
            received.append(event)

        api.on_message_end(async_handler)

        runner = ExtensionRunner()
        runner.add_extension("ext1", api)
        await runner.dispatch("message_end", {"type": "message_end"})
        assert len(received) == 1

    def test_unload_extension(self) -> None:
        from pi_mono.coding_agent.core.extensions.runner import ExtensionRunner
        from pi_mono.coding_agent.core.extensions.types import ExtensionAPI

        api = ExtensionAPI(source_id="ext1")
        api.register_tool(
            name="ext_tool",
            description="Extension tool",
            parameters={"type": "object", "properties": {}},
            execute=MagicMock(),
        )
        api.on_message_end(lambda e: None)

        runner = ExtensionRunner()
        runner.add_extension("ext1", api)
        assert len(runner.get_all_tools()) == 1

        runner.remove_extension("ext1")
        assert len(runner.get_all_tools()) == 0
