"""Tests for pi_mono.coding_agent.core.tools — built-in tool set."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pi_mono.coding_agent.core.tools import create_all_tools
from pi_mono.coding_agent.core.tools.bash import execute_bash
from pi_mono.coding_agent.core.tools.edit import execute_edit
from pi_mono.coding_agent.core.tools.find import execute_find
from pi_mono.coding_agent.core.tools.grep import execute_grep
from pi_mono.coding_agent.core.tools.ls import execute_ls
from pi_mono.coding_agent.core.tools.read import execute_read
from pi_mono.coding_agent.core.tools.write import execute_write

if TYPE_CHECKING:
    from pathlib import Path


class TestReadTool:
    async def test_read_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        result = await execute_read("c1", {"file_path": str(f)})
        assert "line1" in result.content[0].text
        assert "line2" in result.content[0].text

    async def test_read_nonexistent(self, tmp_path: Path) -> None:
        result = await execute_read("c1", {"file_path": str(tmp_path / "nope.txt")})
        assert "Error" in result.content[0].text

    async def test_read_relative_path_error(self) -> None:
        result = await execute_read("c1", {"file_path": "relative/path.txt"})
        assert "absolute" in result.content[0].text.lower()

    async def test_read_with_offset_and_limit(self, tmp_path: Path) -> None:
        f = tmp_path / "lines.txt"
        f.write_text("\n".join(f"line{i}" for i in range(1, 11)))
        result = await execute_read("c1", {"file_path": str(f), "offset": 3, "limit": 2})
        text = result.content[0].text
        assert "line3" in text
        assert "line4" in text
        assert "line5" not in text


class TestWriteTool:
    async def test_write_creates_file(self, tmp_path: Path) -> None:
        await execute_write(
            "c1", {"file_path": str(tmp_path / "new.txt"), "content": "hello"},
        )
        assert (tmp_path / "new.txt").read_text() == "hello"

    async def test_write_relative_path_error(self) -> None:
        result = await execute_write("c1", {"file_path": "relative.txt", "content": "x"})
        assert "absolute" in result.content[0].text.lower()


class TestEditTool:
    async def test_edit_replaces_text(self, tmp_path: Path) -> None:
        f = tmp_path / "edit.txt"
        f.write_text("foo bar baz")
        await execute_edit(
            "c1", {"file_path": str(f), "old_string": "bar", "new_string": "qux"},
        )
        assert f.read_text() == "foo qux baz"

    async def test_edit_nonexistent_file(self, tmp_path: Path) -> None:
        result = await execute_edit(
            "c1",
            {"file_path": str(tmp_path / "nope.txt"), "old_string": "x", "new_string": "y"},
        )
        assert "error" in result.content[0].text.lower()


class TestFindTool:
    async def test_find_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.txt").write_text("")
        result = await execute_find("c1", {"pattern": "*.py", "path": str(tmp_path)})
        text = result.content[0].text
        assert "a.py" in text

    async def test_find_no_matches(self, tmp_path: Path) -> None:
        result = await execute_find("c1", {"pattern": "*.xyz", "path": str(tmp_path)})
        text = result.content[0].text
        assert "0" in text or "no" in text.lower()


class TestGrepTool:
    async def test_grep_finds_pattern(self, tmp_path: Path) -> None:
        f = tmp_path / "code.py"
        f.write_text("def hello():\n    pass\n# TODO fix\n")
        result = await execute_grep("c1", {"pattern": "TODO", "path": str(tmp_path)})
        text = result.content[0].text
        assert "TODO" in text

    async def test_grep_no_match(self, tmp_path: Path) -> None:
        f = tmp_path / "code.py"
        f.write_text("def hello():\n    pass\n")
        result = await execute_grep("c1", {"pattern": "NONEXISTENT_PATTERN_XYZ", "path": str(tmp_path)})
        # Should succeed without error (just no matches)
        assert result.content[0].text is not None


class TestLsTool:
    async def test_ls_lists_contents(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("")
        (tmp_path / "subdir").mkdir()
        result = await execute_ls("c1", {"path": str(tmp_path)})
        text = result.content[0].text
        assert "file.txt" in text

    async def test_ls_shows_directories(self, tmp_path: Path) -> None:
        (tmp_path / "subdir").mkdir()
        result = await execute_ls("c1", {"path": str(tmp_path)})
        text = result.content[0].text
        assert "subdir" in text


class TestBashTool:
    async def test_bash_echo(self) -> None:
        result = await execute_bash("c1", {"command": "echo hello"})
        assert "hello" in result.content[0].text

    async def test_bash_exit_code(self) -> None:
        result = await execute_bash("c1", {"command": "exit 1"})
        # Should complete without raising, but report the failure
        assert result.content[0].text is not None


class TestCreateAllTools:
    def test_creates_seven_tools(self) -> None:
        tools = create_all_tools()
        assert len(tools) == 7
        names = {t.name for t in tools}
        assert names == {"bash", "read", "write", "edit", "grep", "glob", "ls"}
