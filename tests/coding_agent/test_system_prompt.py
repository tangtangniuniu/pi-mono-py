"""Tests for pi_mono.coding_agent.core.system_prompt."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pi_mono.agent.types import AgentTool
from pi_mono.coding_agent.core.system_prompt import build_system_prompt, get_git_context


class TestBuildSystemPrompt:
    def test_contains_role_description(self) -> None:
        prompt = build_system_prompt()
        assert "AI coding assistant" in prompt

    def test_contains_environment_info(self) -> None:
        prompt = build_system_prompt()
        assert "Working directory:" in prompt
        assert "Platform:" in prompt
        assert "Date:" in prompt

    def test_contains_guidelines(self) -> None:
        prompt = build_system_prompt()
        assert "Guidelines" in prompt
        assert "Read files before modifying" in prompt

    def test_includes_tool_descriptions(self) -> None:
        mock_tool = AgentTool(
            name="test_tool",
            description="A tool for testing purposes.",
            label="Test",
            parameters={"type": "object", "properties": {}},
            execute=MagicMock(),
        )
        prompt = build_system_prompt(tools=[mock_tool])
        assert "test_tool" in prompt
        assert "A tool for testing purposes." in prompt
        assert "Available Tools" in prompt

    def test_no_tools_section_when_empty(self) -> None:
        prompt = build_system_prompt(tools=None)
        assert "Available Tools" not in prompt

    def test_context_overrides_cwd(self) -> None:
        prompt = build_system_prompt(context={"cwd": "/custom/path"})
        assert "/custom/path" in prompt

    def test_git_branch_included(self) -> None:
        prompt = build_system_prompt(context={"git_branch": "feature-branch"})
        assert "feature-branch" in prompt

    def test_git_repo_status_included(self) -> None:
        prompt = build_system_prompt(context={"is_git_repo": "yes"})
        assert "yes" in prompt


class TestGetGitContext:
    @patch("pi_mono.coding_agent.core.system_prompt.subprocess")
    def test_returns_git_info_in_repo(self, mock_subprocess) -> None:
        # Mock successful git commands
        rev_parse_result = MagicMock()
        rev_parse_result.returncode = 0
        branch_result = MagicMock()
        branch_result.returncode = 0
        branch_result.stdout = "main\n"

        mock_subprocess.run.side_effect = [rev_parse_result, branch_result]
        ctx = get_git_context()
        assert ctx["is_git_repo"] == "yes"
        assert ctx["git_branch"] == "main"

    @patch("pi_mono.coding_agent.core.system_prompt.subprocess")
    def test_returns_no_when_not_in_repo(self, mock_subprocess) -> None:
        result = MagicMock()
        result.returncode = 128
        mock_subprocess.run.return_value = result

        ctx = get_git_context()
        assert ctx["is_git_repo"] == "no"

    @patch("pi_mono.coding_agent.core.system_prompt.subprocess")
    def test_handles_subprocess_error(self, mock_subprocess) -> None:
        mock_subprocess.run.side_effect = OSError("git not found")
        ctx = get_git_context()
        assert ctx["is_git_repo"] == "no"
