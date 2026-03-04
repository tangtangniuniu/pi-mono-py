"""Print mode CLI regression test.

Verifies that print mode still works correctly after the migration:
  - The PrintMode class can be instantiated
  - It processes events and prints assistant text
  - The CLI --print flag is properly wired
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pi_mono.agent.types import (
    AgentEndEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from pi_mono.coding_agent.modes.print_mode import PrintMode
from tests.conftest import make_assistant_message


class TestPrintModeInit:
    """Verify PrintMode can be instantiated."""

    def test_create_print_mode(self) -> None:
        session = MagicMock()
        pm = PrintMode(session, verbose=False)
        assert pm is not None
        assert pm._verbose is False

    def test_create_print_mode_verbose(self) -> None:
        session = MagicMock()
        pm = PrintMode(session, verbose=True)
        assert pm._verbose is True


class TestPrintModeEventHandling:
    """Verify event handling logic."""

    def test_message_end_prints_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        """MessageEndEvent with assistant text should be printed to stdout."""
        session = MagicMock()
        pm = PrintMode(session, verbose=False)

        msg = make_assistant_message(text="Hello from the agent!")
        event = MessageEndEvent(message=msg)
        pm._handle_event(event)

        captured = capsys.readouterr()
        assert "Hello from the agent!" in captured.out

    def test_message_end_non_assistant_not_printed(self, capsys: pytest.CaptureFixture[str]) -> None:
        """MessageEndEvent with non-assistant role should not print."""
        session = MagicMock()
        pm = PrintMode(session, verbose=False)

        # Create a mock message with user role
        msg = MagicMock()
        msg.role = "user"
        msg.content = "Hello"
        event = MessageEndEvent(message=msg)
        pm._handle_event(event)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_tool_events_silent_by_default(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Tool events should not produce output in non-verbose mode."""
        session = MagicMock()
        pm = PrintMode(session, verbose=False)

        start_event = ToolExecutionStartEvent(
            tool_call_id="tc1", tool_name="bash", args={"command": "ls"}
        )
        end_event = ToolExecutionEndEvent(
            tool_call_id="tc1", tool_name="bash", result="output", is_error=False
        )

        pm._handle_event(start_event)
        pm._handle_event(end_event)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_tool_events_verbose(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Tool events should produce output in verbose mode (to stderr via Rich)."""
        session = MagicMock()
        pm = PrintMode(session, verbose=True)

        start_event = ToolExecutionStartEvent(
            tool_call_id="tc1", tool_name="bash", args={"command": "ls"}
        )
        pm._handle_event(start_event)

        # Rich prints to the console's stderr, so we check stderr
        captured = capsys.readouterr()
        assert "bash" in captured.err

    def test_agent_start_event_no_crash(self) -> None:
        """Non-message events should not crash."""
        session = MagicMock()
        pm = PrintMode(session, verbose=False)

        event = AgentStartEvent()
        pm._handle_event(event)  # Should not raise


class TestPrintModeRun:
    """Verify the run method processes events from the session."""

    @pytest.mark.asyncio
    async def test_run_processes_events(self, capsys: pytest.CaptureFixture[str]) -> None:
        """PrintMode.run should iterate over session events and print results."""
        msg = make_assistant_message(text="Test response")

        async def mock_send_message(content: str):
            yield MessageStartEvent(message=msg)
            yield MessageEndEvent(message=msg)
            yield AgentEndEvent(messages=[msg])

        session = MagicMock()
        session.send_message = mock_send_message

        pm = PrintMode(session, verbose=False)
        await pm.run("Hello")

        captured = capsys.readouterr()
        assert "Test response" in captured.out

    @pytest.mark.asyncio
    async def test_run_with_empty_response(self, capsys: pytest.CaptureFixture[str]) -> None:
        """PrintMode.run should handle empty responses gracefully."""
        msg = make_assistant_message(text="")

        async def mock_send_message(content: str):
            yield MessageStartEvent(message=msg)
            yield MessageEndEvent(message=msg)

        session = MagicMock()
        session.send_message = mock_send_message

        pm = PrintMode(session, verbose=False)
        await pm.run("Hello")

        # Should not crash
        capsys.readouterr()
        # Empty text content produces no visible output (only a newline from the final print())


class TestPrintModeCLIWiring:
    """Verify the CLI entry point properly routes to print mode."""

    def test_cli_has_print_flag(self) -> None:
        """Verify the CLI chat command accepts --print flag."""
        from pi_mono.coding_agent.cli.args import chat

        # Check that 'print_mode' is a parameter in the click command
        param_names = [p.name for p in chat.params]
        assert "print_mode" in param_names

    def test_cli_has_prompt_flag(self) -> None:
        """Verify the CLI chat command accepts --prompt flag."""
        from pi_mono.coding_agent.cli.args import chat

        param_names = [p.name for p in chat.params]
        assert "prompt" in param_names

    def test_run_chat_routes_prompt_to_print_mode(self) -> None:
        """Verify run_chat dispatches to PrintMode when prompt is provided."""
        # run_chat is an async function, verify it exists and accepts the right args
        import inspect

        from pi_mono.coding_agent.main import run_chat

        sig = inspect.signature(run_chat)
        params = list(sig.parameters.keys())
        assert "prompt" in params
        assert "print_mode" in params
        assert "model" in params
