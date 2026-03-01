from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pi_mono.agent.types import AgentTool, AgentToolResult
from pi_mono.ai.types import TextContent

BASH_TOOL_NAME = "bash"
BASH_TOOL_DESCRIPTION = "Execute a shell command and return stdout/stderr."
BASH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "The shell command to execute.",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds. Default 120.",
        },
    },
    "required": ["command"],
}


@dataclass(frozen=True)
class BashResult:
    stdout: str
    stderr: str
    exit_code: int


async def execute_bash(
    tool_call_id: str,
    params: dict[str, Any],
    signal: asyncio.Event | None = None,
    on_update: Any = None,
) -> AgentToolResult:
    command = params["command"]
    timeout = params.get("timeout", 120)

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return AgentToolResult(
                content=[TextContent(text=f"Command timed out after {timeout}s")],
                details=BashResult(stdout="", stderr="", exit_code=-1),
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        exit_code = proc.returncode or 0

        # Truncate long output
        max_len = 30000
        if len(stdout) > max_len:
            stdout = stdout[:max_len] + f"\n... (truncated, {len(stdout)} total chars)"
        if len(stderr) > max_len:
            stderr = stderr[:max_len] + f"\n... (truncated, {len(stderr)} total chars)"

        output_parts = []
        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.append(f"STDERR:\n{stderr}")
        if exit_code != 0:
            output_parts.append(f"Exit code: {exit_code}")

        text = "\n".join(output_parts) if output_parts else "(no output)"

        return AgentToolResult(
            content=[TextContent(text=text)],
            details=BashResult(stdout=stdout, stderr=stderr, exit_code=exit_code),
        )
    except Exception as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error executing command: {e}")],
            details=BashResult(stdout="", stderr=str(e), exit_code=-1),
        )


def create_bash_tool() -> AgentTool:
    return AgentTool(
        name=BASH_TOOL_NAME,
        description=BASH_TOOL_DESCRIPTION,
        label="Shell",
        parameters=BASH_TOOL_SCHEMA,
        execute=execute_bash,
    )
