from __future__ import annotations

import asyncio
import shutil
from typing import Any

from pi_mono.agent.types import AgentTool, AgentToolResult
from pi_mono.ai.types import TextContent

GREP_TOOL_NAME = "grep"
GREP_TOOL_DESCRIPTION = "Search for a pattern in files using ripgrep."
GREP_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Regex pattern to search for."},
        "path": {"type": "string", "description": "Directory or file to search in."},
        "glob": {"type": "string", "description": "File glob filter (e.g., '*.py')."},
        "case_insensitive": {"type": "boolean", "description": "Case insensitive search."},
        "max_results": {"type": "integer", "description": "Maximum results. Default 50."},
    },
    "required": ["pattern"],
}


async def execute_grep(
    tool_call_id: str,
    params: dict[str, Any],
    signal: asyncio.Event | None = None,
    on_update: Any = None,
) -> AgentToolResult:
    pattern = params["pattern"]
    path = params.get("path", ".")
    glob_filter = params.get("glob")
    case_insensitive = params.get("case_insensitive", False)
    max_results = params.get("max_results", 50)

    # Try ripgrep first, fall back to grep
    rg = shutil.which("rg")

    cmd_parts = []
    if rg:
        cmd_parts = [rg, "--no-heading", "--line-number", "--color=never", f"--max-count={max_results}"]
        if case_insensitive:
            cmd_parts.append("-i")
        if glob_filter:
            cmd_parts.extend(["--glob", glob_filter])
        cmd_parts.extend(["--", pattern, path])
    else:
        cmd_parts = ["grep", "-rn", f"--max-count={max_results}"]
        if case_insensitive:
            cmd_parts.append("-i")
        if glob_filter:
            cmd_parts.extend(["--include", glob_filter])
        cmd_parts.extend(["--", pattern, path])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, _stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=30)

        stdout = stdout_bytes.decode("utf-8", errors="replace")

        if not stdout.strip():
            return AgentToolResult(
                content=[TextContent(text="No matches found.")],
                details={"matches": 0},
            )

        # Truncate
        if len(stdout) > 30000:
            stdout = stdout[:30000] + "\n... (truncated)"

        return AgentToolResult(
            content=[TextContent(text=stdout)],
            details={"matches": stdout.count("\n")},
        )
    except TimeoutError:
        return AgentToolResult(
            content=[TextContent(text="Search timed out after 30s")],
            details={"error": "timeout"},
        )
    except Exception as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error searching: {e}")],
            details={"error": str(e)},
        )


def create_grep_tool() -> AgentTool:
    return AgentTool(
        name=GREP_TOOL_NAME,
        description=GREP_TOOL_DESCRIPTION,
        label="Grep",
        parameters=GREP_TOOL_SCHEMA,
        execute=execute_grep,
    )
