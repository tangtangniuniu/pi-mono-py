from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pi_mono.agent.types import AgentTool, AgentToolResult
from pi_mono.ai.types import TextContent

READ_TOOL_NAME = "read"
READ_TOOL_DESCRIPTION = "Read file contents with optional line range."
READ_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "Absolute path to file."},
        "offset": {"type": "integer", "description": "Starting line number (1-based)."},
        "limit": {"type": "integer", "description": "Number of lines to read."},
    },
    "required": ["file_path"],
}


async def execute_read(
    tool_call_id: str,
    params: dict[str, Any],
    signal: asyncio.Event | None = None,
    on_update: Any = None,
) -> AgentToolResult:
    file_path = Path(params["file_path"])
    offset = params.get("offset", 1)
    limit = params.get("limit", 2000)

    if not file_path.is_absolute():
        return AgentToolResult(
            content=[TextContent(text="Error: file_path must be absolute")],
            details={"error": "not_absolute"},
        )

    if not file_path.exists():
        return AgentToolResult(
            content=[TextContent(text=f"Error: file not found: {file_path}")],
            details={"error": "not_found"},
        )

    if not file_path.is_file():
        return AgentToolResult(
            content=[TextContent(text=f"Error: not a file: {file_path}")],
            details={"error": "not_file"},
        )

    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)

        start = max(0, offset - 1)
        end = start + limit
        selected = lines[start:end]

        # Format with line numbers (cat -n style)
        numbered = []
        for i, line in enumerate(selected, start=start + 1):
            line_text = line.rstrip("\n\r")
            # Truncate long lines
            if len(line_text) > 2000:
                line_text = line_text[:2000] + "..."
            numbered.append(f"{i:>6}\t{line_text}")

        result_text = "\n".join(numbered)

        return AgentToolResult(
            content=[TextContent(text=result_text)],
            details={"path": str(file_path), "lines_read": len(selected), "total_lines": len(lines)},
        )
    except Exception as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error reading file: {e}")],
            details={"error": str(e)},
        )


def create_read_tool() -> AgentTool:
    return AgentTool(
        name=READ_TOOL_NAME,
        description=READ_TOOL_DESCRIPTION,
        label="Read",
        parameters=READ_TOOL_SCHEMA,
        execute=execute_read,
    )
