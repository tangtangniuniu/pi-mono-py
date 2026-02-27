from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pi_mono.agent.types import AgentTool, AgentToolResult
from pi_mono.ai.types import TextContent

FIND_TOOL_NAME = "glob"
FIND_TOOL_DESCRIPTION = "Find files matching a glob pattern."
FIND_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.py')."},
        "path": {"type": "string", "description": "Base directory to search from."},
    },
    "required": ["pattern"],
}


async def execute_find(
    tool_call_id: str,
    params: dict[str, Any],
    signal: asyncio.Event | None = None,
    on_update: Any = None,
) -> AgentToolResult:
    pattern = params["pattern"]
    base_path = Path(params.get("path", "."))

    try:
        matches = sorted(base_path.glob(pattern))
        # Limit results
        total = len(matches)
        matches = matches[:200]

        if not matches:
            return AgentToolResult(
                content=[TextContent(text="No files found.")],
                details={"matches": 0},
            )

        result = "\n".join(str(m) for m in matches)
        if total > 200:
            result += f"\n... ({total} total, showing first 200)"

        return AgentToolResult(
            content=[TextContent(text=result)],
            details={"matches": total},
        )
    except Exception as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error: {e}")],
            details={"error": str(e)},
        )


def create_find_tool() -> AgentTool:
    return AgentTool(
        name=FIND_TOOL_NAME,
        description=FIND_TOOL_DESCRIPTION,
        label="Glob",
        parameters=FIND_TOOL_SCHEMA,
        execute=execute_find,
    )
