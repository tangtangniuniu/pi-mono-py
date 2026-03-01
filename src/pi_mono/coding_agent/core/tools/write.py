from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pi_mono.agent.types import AgentTool, AgentToolResult
from pi_mono.ai.types import TextContent

if TYPE_CHECKING:
    import asyncio

WRITE_TOOL_NAME = "write"
WRITE_TOOL_DESCRIPTION = "Create or overwrite a file with the given content."
WRITE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "Absolute path to file."},
        "content": {"type": "string", "description": "Content to write."},
    },
    "required": ["file_path", "content"],
}


async def execute_write(
    tool_call_id: str,
    params: dict[str, Any],
    signal: asyncio.Event | None = None,
    on_update: Any = None,
) -> AgentToolResult:
    file_path = Path(params["file_path"])
    content = params["content"]

    if not file_path.is_absolute():
        return AgentToolResult(
            content=[TextContent(text="Error: file_path must be absolute")],
            details={"error": "not_absolute"},
        )

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

        return AgentToolResult(
            content=[TextContent(text=f"Successfully wrote {len(content)} bytes to {file_path}")],
            details={"path": str(file_path), "bytes_written": len(content)},
        )
    except Exception as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error writing file: {e}")],
            details={"error": str(e)},
        )


def create_write_tool() -> AgentTool:
    return AgentTool(
        name=WRITE_TOOL_NAME,
        description=WRITE_TOOL_DESCRIPTION,
        label="Write",
        parameters=WRITE_TOOL_SCHEMA,
        execute=execute_write,
    )
