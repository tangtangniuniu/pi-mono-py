from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pi_mono.agent.types import AgentTool, AgentToolResult
from pi_mono.ai.types import TextContent

if TYPE_CHECKING:
    import asyncio

EDIT_TOOL_NAME = "edit"
EDIT_TOOL_DESCRIPTION = "Edit a file by replacing an exact string match with new content."
EDIT_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "Absolute path to file."},
        "old_string": {"type": "string", "description": "Exact string to find and replace."},
        "new_string": {"type": "string", "description": "Replacement string."},
        "replace_all": {"type": "boolean", "description": "Replace all occurrences. Default false."},
    },
    "required": ["file_path", "old_string", "new_string"],
}


async def execute_edit(
    tool_call_id: str,
    params: dict[str, Any],
    signal: asyncio.Event | None = None,
    on_update: Any = None,
) -> AgentToolResult:
    file_path = Path(params["file_path"])
    old_string = params["old_string"]
    new_string = params["new_string"]
    replace_all = params.get("replace_all", False)

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

    try:
        content = file_path.read_text(encoding="utf-8")

        count = content.count(old_string)
        if count == 0:
            return AgentToolResult(
                content=[TextContent(text="Error: old_string not found in file")],
                details={"error": "not_found"},
            )

        if not replace_all and count > 1:
            return AgentToolResult(
                content=[TextContent(text=f"Error: old_string found {count} times. Use replace_all=true or provide more context.")],
                details={"error": "ambiguous", "count": count},
            )

        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            new_content = content.replace(old_string, new_string, 1)

        file_path.write_text(new_content, encoding="utf-8")

        return AgentToolResult(
            content=[TextContent(text=f"Successfully edited {file_path} ({count} replacement{'s' if count > 1 else ''})")],
            details={"path": str(file_path), "replacements": count},
        )
    except Exception as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error editing file: {e}")],
            details={"error": str(e)},
        )


def create_edit_tool() -> AgentTool:
    return AgentTool(
        name=EDIT_TOOL_NAME,
        description=EDIT_TOOL_DESCRIPTION,
        label="Edit",
        parameters=EDIT_TOOL_SCHEMA,
        execute=execute_edit,
    )
