from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pi_mono.agent.types import AgentTool, AgentToolResult
from pi_mono.ai.types import TextContent

if TYPE_CHECKING:
    import asyncio

LS_TOOL_NAME = "ls"
LS_TOOL_DESCRIPTION = "List directory contents."
LS_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Directory path to list."},
    },
    "required": ["path"],
}


async def execute_ls(
    tool_call_id: str,
    params: dict[str, Any],
    signal: asyncio.Event | None = None,
    on_update: Any = None,
) -> AgentToolResult:
    dir_path = Path(params["path"])

    if not dir_path.exists():
        return AgentToolResult(
            content=[TextContent(text=f"Error: path not found: {dir_path}")],
            details={"error": "not_found"},
        )

    if not dir_path.is_dir():
        return AgentToolResult(
            content=[TextContent(text=f"Error: not a directory: {dir_path}")],
            details={"error": "not_dir"},
        )

    try:
        entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = []
        for entry in entries[:500]:
            prefix = "d " if entry.is_dir() else "- "
            try:
                size = entry.stat().st_size if entry.is_file() else 0
                size_str = _format_size(size) if entry.is_file() else ""
            except OSError:
                size_str = ""
            lines.append(f"{prefix}{entry.name:<50} {size_str}")

        result = "\n".join(lines)
        if len(entries) > 500:
            result += f"\n... ({len(entries)} total entries)"

        return AgentToolResult(
            content=[TextContent(text=result or "(empty directory)")],
            details={"entries": len(entries)},
        )
    except Exception as e:
        return AgentToolResult(
            content=[TextContent(text=f"Error: {e}")],
            details={"error": str(e)},
        )


def _format_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:>8.1f} {unit}" if unit != "B" else f"{size:>8d} {unit}"
        size /= 1024
    return f"{size:>8.1f} TB"


def create_ls_tool() -> AgentTool:
    return AgentTool(
        name=LS_TOOL_NAME,
        description=LS_TOOL_DESCRIPTION,
        label="List",
        parameters=LS_TOOL_SCHEMA,
        execute=execute_ls,
    )
