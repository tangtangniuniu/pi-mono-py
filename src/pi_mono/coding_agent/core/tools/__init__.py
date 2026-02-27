from __future__ import annotations

from pi_mono.coding_agent.core.tools.bash import create_bash_tool
from pi_mono.coding_agent.core.tools.edit import create_edit_tool
from pi_mono.coding_agent.core.tools.find import create_find_tool
from pi_mono.coding_agent.core.tools.grep import create_grep_tool
from pi_mono.coding_agent.core.tools.ls import create_ls_tool
from pi_mono.coding_agent.core.tools.read import create_read_tool
from pi_mono.coding_agent.core.tools.write import create_write_tool

ALL_TOOLS = [
    create_bash_tool,
    create_read_tool,
    create_write_tool,
    create_edit_tool,
    create_grep_tool,
    create_find_tool,
    create_ls_tool,
]

def create_all_tools():
    return [fn() for fn in ALL_TOOLS]

__all__ = [
    "create_bash_tool",
    "create_read_tool",
    "create_write_tool",
    "create_edit_tool",
    "create_grep_tool",
    "create_find_tool",
    "create_ls_tool",
    "create_all_tools",
]
