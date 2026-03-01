from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pi_mono.ai.types import Tool, ToolCall


def validate_tool_arguments(tool: Tool, tool_call: ToolCall) -> dict[str, Any]:
    """Validate tool call arguments against the tool's JSON Schema parameters.

    Returns a validated copy of the arguments dictionary.  Raises
    :class:`ValueError` when validation fails.

    .. note::

       The current implementation performs basic presence checks for required
       fields only.  Full JSON Schema validation (e.g. via ``jsonschema``) can
       be layered in later without changing the public interface.
    """
    args: dict[str, Any] = dict(tool_call.arguments)
    schema = tool.parameters

    if schema is None:
        return args

    required: list[str] = schema.get("required", [])
    missing = [field for field in required if field not in args]
    if missing:
        raise ValueError(
            f"Tool '{tool.name}' missing required argument(s): {', '.join(missing)}"
        )

    return args
