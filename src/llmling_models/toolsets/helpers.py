"""CodeModeToolset helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai import RunContext  # noqa: TC002
from pydantic_ai.toolsets.function import FunctionToolset, FunctionToolsetTool
from schemez.functionschema import FunctionSchema


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic_ai.toolsets import ToolsetTool


def validate_code(python_code: str) -> None:
    """Validate code structure and raise ModelRetry for fixable issues."""
    from pydantic_ai import ModelRetry

    code = python_code.strip()
    if not code:
        msg = "Empty code provided. Please write code inside 'async def main():' function."
        raise ModelRetry(msg)

    if "async def main(" not in code:
        msg = (
            "Code must be wrapped in 'async def main():' function. "
            "Please rewrite your code like:\n"
            "async def main():\n"
            "    # your code here\n"
            "    return result"
        )
        raise ModelRetry(msg)

    # Check if code contains a return statement
    if "return " not in code:
        msg = (
            "The main() function should return a value. "
            "Add 'return result' or 'return \"completed\"' at the end of your function."
        )
        raise ModelRetry(msg)


def create_tool_callable(
    toolset_tool: ToolsetTool,
    tool_name: str,
    ctx: RunContext[Any],
) -> Callable[..., Awaitable[Any]]:
    """Create a callable with proper signature.

    Uses either the available callable or the output schema
    to genererate the signature.
    """
    # Create FunctionSchema from tool definition
    schema_dict = {
        "name": tool_name,
        "description": toolset_tool.tool_def.description or "",
        "parameters": toolset_tool.tool_def.parameters_json_schema,
    }
    out_schema = (toolset_tool.tool_def.metadata or {}).get("output_schema")
    if out_schema:
        out_schema = out_schema.get("properties", {}).get("result")
    function_schema = FunctionSchema.from_dict(schema_dict, output_schema=out_schema)
    sig = function_schema.to_python_signature()

    # Create the wrapper function
    async def tool_wrapper(*args: Any, **kwargs: Any) -> Any:
        # Bind arguments to parameter names
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        result = await toolset_tool.toolset.call_tool(tool_name, bound.arguments, ctx, toolset_tool)

        # If we dont know, stringify so that the model knows what to work with.
        if not isinstance(toolset_tool, FunctionToolsetTool) and not out_schema:
            return str(result) if result is not None else "None"
        return result

    # Set proper function metadata
    tool_wrapper.__name__ = tool_name
    tool_wrapper.__doc__ = toolset_tool.tool_def.description
    tool_wrapper.__annotations__ = {p.name: p.annotation for p in sig.parameters.values()}
    final_return_annotation = None

    if out_schema:  # First: Try to get return type from output_schema -> signature.
        final_return_annotation = sig.return_annotation
    elif isinstance(toolset_tool, FunctionToolsetTool):
        # For FunctionToolsets, get actual return type from original function
        return_type = _get_return_type(toolset_tool)
        final_return_annotation = return_type or str
    else:
        # For rest, always return str since we stringify the results for consistency
        final_return_annotation = str

    if final_return_annotation is not None:
        sig = sig.replace(return_annotation=final_return_annotation)
        tool_wrapper.__annotations__["return"] = final_return_annotation

    tool_wrapper.__signature__ = sig  # type: ignore

    return tool_wrapper


def _get_return_type(toolset_tool: ToolsetTool[Any]) -> type[Any] | None:
    """Extract return type from ToolsetTool if possible, otherwise return None.

    This only works for FunctionToolsets since they have access to the original
    Python function annotations.
    """
    if isinstance(toolset_tool, FunctionToolsetTool) and isinstance(
        toolset_tool.toolset, FunctionToolset
    ):
        # Find the tool in the toolset's tools dict
        for tool_name, tool in toolset_tool.toolset.tools.items():
            if tool_name == toolset_tool.tool_def.name:
                # Get return annotation from the original function
                return tool.function_schema.function.__annotations__.get("return")

    return None
