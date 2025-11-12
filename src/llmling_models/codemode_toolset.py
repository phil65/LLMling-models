"""CodeModeToolset for pydantic-ai - LLM tool execution via code."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from pydantic_ai.toolsets import AbstractToolset
from pydantic_core import SchemaValidator


if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.toolsets import ToolsetTool
    from schemez import ToolsetCodeGenerator


USAGE_NOTES = """
Usage notes:
- Write your code inside an 'async def main():' function
- All tool functions are async, use 'await'
- Use 'return' statements to return values from main()
- DO NOT call asyncio.run() or try to run the main function yourself
- DO NOT import asyncio or other modules - tools are already available
- Example:
    async def main():
        result = await some_tool_function('parameter')
        return f'Completed with result: {result}'
"""


class CodeExecutionParams(BaseModel):
    """Parameters for Python code execution."""

    python_code: str = Field(description="Python code to execute with tools available")


def _fix_code(python_code: str) -> str:
    """Fix code to be executable by wrapping in main() if needed."""
    if "async def main(" not in python_code:
        # Auto-wrap code in main function, ensuring last expression is returned
        lines = python_code.strip().splitlines()
        if lines:
            # Check if last line is an expression (not a statement)
            last_line = lines[-1].strip()
            if last_line and not any(
                last_line.startswith(kw)
                for kw in [
                    "import ",
                    "from ",
                    "def ",
                    "class ",
                    "if ",
                    "for ",
                    "while ",
                    "try ",
                    "with ",
                    "async def ",
                ]
            ):
                # Last line looks like an expression, add return
                lines[-1] = f"    return {last_line}"
                indented_lines = [f"    {line}" for line in lines[:-1]] + [lines[-1]]
            else:
                indented_lines = [f"    {line}" for line in lines]
            python_code = "async def main():\n" + "\n".join(indented_lines)
        else:
            python_code = "async def main():\n    pass"
    return python_code


class CodeModeToolset(AbstractToolset[Any]):
    """A toolset that wraps other toolsets and provides Python code execution of these."""

    def __init__(
        self,
        toolsets: list[AbstractToolset[Any]],
        *,
        toolset_id: str | None = None,
        include_signatures: bool = True,
        include_docstrings: bool = True,
        usage_notes: str = USAGE_NOTES,
    ):
        """Initialize CodeModeToolset.

        Args:
            toolsets: List of toolsets whose tools should be available in code execution
            toolset_id: Optional unique ID for this toolset
            include_signatures: Include function signatures in tool documentation
            include_docstrings: Include function docstrings in tool documentation
            usage_notes: Usage notes to include in the tool description
        """
        self.toolsets = toolsets
        self._id = toolset_id
        self.include_signatures = include_signatures
        self.include_docstrings = include_docstrings
        self.usage_notes = usage_notes
        self._toolset_generator: ToolsetCodeGenerator | None = None
        self._cached_tools: dict[str, ToolsetTool[Any]] | None = None

    @property
    def id(self) -> str | None:
        """Return the toolset ID."""
        return self._id

    @property
    def label(self) -> str:
        """Return a label for error messages."""
        label = "CodeModeToolset"
        if self.id:
            label += f" {self.id!r}"
        return label

    @property
    def tool_name_conflict_hint(self) -> str:
        """Return hint for resolving name conflicts."""
        return "Rename the toolset ID or use a different CodeModeToolset instance."

    async def __aenter__(self):
        """Enter async context."""
        # Enter context for all wrapped toolsets
        for toolset in self.toolsets:
            await toolset.__aenter__()
        return self

    async def __aexit__(self, *args):
        """Exit async context."""
        # Exit context for all wrapped toolsets
        for toolset in self.toolsets:
            await toolset.__aexit__(*args)

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        """Return the single code execution tool."""
        from pydantic_ai.tools import ToolDefinition
        from pydantic_ai.toolsets.abstract import ToolsetTool

        # Generate tool description with all available tools documented
        toolset_generator = await self._get_code_generator(ctx)
        description = toolset_generator.generate_tool_description()
        description += "\n\n" + self.usage_notes

        # Create tool definition
        tool_def = ToolDefinition(
            name="execute_python",
            description=description,
            parameters_json_schema=CodeExecutionParams.model_json_schema(),
        )

        # Create validator for parameters
        validator = SchemaValidator(CodeExecutionParams.__pydantic_core_schema__)

        # Create toolset tool
        toolset_tool = ToolsetTool(
            toolset=self,
            tool_def=tool_def,
            max_retries=1,
            args_validator=validator,
        )

        return {"execute_python": toolset_tool}

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        """Execute the Python code with all wrapped tools available."""
        if name != "execute_python":
            msg = f"Unknown tool: {name}"
            raise ValueError(msg)

        # Validate and extract parameters
        params = CodeExecutionParams.model_validate(tool_args)

        # Build execution namespace with all tools
        toolset_generator = await self._get_code_generator(ctx)
        namespace = toolset_generator.generate_execution_namespace()

        # Fix the Python code
        python_code = _fix_code(params.python_code)

        try:
            # Execute the code
            exec(python_code, namespace)
            result = await namespace["main"]()

            # Handle edge cases with coroutines and return values
            if inspect.iscoroutine(result):
                result = await result

            # Ensure we return a serializable value
            if result is None:
                return "Code executed successfully"

            if hasattr(result, "__dict__") and not isinstance(
                result, (str, int, float, bool, list, dict)
            ):
                # Handle complex objects that might not serialize well
                return f"Operation completed. Result type: {type(result).__name__}"
        except Exception as e:  # noqa: BLE001
            return f"Error executing code: {e!s}"
        else:
            return result

    def apply(self, visitor) -> None:
        """Apply visitor to all wrapped toolsets."""
        for toolset in self.toolsets:
            toolset.apply(visitor)

    def visit_and_replace(self, visitor):
        """Visit and replace all wrapped toolsets."""
        self.toolsets = [toolset.visit_and_replace(visitor) for toolset in self.toolsets]
        return self

    async def _get_code_generator(self, ctx: RunContext[Any]) -> ToolsetCodeGenerator:
        """Get cached toolset generator, creating it if needed."""
        from schemez import ToolsetCodeGenerator
        from schemez.code_generation.namespace_callable import NamespaceCallable
        from schemez.code_generation.tool_code_generator import ToolCodeGenerator
        from schemez.functionschema import FunctionSchema

        if self._toolset_generator is None:
            # Collect all tools from wrapped toolsets
            all_tools = {}
            for toolset in self.toolsets:
                tools = await toolset.get_tools(ctx)
                # Check for name conflicts
                for tool_name, tool_def in tools.items():
                    if tool_name in all_tools:
                        msg = (
                            f"Tool name conflict: {tool_name!r} is defined "
                            "in multiple toolsets"
                        )
                        raise ValueError(msg)
                    all_tools[tool_name] = tool_def

            # Create tool code generators
            generators = []
            for tool_name, toolset_tool in all_tools.items():
                # Create a closure to capture the current tool and name
                def create_wrapper(ts_tool: ToolsetTool[Any], ts_name: str):
                    async def tool_wrapper(**kwargs):
                        # Call the original toolset with proper arguments
                        return await ts_tool.toolset.call_tool(
                            ts_name, kwargs, ctx, ts_tool
                        )

                    return tool_wrapper

                wrapper_func = create_wrapper(toolset_tool, tool_name)

                # Create FunctionSchema from pydantic-ai tool definition
                # Convert the JSON schema format
                schema_dict = {
                    "name": tool_name,
                    "description": toolset_tool.tool_def.description or "",
                    "parameters": toolset_tool.tool_def.parameters_json_schema,
                }

                function_schema = FunctionSchema.from_dict(schema_dict)

                generator = ToolCodeGenerator(
                    schema=function_schema,
                    callable=NamespaceCallable(wrapper_func),
                    name_override=tool_name,
                )
                generators.append(generator)

            self._toolset_generator = ToolsetCodeGenerator(
                generators, self.include_signatures, self.include_docstrings
            )

        return self._toolset_generator


if __name__ == "__main__":
    import asyncio
    import logging
    import sys
    import webbrowser

    from pydantic_ai import Agent
    from pydantic_ai.toolsets.function import FunctionToolset

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    function_toolset = FunctionToolset(tools=[webbrowser.open])
    toolsets = CodeModeToolset([function_toolset])

    async def main():
        agent = Agent(model="openai:gpt-5-nano", toolsets=[toolsets])
        async with agent:
            result = await agent.run("Open google.com in a new tab.")
            print(f"Result: {result}")

    asyncio.run(main())
