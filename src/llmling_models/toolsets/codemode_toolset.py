"""CodeModeToolset for pydantic-ai - LLM tool execution via code."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from pydantic_ai.toolsets import AbstractToolset
from pydantic_core import SchemaValidator


if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.toolsets import ToolsetTool
    from schemez import ToolsetCodeGenerator


USAGE_NOTES = """Usage notes:
- Write your code inside an 'async def main():' function
- All tool functions are async, use 'await'
- Use 'return' statements to return values from main()
- DO NOT call asyncio.run() or try to run the main function yourself
- Example:
    async def main():
        result = await open_website(url='https://example.com')
        return result"""


class CodeExecutionParams(BaseModel):
    """Parameters for Python code execution."""

    python_code: str = Field(description="Python code to execute with tools available")


def _validate_code(python_code: str) -> None:
    """Validate code structure and raise ModelRetry for fixable issues."""
    from pydantic_ai import ModelRetry

    code = python_code.strip()
    if not code:
        msg = (
            "Empty code provided. Please write code inside 'async def main():' function."
        )
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

    # Check if last line has return statement
    lines = code.strip().splitlines()
    if lines and not lines[-1].strip().startswith("return"):
        msg = (
            "The main() function should return a value. "
            "Add 'return result' or 'return \"completed\"' at the end of your function."
        )
        raise ModelRetry(msg)


class CodeModeToolset(AbstractToolset[Any]):
    """A toolset that wraps other toolsets and provides Python code execution."""

    def __init__(
        self,
        toolsets: list[AbstractToolset[Any]],
        *,
        toolset_id: str | None = None,
        include_docstrings: bool = True,
        usage_notes: str = USAGE_NOTES,
    ):
        """Initialize CodeModeToolset.

        Args:
            toolsets: List of toolsets whose tools should be available in code execution
            toolset_id: Optional unique ID for this toolset
            include_docstrings: Include function docstrings in tool documentation
            usage_notes: Usage notes to include in the tool description
        """
        self.toolsets = toolsets
        self._id = toolset_id
        self.include_docstrings = include_docstrings
        self.usage_notes = usage_notes
        self._toolset_generator: ToolsetCodeGenerator | None = None

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
        for toolset in self.toolsets:
            await toolset.__aenter__()
        return self

    async def __aexit__(self, *args):
        """Exit async context."""
        for toolset in self.toolsets:
            await toolset.__aexit__(*args)

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        """Return the single code execution tool."""
        from pydantic_ai.tools import ToolDefinition
        from pydantic_ai.toolsets.abstract import ToolsetTool

        toolset_generator = await self._get_code_generator(ctx)
        description = toolset_generator.generate_tool_description()
        description += "\n\n" + self.usage_notes

        tool_def = ToolDefinition(
            name="execute_python",
            description=description,
            parameters_json_schema=CodeExecutionParams.model_json_schema(),
        )

        validator = SchemaValidator(CodeExecutionParams.__pydantic_core_schema__)

        toolset_tool = ToolsetTool(
            toolset=self,
            tool_def=tool_def,
            max_retries=3,
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

        params = CodeExecutionParams.model_validate(tool_args)
        _validate_code(params.python_code)
        toolset_generator = await self._get_code_generator(ctx)
        namespace = toolset_generator.generate_execution_namespace()

        try:
            exec(params.python_code, namespace)
            result = await namespace["main"]()
        except Exception as e:  # noqa: BLE001
            return f"Error executing code: {e!s}"
        else:
            return result if result is not None else "Code executed successfully"

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
            all_tools = {}
            for toolset in self.toolsets:
                tools = await toolset.get_tools(ctx)
                for tool_name, tool_def in tools.items():
                    if tool_name in all_tools:
                        msg = (
                            f"Tool name conflict: {tool_name!r} is defined "
                            "in multiple toolsets"
                        )
                        raise ValueError(msg)
                    all_tools[tool_name] = tool_def

            generators = []
            for tool_name, toolset_tool in all_tools.items():

                def create_wrapper(ts_tool: ToolsetTool[Any], ts_name: str):
                    async def tool_wrapper(**kwargs):
                        return await ts_tool.toolset.call_tool(
                            ts_name, kwargs, ctx, ts_tool
                        )

                    return tool_wrapper

                wrapper_func = create_wrapper(toolset_tool, tool_name)

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
                generators, self.include_docstrings
            )

        return self._toolset_generator


if __name__ == "__main__":
    import asyncio
    import webbrowser

    from pydantic_ai import Agent
    from pydantic_ai.toolsets.function import FunctionToolset

    async def open_website(url: str) -> dict[str, Any]:
        """Open a website in the default browser."""
        webbrowser.open(url)
        return {"success": True, "url": url}

    async def main():
        function_toolset = FunctionToolset(tools=[open_website])
        toolset = CodeModeToolset([function_toolset])

        agent = Agent(model="openai:gpt-4o-mini", toolsets=[toolset])
        async with agent:
            result = await agent.run("Open google.com in a new tab.")
            print(f"Result: {result}")

    asyncio.run(main())
