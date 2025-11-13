"""CodeModeToolset for pydantic-ai - LLM tool execution via code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool
from pydantic_core import SchemaValidator
from schemez import ToolsetCodeGenerator
from schemez.code_generation.namespace_callable import NamespaceCallable
from schemez.code_generation.tool_code_generator import ToolCodeGenerator
from schemez.functionschema import FunctionSchema

from llmling_models.toolsets.helpers import validate_code


if TYPE_CHECKING:
    from pydantic_ai import RunContext


USAGE_NOTES = """
A tool to execute python code.
You can (and should) use the provided stubs as async function calls
if you need to.
Write an async main function that returns the result.
DONT write placeholders.

Example tool call:
<code>
async def main():
    result_1 = await provided_function()
    result_2 = await provided_function_2("some_arg")
    return result_1 + result_2
</code>
"""


class CodeExecutionParams(BaseModel):
    """Parameters for Python code execution."""

    python_code: str = Field(description="Python code to execute with tools available")


@dataclass
class CodeModeToolset(AbstractToolset[Any]):
    """A toolset that wraps other toolsets and provides Python code execution."""

    toolsets: list[AbstractToolset[Any]]
    """List of toolsets whose tools should be available in code execution"""

    toolset_id: str | None = None
    """Optional unique ID for this toolset"""

    include_docstrings: bool = True
    """Include function docstrings in tool documentation"""

    usage_notes: str = USAGE_NOTES
    """Usage notes to include in the tool description"""

    _toolset_generator: ToolsetCodeGenerator | None = None
    """Toolset code generator"""

    @property
    def id(self) -> str | None:
        """Return the toolset ID."""
        return self.toolset_id

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
        validate_code(params.python_code)
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
        if self._toolset_generator is None:
            all_tools: dict[str, ToolsetTool] = {}
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
                function_schema = FunctionSchema(
                    name=tool_name,
                    parameters=toolset_tool.tool_def.parameters_json_schema,
                    description=toolset_tool.tool_def.description,
                )
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
    import logging

    logging.basicConfig(level=logging.DEBUG)
    from pydantic_ai import Agent
    from pydantic_ai.toolsets.function import FunctionToolset

    async def get_todays_date() -> str:
        """Get today's date."""
        return "2024-12-13"

    async def what_happened_on_date(date: str) -> str:
        """Get what happened on a date."""
        return f"On {date}, the great coding session happened!"

    async def main():
        function_toolset = FunctionToolset(tools=[get_todays_date, what_happened_on_date])
        toolset = CodeModeToolset([function_toolset])

        agent = Agent(model="openai:gpt-4o-mini", toolsets=[toolset])
        async with agent:
            result = await agent.run("What happened today?")
            print(f"Result: {result}")

    asyncio.run(main())
