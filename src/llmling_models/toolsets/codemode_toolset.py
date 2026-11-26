"""CodeModeToolset for pydantic-ai - LLM tool execution via code."""

import asyncio
from dataclasses import dataclass, field
import inspect
import logging
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_core import SchemaValidator
from schemez import ToolsetCodeGenerator

from llmling_models.toolsets.helpers import create_tool_callable, validate_code


if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

TOOL_NAME = "execute_python"
USAGE_NOTES = """
A tool to execute python code.
You can (and should) use the provided stubs as async function calls
if you need to.
Write an async main function that returns the result.
DONT write placeholders. DONT run the function yourself. just write the function.
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


@dataclass(kw_only=True)
class CodeModeToolset[AgentDepsT = None](AbstractToolset[AgentDepsT]):
    """A toolset that wraps other toolsets and provides Python code execution."""

    toolsets: list[AbstractToolset[Any]]
    """List of toolsets whose tools should be available in code execution"""

    toolset_id: str | None = None
    """Optional unique ID for this toolset"""

    include_docstrings: bool = True
    """Include function docstrings in tool documentation"""

    usage_notes: str = USAGE_NOTES
    """Usage notes to include in the tool description"""

    max_retries: int = 3
    """Maximum number of tool retries."""

    _tools_cache: tuple[tuple[str, int], list[dict[str, ToolsetTool[Any]]]] | None = field(
        default=None, init=False, repr=False
    )

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

    async def __aenter__(self) -> Self:
        """Enter async context."""
        for toolset in self.toolsets:
            await toolset.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit async context."""
        for toolset in self.toolsets:
            await toolset.__aexit__(*args)

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        """Return the single code execution tool."""
        toolset_generator = await self._get_code_generator(ctx)
        description = toolset_generator.generate_tool_description()
        tool_def = ToolDefinition(
            name=TOOL_NAME,
            description=description + "\n\n" + self.usage_notes,
            parameters_json_schema=CodeExecutionParams.model_json_schema(),
        )
        tool = ToolsetTool(
            toolset=self,
            tool_def=tool_def,
            max_retries=self.max_retries,
            args_validator=SchemaValidator(CodeExecutionParams.__pydantic_core_schema__),
        )
        return {TOOL_NAME: tool}

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        """Execute the Python code with all wrapped tools available."""
        if name != TOOL_NAME:
            msg = f"Unknown tool: {name}"
            raise ValueError(msg)
        params = CodeExecutionParams.model_validate(tool_args)
        logger.info("Executing Python code: %r", params.python_code)
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

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        """Apply visitor to all wrapped toolsets."""
        for toolset in self.toolsets:
            toolset.apply(visitor)

    def visit_and_replace(
        self,
        visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]],
    ) -> AbstractToolset[AgentDepsT]:
        """Visit and replace all wrapped toolsets."""
        self.toolsets = [toolset.visit_and_replace(visitor) for toolset in self.toolsets]
        return self

    async def _get_wrapped_tools(self, ctx: RunContext[Any]) -> list[dict[str, ToolsetTool[Any]]]:
        """List the wrapped toolsets' tools, once per run step.

        Listing an MCP toolset is a round trip; the listing made for the model request
        is reused when the model's code runs in the same step.
        """
        key = (ctx.run_id, ctx.run_step) if ctx.run_id is not None else None
        if key is not None and self._tools_cache is not None and self._tools_cache[0] == key:
            return self._tools_cache[1]
        tools = list(await asyncio.gather(*(ts.get_tools(ctx) for ts in self.toolsets)))
        if key is not None:
            self._tools_cache = (key, tools)
        return tools

    async def _get_code_generator(self, ctx: RunContext[Any]) -> ToolsetCodeGenerator:
        """Build the toolset code generator, its callables bound to `ctx`."""
        callables = [
            create_tool_callable(toolset_tool, tool_name, ctx)
            for tools in await self._get_wrapped_tools(ctx)
            for tool_name, toolset_tool in tools.items()
        ]
        return ToolsetCodeGenerator.from_callables(callables, self.include_docstrings)


if __name__ == "__main__":
    import asyncio

    from fastmcp.client.transports import StdioTransport
    from pydantic_ai import Agent, RunUsage, Tool
    from pydantic_ai.mcp import MCPToolset
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.toolsets.function import FunctionToolset

    logging.basicConfig(level=logging.INFO)

    # Example tools which require chaining to show CodeMode advantage.
    async def get_todays_date() -> str:
        """Get today's date."""
        return "2024-12-13"

    async def what_happened_on_date(date: str) -> str:
        """Get what happened on a date."""
        return f"On {date}, the great coding session happened!"

    async def complex_return_type() -> dict[str, list[int]]:
        """Test function with complex return type."""
        return {"numbers": [1, 2, 3]}

    async def main() -> None:
        # Test with function toolset (should work with unified approach)
        function_toolset = FunctionToolset(
            tools=[Tool(get_todays_date), Tool(what_happened_on_date), Tool(complex_return_type)]
        )
        toolset = CodeModeToolset(toolsets=[function_toolset])
        print("✅ Testing unified approach with dynamic signature generation...")
        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
        async with toolset:
            generator = await toolset._get_code_generator(ctx)
            namespace = generator.generate_execution_namespace()
            print("Generated tool signatures:")
            for name, func_wrapper in namespace.items():
                if name.startswith(("get_", "what_", "complex_")):
                    print(f"Function: {name}")
                    print(f"Generated: {inspect.signature(func_wrapper.callable)}")
                    print(f"Annotations: {func_wrapper.callable.__annotations__}")
                    # Compare with original function signature
                    original_func = (
                        get_todays_date
                        if name == "get_todays_date"
                        else what_happened_on_date
                        if name == "what_happened_on_date"
                        else complex_return_type
                    )
                    print(f"Original:  {inspect.signature(original_func)}")
                    # Check if return types match for FunctionToolsets
                    generated_return = func_wrapper.callable.__annotations__.get("return")
                    original_return = original_func.__annotations__.get("return")
                    print(f"Return types: {generated_return=} vs {original_return=}")
                    print()

            print("✅ Works with any toolset type (Function, MCP, External, etc.)")

        async with Agent(model="anthropic:claude-haiku-4-5", toolsets=[toolset]) as agent:
            result = await agent.run("what happened today? ")
            print(f"Function toolset result: {result}")

        print("\n🔧 Testing with MCP toolset...")
        mcp_toolset = MCPToolset(StdioTransport(command="uvx", args=["mcp-server-git"]))
        toolset = CodeModeToolset(toolsets=[mcp_toolset])
        async with Agent(model="anthropic:claude-haiku-4-5", toolsets=[toolset]) as agent:
            result = await agent.run(
                "Show diff of latest commit in cwd. Use the provdided stub functions."
            )
            print(f"Result: {result}")

    asyncio.run(main())
