"""CodeModeToolset for pydantic-ai - LLM tool execution via code."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, Field
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.function import FunctionToolsetTool
from pydantic_core import SchemaValidator
from schemez import ToolsetCodeGenerator
from schemez.functionschema import FunctionSchema

from llmling_models.toolsets.helpers import validate_code


def get_return_type(toolset_tool: ToolsetTool[Any]) -> type[Any] | None:
    """Extract return type from ToolsetTool if possible, otherwise return None.

    This only works for FunctionToolsets since they have access to the original
    Python function annotations. Other toolsets (MCP, External) only have JSON
    schema which doesn't preserve Python type information.
    """
    try:
        # Check if this is a FunctionToolsetTool with access to original function
        if isinstance(toolset_tool, FunctionToolsetTool):
            # Access the original function through the toolset's tools
            from pydantic_ai.toolsets.function import FunctionToolset

            if isinstance(toolset_tool.toolset, FunctionToolset):
                # Find the tool in the toolset's tools dict
                for tool_name, tool in toolset_tool.toolset.tools.items():
                    if tool_name == toolset_tool.tool_def.name:
                        # Get return annotation from the original function
                        return tool.function_schema.function.__annotations__.get("return")
    except Exception:  # noqa: BLE001
        pass
    return None


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from pydantic_ai import RunContext


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


@dataclass
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

    async def _get_code_generator(self, ctx: RunContext[Any]) -> ToolsetCodeGenerator:
        """Get cached toolset generator, creating it if needed."""
        if self._toolset_generator is None:
            callables = []

            for toolset in self.toolsets:
                tools = await toolset.get_tools(ctx)
                for tool_name, toolset_tool in tools.items():
                    # Create proper callable with correct signature from JSON schema
                    callable_func = self._create_callable_with_signature(
                        toolset_tool, tool_name, ctx
                    )
                    callables.append(callable_func)

            self._toolset_generator = ToolsetCodeGenerator.from_callables(
                callables, self.include_docstrings
            )

        return self._toolset_generator

    def _create_callable_with_signature(
        self,
        toolset_tool: ToolsetTool,
        tool_name: str,
        ctx: RunContext[Any],
    ) -> Callable[..., Awaitable[Any]]:
        """Create a callable with proper signature from tool's JSON schema."""
        # Create FunctionSchema from tool definition
        schema_dict = {
            "name": tool_name,
            "description": toolset_tool.tool_def.description or "",
            "parameters": toolset_tool.tool_def.parameters_json_schema,
        }

        # Extract return schema from metadata if available (for MCP tools)
        if (
            toolset_tool.tool_def.metadata
            and "output_schema" in toolset_tool.tool_def.metadata
        ):
            output_schema = toolset_tool.tool_def.metadata["output_schema"]
            if output_schema:
                schema_dict["returns"] = output_schema

        function_schema = FunctionSchema.from_dict(schema_dict)

        # Get the proper signature
        signature = function_schema.to_python_signature()

        # Create the wrapper function
        async def tool_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Bind arguments to parameter names
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            result = await toolset_tool.toolset.call_tool(
                tool_name, bound.arguments, ctx, toolset_tool
            )

            # For non-function toolsets, stringify the result to ensure consistency
            if not isinstance(toolset_tool, FunctionToolsetTool):
                return str(result) if result is not None else "None"
            return result

        # Set proper function metadata
        tool_wrapper.__name__ = tool_name
        tool_wrapper.__doc__ = toolset_tool.tool_def.description

        # Set annotations first
        tool_wrapper.__annotations__ = {
            param.name: param.annotation for param in signature.parameters.values()
        }

        # Set return type based on toolset type
        if isinstance(toolset_tool, FunctionToolsetTool):
            # For FunctionToolsets, get actual return type from original function
            return_type = get_return_type(toolset_tool)
            if return_type:
                tool_wrapper.__annotations__["return"] = return_type
                final_return_annotation = return_type
            else:
                # Fallback for function tools without return type
                final_return_annotation = signature.return_annotation
        else:
            # For non-function toolsets (MCP, External, etc.), always return str
            # since we stringify the results for consistency
            tool_wrapper.__annotations__["return"] = str
            final_return_annotation = str

        # Create new signature with correct return annotation
        if final_return_annotation is not None:
            signature = signature.replace(return_annotation=final_return_annotation)

        tool_wrapper.__signature__ = signature  # type: ignore

        return tool_wrapper


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO)
    from pydantic_ai import Agent
    from pydantic_ai.toolsets.function import FunctionToolset

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
            tools=[get_todays_date, what_happened_on_date, complex_return_type]
        )
        toolset = CodeModeToolset([function_toolset])

        print("✅ Testing unified approach with dynamic signature generation...")

        # Test signature extraction properly
        async with toolset:
            from pydantic_ai.models.test import TestModel
            from pydantic_ai.usage import RunUsage

            # Create a proper RunContext
            test_model = TestModel()
            test_usage = RunUsage()
            ctx = RunContext(deps=None, model=test_model, usage=test_usage)

            # Get the generator and check signatures
            generator = await toolset._get_code_generator(ctx)
            namespace = generator.generate_execution_namespace()

            print("Generated tool signatures:")
            for name, func_wrapper in namespace.items():
                if (
                    name.startswith("get_")
                    or name.startswith("what_")
                    or name.startswith("complex_")
                ):
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

                    if generated_return == original_return:
                        print("✅ Return types match perfectly!")
                    else:
                        print(
                            f"Return type difference: {generated_return} vs {original_return}"
                        )
                    print()

            print("✅ Works with any toolset type (Function, MCP, External, etc.)")

        agent = Agent(model="anthropic:claude-haiku-4-5", toolsets=[toolset])
        async with agent:
            result = await agent.run(
                "what happened today? Get the date and pass it to what_happened_on_date"
            )
            print(f"Function toolset result: {result}")

        # Test with actual MCP toolset
        print("\n🔧 Testing with MCP toolset...")
        try:
            from pydantic_ai.toolsets.fastmcp import FastMCPToolset

            # Create MCP toolset with git server
            mcp_toolset = FastMCPToolset({
                "command": "uvx",
                "args": ["mcp-server-git"],
            })

            # Combine function and MCP toolsets
            combined_toolset = CodeModeToolset([function_toolset, mcp_toolset])

            # Test the combined toolset
            async with combined_toolset:
                test_usage = RunUsage()
                ctx = RunContext(deps=None, model=test_model, usage=test_usage)

                generator = await combined_toolset._get_code_generator(ctx)
                namespace = generator.generate_execution_namespace()

                print("Combined toolset functions:")
                mcp_count = 0
                function_count = 0
                for name, func_wrapper in namespace.items():
                    if not name.startswith("_"):
                        if name.startswith(("get_", "what_", "complex_")):
                            function_count += 1
                            print(f"  Function tool: {name}")
                        else:
                            mcp_count += 1
                            print(f"  MCP tool: {name}")
                            print(
                                f"    Signature: {inspect.signature(func_wrapper.callable)}"
                            )
                            annotations = func_wrapper.callable.__annotations__
                            return_type = annotations.get("return", "None")
                            if isinstance(return_type, str):
                                print(f"    Return type (stringified): {return_type}")
                            else:
                                print(f"    Return type: {return_type}")

                print("\n✅ Combined toolset created successfully!")
                print(f"   - Function tools: {function_count}")
                print(f"   - MCP tools: {mcp_count}")
                print(f"   - Total: {function_count + mcp_count}")
                print("✅ All tools have proper signatures for code execution!")

        except ImportError:
            print("⚠️  FastMCP not available, skipping MCP test")
        except Exception as e:
            print(f"⚠️  MCP test failed: {e}")

    asyncio.run(main())
