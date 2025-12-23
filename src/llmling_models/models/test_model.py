"""Test model with fixed tool arguments support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.models.test import TestModel


if TYPE_CHECKING:
    from pydantic_ai.tools import ToolDefinition


@dataclass(init=False)
class FixedArgsTestModel(TestModel):
    """TestModel subclass that supports fixed tool arguments.

    This is useful for testing scenarios where you need deterministic,
    specific tool arguments instead of auto-generated ones.

    Example:
        ```python
        model = FixedArgsTestModel(
            call_tools=['read_file'],
            tool_args={'read_file': {'path': '/test/hello.txt'}}
        )
        # When read_file is called, it will use {'path': '/test/hello.txt'}
        # instead of generating random args from the schema
        ```
    """

    tool_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Mapping of tool_name -> args to use instead of generated args."""

    def __init__(
        self,
        *,
        tool_args: dict[str, dict[str, Any]] | None = None,
        call_tools: list[str] | Literal["all"] = "all",
        custom_output_text: str | None = None,
        custom_output_args: Any | None = None,
        seed: int = 0,
    ) -> None:
        """Initialize FixedArgsTestModel.

        Args:
            tool_args: Mapping of tool_name -> args to use for that tool.
                      Tools not in this mapping will use auto-generated args.
            call_tools: List of tools to call, or 'all' to call all tools.
            custom_output_text: Custom text to return as final output.
            custom_output_args: Custom args to pass to output tool.
            seed: Seed for generating random args (for tools not in tool_args).
        """
        super().__init__(
            call_tools=call_tools,
            custom_output_text=custom_output_text,
            custom_output_args=custom_output_args,
            seed=seed,
        )
        self.tool_args = tool_args or {}

    def gen_tool_args(self, tool_def: ToolDefinition) -> Any:
        """Generate tool arguments, using fixed args if available.

        Args:
            tool_def: The tool definition to generate args for.

        Returns:
            Fixed args if tool_name is in tool_args, otherwise auto-generated.
        """
        if tool_def.name in self.tool_args:
            return self.tool_args[tool_def.name]
        return super().gen_tool_args(tool_def)
