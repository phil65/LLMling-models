"""Tests for FixedArgsTestModel and TestModelConfig with tool_args."""

from __future__ import annotations

from pydantic_ai import Agent
import pytest

from llmling_models.configs import TestModelConfig
from llmling_models.models import FixedArgsTestModel


class TestFixedArgsTestModel:
    """Tests for FixedArgsTestModel."""

    def test_uses_fixed_args_for_specified_tool(self) -> None:
        """Test that fixed args are used for tools in tool_args mapping."""
        model = FixedArgsTestModel(
            call_tools=["my_tool"],
            tool_args={"my_tool": {"path": "/fixed/path", "count": 42}},
        )

        # Create a mock tool definition
        from pydantic_ai.tools import ToolDefinition

        tool_def = ToolDefinition(
            name="my_tool",
            description="A test tool",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "count": {"type": "integer"},
                },
            },
        )

        args = model.gen_tool_args(tool_def)
        assert args == {"path": "/fixed/path", "count": 42}

    def test_uses_generated_args_for_unspecified_tool(self) -> None:
        """Test that tools not in tool_args use auto-generated args."""
        model = FixedArgsTestModel(
            call_tools=["other_tool"],
            tool_args={"my_tool": {"path": "/fixed/path"}},
            seed=123,
        )

        from pydantic_ai.tools import ToolDefinition

        tool_def = ToolDefinition(
            name="other_tool",
            description="Another tool",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            },
        )

        # Should not raise, and should return generated args
        args = model.gen_tool_args(tool_def)
        assert isinstance(args, dict)
        assert "name" in args

    async def test_agent_uses_fixed_args(self) -> None:
        """Test that an agent with FixedArgsTestModel uses the fixed args."""
        captured_args: dict = {}

        agent: Agent[None, str] = Agent(
            model=FixedArgsTestModel(
                call_tools=["get_weather"],
                tool_args={"get_weather": {"city": "London"}},
            )
        )

        @agent.tool_plain
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            captured_args["city"] = city
            return f"Weather in {city}: Sunny"

        result = await agent.run("What's the weather?")

        assert captured_args["city"] == "London"
        assert "London" in result.output or "Sunny" in result.output


class TestTestModelConfig:
    """Tests for TestModelConfig with tool_args."""

    def test_returns_test_model_without_tool_args(self) -> None:
        """Test that config without tool_args returns standard TestModel."""
        from pydantic_ai.models.test import TestModel

        config = TestModelConfig(
            call_tools=["tool1"],
            custom_output_text="test output",
        )
        model = config.get_model()

        assert isinstance(model, TestModel)
        assert not isinstance(model, FixedArgsTestModel)

    def test_returns_fixed_args_model_with_tool_args(self) -> None:
        """Test that config with tool_args returns FixedArgsTestModel."""
        config = TestModelConfig(
            call_tools=["read_file"],
            tool_args={"read_file": {"path": "/test/file.txt"}},
        )
        model = config.get_model()

        assert isinstance(model, FixedArgsTestModel)
        assert model.tool_args == {"read_file": {"path": "/test/file.txt"}}

    def test_seed_is_passed_through(self) -> None:
        """Test that seed parameter is passed to the model."""
        config = TestModelConfig(
            call_tools="all",
            seed=42,
        )
        model = config.get_model()
        assert model.seed == 42  # noqa: PLR2004

    def test_config_with_all_options(self) -> None:
        """Test config with all options set."""
        config = TestModelConfig(
            call_tools=["tool1", "tool2"],
            custom_output_text="Final output",
            tool_args={
                "tool1": {"arg1": "value1"},
                "tool2": {"arg2": 123},
            },
            seed=99,
        )
        model = config.get_model()

        assert isinstance(model, FixedArgsTestModel)
        assert model.call_tools == ["tool1", "tool2"]
        assert model.custom_output_text == "Final output"
        assert model.tool_args == {
            "tool1": {"arg1": "value1"},
            "tool2": {"arg2": 123},
        }
        assert model.seed == 99  # noqa: PLR2004


if __name__ == "__main__":
    pytest.main(["-v", __file__])
