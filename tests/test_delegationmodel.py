"""Tests for delegation model implementation."""

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
import pytest

from llmling_models.models import DelegationMultiModel


class ComplexTestModel(TestModel):
    """Test model for complex tasks."""

    @property
    def model_name(self) -> str:
        return "complex-test-model"


class SimpleTestModel(TestModel):
    """Test model for simple tasks."""

    @property
    def model_name(self) -> str:
        return "simple-test-model"


async def test_delegation_with_list():
    """Test delegation model with simple list of models."""
    test_models = [
        TestModel(custom_output_text="Model 1 response"),
        TestModel(custom_output_text="Model 2 response"),
    ]

    delegation_model = DelegationMultiModel(
        selector_model=TestModel(custom_output_text=test_models[0].model_name),
        models=test_models,  # type: ignore
        selection_prompt="Pick first model for complex, second for simple tasks.",
    )

    agent = Agent[None, str](delegation_model)
    result = await agent.run("A complex task")
    assert result.output == "Model 1 response"


async def test_delegation_with_descriptions():
    """Test delegation model with model descriptions."""
    complex_model = ComplexTestModel(custom_output_text="Complex response")
    simple_model = SimpleTestModel(custom_output_text="Simple response")

    # Use model names as keys
    test_models = {
        "complex-test-model": "complex tasks",
        "simple-test-model": "simple queries",
    }

    # Selector that will pick the complex model
    selector = TestModel(custom_output_text="complex-test-model")

    delegation_model = DelegationMultiModel(
        selector_model=selector,
        models=[complex_model, simple_model],
        model_descriptions=test_models,  # type: ignore
        selection_prompt="Select appropriate model.",
    )

    agent = Agent[None, str](delegation_model)
    result = await agent.run("A complex calculation")
    assert result.output == "Complex response"


async def test_delegation_invalid_selection():
    """Test delegation model when selector returns invalid model name."""
    test_models = [
        TestModel(custom_output_text="Model 1 response"),
        TestModel(custom_output_text="Model 2 response"),
    ]

    # Create selector that returns invalid model name
    selector = TestModel(custom_output_text="invalid_model")

    delegation_model = DelegationMultiModel(
        selector_model=selector,
        models=test_models,  # type: ignore
        selection_prompt="Pick a model.",
    )

    agent = Agent(delegation_model)
    with pytest.raises(ValueError, match="Selector returned unknown model"):
        await agent.run("Any task")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


async def test_delegation_matches_configured_ids_and_hides_agent_tools():
    """The selector answers with the configured id, sees descriptions, not the tools."""
    from pydantic_ai import ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.usage import RequestUsage

    seen: dict[str, object] = {}

    def select(messages, info: AgentInfo) -> ModelResponse:
        seen["tools"] = [t.name for t in info.function_tools]
        seen["prompt"] = messages[-1].parts[0].content
        return ModelResponse(
            parts=[TextPart("'openai:gpt-4o-mini'\n")],
            usage=RequestUsage(input_tokens=7, output_tokens=1),
        )

    delegation_model = DelegationMultiModel(
        selector_model=FunctionModel(select),
        models={  # type: ignore[arg-type]
            "openai:gpt-4o-mini": "coding",
            "test": "chat",
        },
        selection_prompt="Pick a model.",
    )
    coding_model = delegation_model.available_models[0]

    def delete_everything() -> str:
        return "gone"

    agent = Agent[None, str](delegation_model, tools=[delete_everything])
    selected = await delegation_model._select_model("write code", None)
    assert selected[0] is coding_model
    assert selected[1].input_tokens == 7  # noqa: PLR2004
    assert "coding" in str(seen["prompt"])
    assert "openai:gpt-4o-mini" in str(seen["prompt"])

    # A full run with an agent tool (TestModel calls it): the selector never sees it,
    # and the step after the tool call still finds the user prompt.
    def select_test(messages, info: AgentInfo) -> ModelResponse:
        seen["tools"] = [t.name for t in info.function_tools]
        return ModelResponse(parts=[TextPart("test")])

    delegation_model.selector_model = FunctionModel(select_test)
    result = await agent.run("chat with me")
    assert seen["tools"] == []
    assert result.output
