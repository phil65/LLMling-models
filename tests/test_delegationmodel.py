"""Tests for delegation model implementation."""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
import pytest

from llmling_models.multimodels import DelegationMultiModel


class ComplexTestModel(TestModel):
    """Test model for complex tasks."""

    def name(self) -> str:
        return "complex-test-model"


class SimpleTestModel(TestModel):
    """Test model for simple tasks."""

    def name(self) -> str:
        return "simple-test-model"


@pytest.mark.asyncio
async def test_delegation_with_list() -> None:
    """Test delegation model with simple list of models."""
    test_models = [
        TestModel(custom_result_text="Model 1 response"),
        TestModel(custom_result_text="Model 2 response"),
    ]

    delegation_model = DelegationMultiModel[Any](
        selector_model=TestModel(custom_result_text=test_models[0].name()),
        models=test_models,  # type: ignore
        selection_prompt="Pick first model for complex, second for simple tasks.",
    )

    agent = Agent[None, str](delegation_model)
    result = await agent.run("A complex task")
    assert result.data == "Model 1 response"


@pytest.mark.asyncio
async def test_delegation_with_descriptions() -> None:
    """Test delegation model with model descriptions."""
    complex_model = ComplexTestModel(custom_result_text="Complex response")
    simple_model = SimpleTestModel(custom_result_text="Simple response")

    # Use model names as keys
    test_models = {
        "complex-test-model": "complex tasks",
        "simple-test-model": "simple queries",
    }

    # Selector that will pick the complex model
    selector = TestModel(custom_result_text="complex-test-model")

    delegation_model = DelegationMultiModel[Any](
        selector_model=selector,
        models=[complex_model, simple_model],
        model_descriptions=test_models,  # type: ignore
        selection_prompt="Select appropriate model.",
    )

    agent = Agent[None, str](delegation_model)
    result = await agent.run("A complex calculation")
    assert result.data == "Complex response"


@pytest.mark.asyncio
async def test_delegation_invalid_selection() -> None:
    """Test delegation model when selector returns invalid model name."""
    test_models = [
        TestModel(custom_result_text="Model 1 response"),
        TestModel(custom_result_text="Model 2 response"),
    ]

    # Create selector that returns invalid model name
    selector = TestModel(custom_result_text="invalid_model")

    delegation_model = DelegationMultiModel[Any](
        selector_model=selector,
        models=test_models,  # type: ignore
        selection_prompt="Pick a model.",
    )

    agent = Agent(delegation_model)
    with pytest.raises(ValueError, match="Selector returned unknown model"):
        await agent.run("Any task")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
