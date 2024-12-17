"""Tests for MultiModel implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError
from pydantic_ai import Agent, Tool
from pydantic_ai.models.test import TestModel
import pytest

from llmling_models import RandomMultiModel
from llmling_models.model_types import _TestModelWrapper


if TYPE_CHECKING:
    from pydantic_ai.models import Model


RANDOM_CFG = """
type: random
models:
  - test
  - openai:gpt-4
"""


@pytest.fixture
def test_models() -> tuple[TestModel, TestModel]:
    """Create two test models with different responses."""
    return (
        TestModel(custom_result_text="Response from Model 1"),
        TestModel(custom_result_text="Response from Model 2"),
    )


@pytest.mark.asyncio
async def test_random_model_basic(test_models: tuple[TestModel, TestModel]) -> None:
    """Test basic RandomMultiModel functionality with pydantic-ai Agent."""
    model1, model2 = test_models
    random_model: RandomMultiModel[Model] = RandomMultiModel(
        models=[_TestModelWrapper(model=model1), _TestModelWrapper(model=model2)],
    )

    # Create a simple agent with our random model
    agent = Agent(random_model)

    # Run multiple times to collect different responses
    responses = set()
    for _ in range(10):
        # Use await with run instead of run_sync
        result = await agent.run("Test prompt")
        responses.add(result.data)

    # Verify we get both responses
    assert len(responses) == 2  # noqa: PLR2004
    assert "Response from Model 1" in responses
    assert "Response from Model 2" in responses


@pytest.mark.asyncio
async def test_random_model_with_tools(test_models: tuple[TestModel, TestModel]) -> None:
    """Test RandomMultiModel with tool usage."""
    model1, model2 = test_models
    random_model: RandomMultiModel[Model] = RandomMultiModel(
        models=[_TestModelWrapper(model=model1), _TestModelWrapper(model=model2)],
    )

    # Create test tool
    async def test_tool(text: str) -> str:
        return f"Processed: {text}"

    # Create agent with tool
    tool = Tool(test_tool)
    agent = Agent(random_model, tools=[tool])

    # Run multiple times
    responses = set()
    for _ in range(10):
        # Use await with run instead of run_sync
        result = await agent.run("Use the test tool")
        responses.add(result.data)

    assert len(responses) == 2  # noqa: PLR2004


def test_random_model_validation() -> None:
    """Test RandomMultiModel validation."""
    # Test empty models list
    with pytest.raises(ValidationError, match=r"List should have at least 1 item"):
        RandomMultiModel(models=[])

    # Test invalid model name
    from pydantic_ai.exceptions import UserError

    with pytest.raises(UserError, match="Unknown model: invalid_model"):
        RandomMultiModel(models=["invalid_model"])


@pytest.mark.asyncio
async def test_yaml_loading() -> None:
    """Test loading RandomMultiModel from YAML configuration."""
    import yaml

    data = yaml.safe_load(RANDOM_CFG)
    model: RandomMultiModel[Model] = RandomMultiModel.model_validate(data)

    assert model.type == "random"
    assert len(model.models) == 2  # noqa: PLR2004

    # Initialize models before checking names
    model.initialize_models()  # type: ignore
    model_names = [m.name() for m in model.available_models]
    assert "test-model" in model_names
    assert "openai:gpt-4" in model_names
