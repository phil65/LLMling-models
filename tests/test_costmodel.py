"""Tests for cost-optimized model implementation."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
import pytest
from tokonomics import TokenLimits

from llmling_models.multimodels import CostOptimizedMultiModel


# Mock costs for testing
MOCK_COSTS = {
    "expensive-model": {
        "input_cost_per_token": "0.0004",
        "output_cost_per_token": "0.0008",
    },
    "cheap-model": {
        "input_cost_per_token": "0.0001",
        "output_cost_per_token": "0.0002",
    },
}

# Mock token limits
MOCK_LIMITS = {
    "expensive-model": TokenLimits(
        total_tokens=8000,
        input_tokens=6000,
        output_tokens=2000,
    ),
    "cheap-model": TokenLimits(
        total_tokens=4000,
        input_tokens=3000,
        output_tokens=1000,
    ),
}


class ExpensiveModel(TestModel):
    """Test model with high cost."""

    def name(self) -> str:
        return "expensive-model"


class CheapModel(TestModel):
    """Test model with low cost."""

    def name(self) -> str:
        return "cheap-model"


@pytest.mark.asyncio
@patch("llmling_models.utils.get_model_costs")
@patch("llmling_models.utils.get_model_limits")
async def test_cost_optimized_cheapest(
    mock_limits: Any,
    mock_costs: Any,
) -> None:
    """Test cost-optimized model selecting cheapest option."""

    # Setup mocks
    async def mock_get_costs(model_name: str) -> dict[str, str]:
        return MOCK_COSTS[model_name]

    async def mock_get_limits(model_name: str) -> TokenLimits:
        return TokenLimits(
            total_tokens=10,
            input_tokens=8,
            output_tokens=2,
        )

    mock_costs.side_effect = mock_get_costs
    mock_limits.side_effect = mock_get_limits

    # Create models
    expensive = ExpensiveModel(custom_result_text="Expensive response")
    cheap = CheapModel(custom_result_text="Cheap response")

    # Configure cost-optimized model
    cost_model = CostOptimizedMultiModel[Any](
        models=[expensive, cheap],
        max_cost=0.1,
        strategy="cheapest_possible",
    )

    # Test with agent
    agent = Agent[None, str](cost_model)
    result = await agent.run("Test prompt")

    # Should select cheapest model
    assert result.data == "Cheap response"


@pytest.mark.asyncio
@patch("llmling_models.utils.get_model_costs")
@patch("llmling_models.utils.get_model_limits")
async def test_cost_optimized_best_within_budget(
    mock_limits: Any,
    mock_costs: Any,
) -> None:
    """Test cost-optimized model selecting best within budget."""

    # Setup mocks
    async def mock_get_costs(model_name: str) -> dict[str, str]:
        return MOCK_COSTS[model_name]

    async def mock_get_limits(model_name: str) -> TokenLimits:
        return TokenLimits(
            total_tokens=10,
            input_tokens=8,
            output_tokens=2,
        )

    mock_costs.side_effect = mock_get_costs
    mock_limits.side_effect = mock_get_limits

    # Create models
    expensive = ExpensiveModel(custom_result_text="Expensive response")
    cheap = CheapModel(custom_result_text="Cheap response")

    # Configure cost-optimized model with high budget
    cost_model = CostOptimizedMultiModel[Any](
        models=[expensive, cheap],
        max_cost=1.0,
        strategy="best_within_budget",
    )

    # Test with agent
    agent = Agent[None, str](cost_model)
    result = await agent.run("Test prompt")

    # Should select most expensive model within budget
    assert result.data == "Expensive response"


@pytest.mark.asyncio
@patch("llmling_models.utils.get_model_costs")
@patch("llmling_models.utils.get_model_limits")
async def test_cost_optimized_token_limit(
    mock_limits: Any,
    mock_costs: Any,
) -> None:
    """Test cost-optimized model respecting token limits."""

    # Setup mocks with very low token limit
    async def mock_get_costs(model_name: str) -> dict[str, str]:
        return MOCK_COSTS[model_name]

    async def mock_get_limits(model_name: str) -> TokenLimits:
        return TokenLimits(
            total_tokens=10,
            input_tokens=8,
            output_tokens=2,
        )

    mock_costs.side_effect = mock_get_costs
    mock_limits.side_effect = mock_get_limits

    # Create models
    expensive = ExpensiveModel(custom_result_text="Expensive response")
    cheap = CheapModel(custom_result_text="Cheap response")

    # Configure cost-optimized model
    cost_model = CostOptimizedMultiModel[Any](
        models=[expensive, cheap],
        max_cost=1.0,
        strategy="best_within_budget",
    )

    # Test with agent using long prompt
    agent = Agent[None, str](cost_model)
    with pytest.raises(RuntimeError, match="No suitable model found"):
        await agent.run("Very " * 100 + "long prompt")


@pytest.mark.asyncio
@patch("llmling_models.utils.get_model_costs")
@patch("llmling_models.utils.get_model_limits")
async def test_cost_optimized_budget_limit(
    mock_limits: Any,
    mock_costs: Any,
) -> None:
    """Test cost-optimized model respecting budget limit."""

    # Setup mocks
    async def mock_get_costs(model_name: str) -> dict[str, str]:
        return MOCK_COSTS[model_name]

    async def mock_get_limits(model_name: str) -> TokenLimits:
        return TokenLimits(
            total_tokens=10,
            input_tokens=8,
            output_tokens=2,
        )

    mock_costs.side_effect = mock_get_costs
    mock_limits.side_effect = mock_get_limits

    # Create models
    expensive = ExpensiveModel(custom_result_text="Expensive response")
    cheap = CheapModel(custom_result_text="Cheap response")

    # Configure cost-optimized model with very low budget
    cost_model = CostOptimizedMultiModel[Any](
        models=[expensive, cheap],
        max_cost=0.00001,  # Very low budget
        strategy="cheapest_possible",
    )

    # Test with agent
    agent = Agent[None, str](cost_model)
    with pytest.raises(RuntimeError, match="No suitable model found"):
        await agent.run("Test prompt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
