"""Tests for cost-optimized model implementation."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, patch

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
import pytest
from tokonomics import TokenLimits

from llmling_models.multimodels import CostOptimizedMultiModel


# Configure logging
logger = logging.getLogger(__name__)


# Mock model classes
class ExpensiveModel(TestModel):
    """Test model with high cost."""

    def name(self) -> str:
        """Get model name."""
        return "expensive-model"


class CheapModel(TestModel):
    """Test model with low cost."""

    def name(self) -> str:
        """Get model name."""
        return "cheap-model"


# Mock data for tests
MOCK_COSTS = {
    "expensive-model": {
        "input_cost_per_token": "0.0003",  # $0.0003 per token
    },
    "cheap-model": {
        "input_cost_per_token": "0.0001",  # $0.0001 per token
    },
}

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


@pytest.fixture
def mock_get_limits() -> AsyncMock:
    """Create a mock for get_model_limits."""

    async def _get_limits(model_name: str) -> TokenLimits:
        limits = MOCK_LIMITS[model_name]
        logger.debug("Mock returning limits for %s: %s", model_name, limits)
        return limits

    return AsyncMock(side_effect=_get_limits)


@pytest.fixture
def mock_get_costs() -> AsyncMock:
    """Create a mock for get_model_costs."""

    async def _get_costs(model_name: str) -> dict[str, str]:
        costs = MOCK_COSTS[model_name]
        logger.debug("Mock returning costs for %s: %s", model_name, costs)
        return costs

    return AsyncMock(side_effect=_get_costs)


@pytest.mark.asyncio
async def test_cost_optimized_token_limit(
    mock_get_limits: AsyncMock,
    mock_get_costs: AsyncMock,
):
    """Test cost-optimized model respecting token limits."""

    # Override the mock limits for this test
    async def _get_limits_low(model_name: str) -> TokenLimits:
        return TokenLimits(
            total_tokens=10,
            input_tokens=8,
            output_tokens=2,
        )

    mock_get_limits.side_effect = _get_limits_low

    with (
        patch("llmling_models.multimodels.cost.get_model_limits", mock_get_limits),
        patch("llmling_models.multimodels.cost.get_model_costs", mock_get_costs),
    ):
        # Create models
        expensive = ExpensiveModel(custom_result_text="Expensive response")
        cheap = CheapModel(custom_result_text="Cheap response")

        # Configure cost-optimized model
        cost_model = CostOptimizedMultiModel[Any](
            models=[expensive, cheap],
            max_input_cost=1.0,  # High budget - should fail on token limit
            strategy="best_within_budget",
        )

        # Test with agent using long prompt
        agent = Agent[None, str](cost_model)
        with pytest.raises(RuntimeError, match="No suitable model found"):
            await agent.run("Very " * 100 + "long prompt")

        # Verify our mocks were called
        assert mock_get_limits.called


@pytest.mark.asyncio
async def test_cost_optimized_budget_limit(
    mock_get_limits: AsyncMock,
    mock_get_costs: AsyncMock,
):
    """Test cost-optimized model respecting budget limit."""

    # Override the mock costs for this test
    async def _get_costs_high(model_name: str) -> dict[str, str]:
        return {
            "input_cost_per_token": "1.0",  # Very expensive input cost
        }

    mock_get_costs.side_effect = _get_costs_high

    with (
        patch("llmling_models.multimodels.cost.get_model_limits", mock_get_limits),
        patch("llmling_models.multimodels.cost.get_model_costs", mock_get_costs),
    ):
        # Create models
        expensive = ExpensiveModel(custom_result_text="Expensive response")
        cheap = CheapModel(custom_result_text="Cheap response")

        # Configure cost-optimized model with very low budget
        cost_model = CostOptimizedMultiModel[Any](
            models=[expensive, cheap],
            max_input_cost=0.00001,  # Very low budget
            strategy="cheapest_possible",
        )

        # Test with agent
        agent = Agent[None, str](cost_model)
        with pytest.raises(RuntimeError, match="No suitable model found"):
            await agent.run("Test prompt")

        # Verify our mocks were called
        assert mock_get_costs.called


@pytest.mark.asyncio
async def test_cost_optimized_cheapest(
    mock_get_limits: AsyncMock,
    mock_get_costs: AsyncMock,
):
    """Test cost-optimized model selecting cheapest option."""
    with (
        patch("llmling_models.multimodels.cost.get_model_limits", mock_get_limits),
        patch("llmling_models.multimodels.cost.get_model_costs", mock_get_costs),
    ):
        # Create models
        expensive = ExpensiveModel(custom_result_text="Expensive response")
        cheap = CheapModel(custom_result_text="Cheap response")

        # Configure cost-optimized model with budget > cheap model cost
        cost_model = CostOptimizedMultiModel[Any](
            models=[expensive, cheap],
            max_input_cost=0.5,  # 50 cents max (should allow cheap model)
            strategy="cheapest_possible",
        )

        # Test with agent
        agent = Agent[None, str](cost_model)
        result = await agent.run("Test prompt")

        # Should select cheapest model
        assert result.data == "Cheap response"

        # Verify our mocks were called
        assert mock_get_limits.called
        assert mock_get_costs.called


@pytest.mark.asyncio
async def test_cost_optimized_best_within_budget(
    mock_get_limits: AsyncMock,
    mock_get_costs: AsyncMock,
):
    """Test cost-optimized model selecting best within budget."""
    with (
        patch("llmling_models.multimodels.cost.get_model_limits", mock_get_limits),
        patch("llmling_models.multimodels.cost.get_model_costs", mock_get_costs),
    ):
        # Create models
        expensive = ExpensiveModel(custom_result_text="Expensive response")
        cheap = CheapModel(custom_result_text="Cheap response")

        # Configure cost-optimized model with high budget
        cost_model = CostOptimizedMultiModel[Any](
            models=[expensive, cheap],
            max_input_cost=1.0,  # $1 max (should allow both models)
            strategy="best_within_budget",
        )

        # Test with agent
        agent = Agent[None, str](cost_model)
        result = await agent.run("Test prompt")

        # Should select most expensive model within budget
        assert result.data == "Expensive response"

        # Verify our mocks were called
        assert mock_get_limits.called
        assert mock_get_costs.called
