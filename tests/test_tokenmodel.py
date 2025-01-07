"""Tests for token-optimized model implementation."""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
import pytest
from tokonomics import get_model_limits

from llmling_models.multimodels import TokenOptimizedMultiModel


logging.basicConfig(level=logging.DEBUG)


async def print_model_limits(model_name: str):
    """Debug helper to print model limits."""
    limits = await get_model_limits(model_name)
    print(f"\nModel {model_name} limits:")
    print(f"  input_tokens: {limits.input_tokens if limits else 'None'}")
    print(f"  total_tokens: {limits.total_tokens if limits else 'None'}")


class LargeModel(TestModel):
    """Test model with large context window (128k)."""

    def name(self) -> str:
        return "gpt-4-turbo"


class MediumModel(TestModel):
    """Test model with medium context window (16k)."""

    def name(self) -> str:
        return "gpt-3.5-turbo-16k"


class StandardModel(TestModel):
    """Test model with standard context window (4k)."""

    def name(self) -> str:
        return "gpt-3.5-turbo"


@pytest.mark.asyncio
async def test_token_optimized_efficient():
    """Test token-optimized model selecting smallest sufficient model."""
    # Print limits for debugging
    for model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4-turbo"]:
        await print_model_limits(model)

    # Create models
    large = LargeModel(custom_result_text="Large model response")
    medium = MediumModel(custom_result_text="Medium model response")
    standard = StandardModel(custom_result_text="Standard model response")

    # Configure token-optimized model
    token_model = TokenOptimizedMultiModel[Any](
        models=[large, medium, standard],
        strategy="efficient",
    )

    # Test with agent using short prompt (should fit in standard model)
    agent = Agent[None, str](token_model)
    result = await agent.run("Test " * 100)  # ~400 chars -> ~100 tokens
    assert result.data == "Standard model response"


@pytest.mark.asyncio
async def test_token_optimized_exceeds_limits():
    """Test token-optimized model handling too-long inputs."""
    # Create models
    large = LargeModel(custom_result_text="Large model response")
    medium = MediumModel(custom_result_text="Medium model response")
    standard = StandardModel(custom_result_text="Standard model response")

    # Configure token-optimized model
    token_model = TokenOptimizedMultiModel[Any](
        models=[large, medium, standard],
        strategy="efficient",
    )

    # Test with agent using very long prompt
    agent = Agent[None, str](token_model)
    # Make it really long to ensure it exceeds all limits
    huge_prompt = "Test " * 200_000  # ~800k chars -> ~200k tokens
    print(f"\nTesting with prompt length: {len(huge_prompt)} chars")

    with pytest.raises(RuntimeError, match="No suitable model found"):
        await agent.run(huge_prompt)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
