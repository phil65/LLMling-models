"""Tests for LLM adapter implementations."""

from __future__ import annotations

from pydantic_ai import Agent
import pytest

from llmling_models.llm_adapter import LLMAdapter


TEST_MODEL = "gpt-4o-mini"


def test_adapter_initialization() -> None:
    """Test basic adapter initialization."""
    adapter = LLMAdapter(model_name=TEST_MODEL)
    assert adapter.name() == "llm:gpt-4o-mini"


@pytest.mark.asyncio
async def test_async_basic_usage() -> None:
    """Test basic async usage with an agent."""
    adapter = LLMAdapter(model_name=TEST_MODEL)
    agent = Agent(adapter)

    result = await agent.run("Test prompt")
    assert result.data.count(" ") > 3  # noqa: PLR2004


def test_sync_basic_usage() -> None:
    """Test basic sync usage with an agent."""
    adapter = LLMAdapter(model_name=TEST_MODEL)
    agent = Agent(adapter)

    result = agent.run_sync("Write a short poem")
    assert result.data.count(" ") > 3  # noqa: PLR2004


@pytest.mark.asyncio
async def test_streaming() -> None:
    """Test streaming functionality."""
    adapter = LLMAdapter(model_name=TEST_MODEL)
    agent = Agent(adapter)

    async with agent.run_stream("Test prompt") as response:
        chunks = [chunk async for chunk in response.stream_text()]

    assert "".join(chunks).count(" ") > 3  # noqa: PLR2004


@pytest.mark.asyncio
async def test_usage_tracking() -> None:
    """Test usage information is properly tracked."""
    adapter = LLMAdapter(model_name=TEST_MODEL)
    agent = Agent(adapter)

    result = await agent.run("Test prompt")
    usage = result.usage()
    assert usage
    assert usage.request_tokens > 5  # noqa: PLR2004
    assert usage.response_tokens > 5  # noqa: PLR2004


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
