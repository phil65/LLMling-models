"""Tests for infer_model function."""

import os
from typing import TYPE_CHECKING

import pytest

from llmling_models.models.helpers import infer_model
from llmling_models.providers import infer_provider
from llmling_models.providers.copilot_provider import CopilotProvider
from llmling_models.providers.lm_studio_provider import LMStudioProvider


if TYPE_CHECKING:
    from pydantic_ai.providers import Provider


# Set environment variables for testing
# os.environ["OPENAI_API_KEY"] = "test_key"
os.environ["GITHUB_COPILOT_API_KEY"] = "test_key"


def test_openai_model():
    """Test creating an OpenAI model."""
    model = infer_model("openai:gpt-3.5-turbo")
    assert model.model_name == "gpt-3.5-turbo"
    # assert model.system == "openai"


def test_openrouter_format():
    """Test openrouter format conversion."""
    model = infer_model("openrouter:anthropic/claude-3-sonnet-20240229")
    # Check that colon was converted to slash in model name
    assert "claude-3-sonnet-20240229" in model.model_name
    assert "/" in model.model_name
    # assert model.system == "openrouter"


def test_import_model():
    """Test importing a model."""
    model = infer_model("import:pydantic_ai.models.test:TestModel")
    assert model.__class__.__name__ == "TestModel"


def test_infer_provider_returns_correct_provider_classes():
    """Test that infer_provider returns the correct provider class instances."""
    provider_map: dict[str, type[Provider]] = {
        "copilot": CopilotProvider,
        "lm-studio": LMStudioProvider,
    }

    for provider_name, expected_class in provider_map.items():
        provider = infer_provider(provider_name)
        assert isinstance(provider, expected_class)
        assert provider.name == provider_name


if __name__ == "__main__":
    pytest.main(["-v", __file__])
