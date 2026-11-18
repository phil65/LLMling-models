"""Tests for provider-specific model configs."""

from __future__ import annotations

from pydantic_ai.models.anthropic import AnthropicModel
import pytest

from llmling_models_config.configs import AnthropicModelConfig


@pytest.fixture(autouse=True)
def _anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")


def test_anthropic_config_with_bare_identifier() -> None:
    """A bare identifier gets the anthropic provider prefix."""
    model = AnthropicModelConfig(identifier="claude-haiku-4-5").get_model()
    assert isinstance(model, AnthropicModel)
    assert model.model_name == "claude-haiku-4-5"
