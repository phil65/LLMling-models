"""Tests for provider-specific model configs."""

from __future__ import annotations

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
import pytest

from llmling_models import infer_model
from llmling_models_config.configs import AnthropicModelConfig, GeminiModelConfig


@pytest.fixture(autouse=True)
def _anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")


def test_anthropic_config_with_bare_identifier() -> None:
    """A bare identifier gets the anthropic provider prefix."""
    model = AnthropicModelConfig(identifier="claude-haiku-4-5").get_model()
    assert isinstance(model, AnthropicModel)
    assert model.model_name == "claude-haiku-4-5"


def test_gemini_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """GeminiModelConfig resolves to pydantic-ai's google provider."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")
    model = GeminiModelConfig(identifier="gemini-2.5-flash").get_model()
    assert isinstance(model, GoogleModel)
    assert model.model_name == "gemini-2.5-flash"


@pytest.mark.parametrize("prefix", ["gemini", "google-gla", "google"])
def test_google_api_prefixes(monkeypatch: pytest.MonkeyPatch, prefix: str) -> None:
    """Legacy google prefixes map onto pydantic-ai v2's `google` provider."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test_key")
    model = infer_model(f"{prefix}:gemini-2.5-flash")
    assert isinstance(model, GoogleModel)
    assert model.model_name == "gemini-2.5-flash"
