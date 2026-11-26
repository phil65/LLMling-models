"""Tests for provider-specific model configs."""

from __future__ import annotations

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
import pytest

from llmling_models import infer_model
from llmling_models_config.configs import (
    AnthropicModelConfig,
    GeminiModelConfig,
    OpenAIModelConfig,
    OpenRouterModelConfig,
)


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


def test_provider_configs_accept_unlisted_identifiers() -> None:
    """Identifiers missing from tokonomics' known-model lists are still accepted."""
    assert OpenAIModelConfig(identifier="gpt-4o").identifier == "gpt-4o"
    assert OpenRouterModelConfig(identifier="openai/gpt-4o").identifier == "openai/gpt-4o"
    assert AnthropicModelConfig(identifier="claude-future-9").identifier == "claude-future-9"
    assert GeminiModelConfig(identifier="gemini-future-9").identifier == "gemini-future-9"


def test_nested_config_settings_reach_the_inner_models() -> None:
    from llmling_models_config.configs import (
        AugmentedModelConfig,
        DelegationModelConfig,
        FallbackModelConfig,
        OpenAIModelConfig,
        PrePostPromptConfig,
    )

    inner = OpenAIModelConfig(identifier="gpt-4o", temperature=0.1)
    fallback = FallbackModelConfig(models=[inner, "test"]).get_model()
    assert fallback.models[0].settings == {"temperature": 0.1}

    delegation = DelegationModelConfig(
        selector_model="test",
        models=[inner, "openai:gpt-4o-mini"],
        selection_prompt="pick",
        model_descriptions={"openai:gpt-4o-mini": "small"},
    ).get_model()
    assert delegation.available_models[0].settings == {"temperature": 0.1}
    assert delegation.model_descriptions == {"openai:gpt-4o-mini": "small"}
    assert "openai:gpt-4o-mini" in delegation._models_by_id()

    augmented = AugmentedModelConfig(
        main_model=inner,
        pre_prompt=PrePostPromptConfig(text="x {input}", model=inner),
    ).get_model()
    assert augmented._main_model.settings == {"temperature": 0.1}
    assert augmented.pre_prompt is not None
    assert augmented.pre_prompt.model_instance.settings == {"temperature": 0.1}


def test_config_package_does_not_import_pydantic_ai() -> None:
    import subprocess
    import sys

    code = "import sys, llmling_models_config; print('pydantic_ai' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
