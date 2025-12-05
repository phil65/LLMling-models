"""Utility functions for model handling."""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ImportString
from pydantic_ai.models import infer_model as infer_model_
from pydantic_ai.models.openai import OpenAIChatModel


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pydantic_ai.models import Model


def get_model(
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Model:
    """Get model instance with appropriate implementation based on environment."""
    # Check if this is a provider model (contains colon)
    provider_name = None
    model_name = model

    if ":" in model:
        provider_name, model_name = model.split(":", 1)

        # Special handling for openrouter (TODO: check this)
        if provider_name == "openrouter":
            model_name = model_name.replace(":", "/")

    # For pyodide environments, use SimpleOpenAIModel
    if not importlib.util.find_spec("openai"):
        from llmling_models.models.pyodide_model import SimpleOpenAIModel

        return SimpleOpenAIModel(model=model_name, api_key=api_key, base_url=base_url)

    # For regular environments and recognized providers, use the provider interface
    from pydantic_ai.models.openai import OpenAIResponsesModel

    if provider_name:
        try:
            from llmling_models.providers import infer_provider

            provider = infer_provider(provider_name)
            return OpenAIChatModel(model_name=model_name, provider=provider)
        except ValueError:
            # If provider not recognized, continue with direct approach
            pass
    from pydantic_ai.providers.openai import OpenAIProvider

    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIResponsesModel(model_name=model_name, provider=provider)


def infer_model(model: str | Model) -> Model:
    """Extended infer_model from pydantic-ai with fallback support.

    For fallback models, use comma-separated model names.
    Example: "openai:gpt-4,openai:gpt-3.5-turbo"
    """
    from pydantic_ai.models.fallback import FallbackModel

    # If model is already a Model instance or something else not string
    if not isinstance(model, str):
        return model

    # Check for comma-separated model list (fallback case)
    if "," in model:
        model_names = [name.strip() for name in model.split(",")]
        if len(model_names) <= 1:
            # Shouldn't happen with the comma check, but just to be safe
            return _infer_single_model(model)

        # Create fallback model chain
        default_model = _infer_single_model(model_names[0])
        fallback_models = [_infer_single_model(m) for m in model_names[1:]]
        return FallbackModel(default_model, *fallback_models)

    # Regular single model case
    return _infer_single_model(model)


def _infer_single_model(  # noqa: PLR0911
    model: str | Model,
) -> Model:
    """Extended infer_model from pydantic-ai."""
    if not isinstance(model, str):
        return model

    if model.startswith("openrouter:"):
        key = os.getenv("OPENROUTER_API_KEY")
        return get_model(model, base_url="https://openrouter.ai/api/v1", api_key=key)
    if model.startswith("grok:"):
        key = os.getenv("X_AI_API_KEY") or os.getenv("GROK_API_KEY")
        return get_model(model, base_url="https://api.x.ai/v1", api_key=key)
    if model.startswith("deepseek:"):
        key = os.getenv("DEEPSEEK_API_KEY")
        return get_model(model, base_url="https://api.deepseek.com", api_key=key)
    if model.startswith("perplexity:"):
        key = os.getenv("PERPLEXITY_API_KEY")
        return get_model(model, base_url="https://api.perplexity.ai", api_key=key)
    if model.startswith("lm-studio:"):
        return get_model(model, base_url="http://localhost:1234/v1/", api_key="lm-studio")
    if model.startswith("openai:"):
        return get_model(model)
    if model.startswith("zen:"):
        from llmling_models.providers.zen_provider_factory import _create_zen_model

        return _create_zen_model(model_name=model.removeprefix("zen:"))

    if model.startswith("copilot:"):
        key = os.getenv("GITHUB_COPILOT_API_KEY")
        return get_model(model, base_url="https://api.githubcopilot.com", api_key=key)
    if model.startswith("openai:"):
        return get_model(model.removeprefix("openai:"))

    if model.startswith("simple-openai:"):
        from llmling_models.models.pyodide_model import SimpleOpenAIModel

        return SimpleOpenAIModel(model=model.removeprefix("simple-openai:"))

    if model.startswith("copilot:"):
        from httpx import AsyncClient
        from pydantic_ai.models.openai import OpenAIResponsesModel
        from pydantic_ai.providers.openai import OpenAIProvider

        token = os.getenv("GITHUB_COPILOT_API_KEY")
        headers = {
            "Authorization": f"Bearer {token}",
            "editor-version": "Neovim/0.9.0",
            "Copilot-Integration-Id": "vscode-chat",
        }
        client = AsyncClient(headers=headers)
        base_url = "https://api.githubcopilot.com"
        prov = OpenAIProvider(base_url=base_url, api_key=token, http_client=client)
        model_name = model.removeprefix("copilot:")
        return OpenAIResponsesModel(model_name=model_name, provider=prov)

    if model == "input":
        from llmling_models import InputModel

        return InputModel()
    if model.startswith("remote_model"):
        from llmling_models.models.remote_model import RemoteProxyModel

        return RemoteProxyModel(url=model.removeprefix("remote_model:"))
    if model.startswith("remote_input"):
        from llmling_models.models.remote_input import RemoteInputModel

        return RemoteInputModel(url=model.removeprefix("remote_input:"))
    if model.startswith("import:"):

        class Importer(BaseModel):
            model: ImportString[Any]

        imported = Importer(model=model.removeprefix("import:")).model
        return imported() if isinstance(imported, type) else imported  # type: ignore[no-any-return]
    if model == "test":
        from pydantic_ai.models.test import TestModel

        return TestModel()
    if model.startswith("test:"):
        from pydantic_ai.models.test import TestModel

        return TestModel(custom_output_text=model.removeprefix("test:"))
    if model.startswith("gemini:"):
        model = model.replace("gemini:", "google-gla:")
    return infer_model_(model)
