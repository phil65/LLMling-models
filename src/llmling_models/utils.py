"""Utility functions for model handling."""

from __future__ import annotations

from decimal import Decimal
import importlib.util
import logging
import os
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ImportString
from pydantic_ai.messages import (
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import infer_model as infer_model_
from pydantic_ai.models.openai import OpenAIChatModel


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pydantic_ai.messages import (
        ModelMessage,
    )
    from pydantic_ai.models import Model
    from tokonomics import ModelCosts, TokenLimits


async def get_model_costs(model_name: str) -> ModelCosts | None:
    """Get costs for model using tokonomics."""
    from tokonomics import get_model_costs

    return await get_model_costs(model_name)


async def get_model_limits(model_name: str) -> TokenLimits | None:
    """Get token limits for model using tokonomics."""
    from tokonomics import get_model_limits

    return await get_model_limits(model_name)


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
        from llmling_models.pyodide_model import SimpleOpenAIModel

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


def infer_model(
    model,
    provider: Literal["pydantic-ai", "litellm", "llm", "aisuite"] = "pydantic-ai",
) -> Model:
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
            return _infer_single_model(model, provider)

        # Create fallback model chain
        default_model = _infer_single_model(model_names[0], provider)
        fallback_models = [_infer_single_model(m, provider) for m in model_names[1:]]
        return FallbackModel(default_model, *fallback_models)

    # Regular single model case
    return _infer_single_model(model, provider)


def _infer_single_model(  # noqa: PLR0911
    model,
    provider: Literal["pydantic-ai", "litellm", "llm", "aisuite"] = "pydantic-ai",
) -> Model:
    """Extended infer_model from pydantic-ai."""
    if not isinstance(model, str):
        return model

    if provider == "litellm" or model.startswith("litellm:"):
        from llmling_models.adapters import LiteLLMAdapter

        return LiteLLMAdapter(model=model.removeprefix("litellm:"))
    if provider == "llm" or model.startswith("llm:"):
        from llmling_models.adapters import LLMAdapter

        return LLMAdapter(model=model.removeprefix("llm:"))
    if provider == "aisuite" or model.startswith("aisuite:"):
        from llmling_models.adapters import AISuiteAdapter

        return AISuiteAdapter(model=model.removeprefix("aisuite:"))

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
    if model.startswith("copilot:"):
        key = os.getenv("GITHUB_COPILOT_API_KEY")
        return get_model(model, base_url="https://api.githubcopilot.com", api_key=key)
    if model.startswith("openai:"):
        return get_model(model.removeprefix("openai:"))

    if model.startswith("simple-openai:"):
        from llmling_models.pyodide_model import SimpleOpenAIModel

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
        from llmling_models.remote_model.client import RemoteProxyModel

        return RemoteProxyModel(url=model.removeprefix("remote_model:"))
    if model.startswith("remote_input"):
        from llmling_models.remote_input.client import RemoteInputModel

        return RemoteInputModel(url=model.removeprefix("remote_input:"))
    if model.startswith("import:"):

        class Importer(BaseModel):
            model: ImportString

        imported = Importer(model=model.removeprefix("import:")).model
        return imported() if isinstance(imported, type) else imported
    if model.startswith("test:"):
        from pydantic_ai.models.test import TestModel

        return TestModel(custom_output_text=model.removeprefix("test:"))
    if model.startswith("gemini:"):
        model = model.replace("gemini:", "google-gla:")
    return infer_model_(model)  # type: ignore


def estimate_tokens(messages: list[ModelMessage]) -> int:
    """Estimate token count for messages using available tokenizers.

    Will try to use tiktoken if available (best for OpenAI models),
    falling back to Mistral's tokenizer (good modern default),
    and finally using a simple character-based estimation.
    """
    import tokonomics

    content = ""
    for message in messages:
        for part in message.parts:
            if isinstance(
                part,
                UserPromptPart | SystemPromptPart | TextPart | ToolReturnPart,
            ):
                content += str(part.content)
    return tokonomics.count_tokens(content)


def estimate_request_cost(costs: ModelCosts, token_count: int) -> Decimal:
    """Estimate input cost for a request.

    Args:
        costs: Cost information (dict or ModelCosts object)
        token_count: Number of tokens in the request

    Returns:
        Decimal: Estimated input cost in USD
    """
    # Extract input cost per token
    input_cost = Decimal(costs["input_cost_per_token"])
    estimated_cost = input_cost * token_count
    msg = "Estimated cost: %s * %d tokens = %s"
    logger.debug(msg, input_cost, token_count, estimated_cost)
    return estimated_cost
