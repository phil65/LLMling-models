"""Utility functions for model handling."""

from __future__ import annotations

from decimal import Decimal
import importlib.util
import logging
import os
from typing import TYPE_CHECKING

from pydantic import BaseModel, ImportString
from pydantic_ai.messages import (
    ModelMessage,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import infer_model as infer_model_


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
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


def is_pyodide() -> bool:
    """Check if code is running in a Pyodide environment."""
    try:
        from js import Object  # type: ignore  # noqa: F401

        return True  # noqa: TRY300
    except ImportError:
        return False


def get_model(
    model_name: str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Model:
    if is_pyodide() or not importlib.util.find_spec("openai"):
        from llmling_models.pyodide_model import SimpleOpenAIModel

        return SimpleOpenAIModel(model=model_name, api_key=api_key, base_url=base_url)
    from pydantic_ai.models.openai import OpenAIModel

    return OpenAIModel(model_name, api_key=api_key, base_url=base_url)


def infer_model(model) -> Model:  # noqa: PLR0911
    """Extended infer_model from pydantic-ai."""
    if not isinstance(model, str):
        return model

    if model.startswith("openrouter:"):
        return get_model(
            model.removeprefix("openrouter:").replace(":", "/"),
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    if model.startswith("grok:"):
        return get_model(
            model.removeprefix("grok:"),
            base_url="https://api.x.ai/v1",
            api_key=os.getenv("X_AI_API_KEY") or os.getenv("GROK_API_KEY"),
        )
    if model.startswith("deepseek:"):
        return get_model(
            model.removeprefix("deepseek:"),
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )
    if model.startswith("perplexity:"):
        return get_model(
            model.removeprefix("perplexity:"),
            base_url="https://api.perplexity.ai",
            api_key=os.getenv("PERPLEXITY_API_KEY"),
        )

    if model.startswith("lm-studio:"):
        return get_model(
            model.removeprefix("lm-studio:"),
            base_url="http://localhost:1234/v1/",
            api_key="lm-studio",
        )

    if model.startswith("llm:"):
        from llmling_models.llm_adapter import LLMAdapter

        return LLMAdapter(model_name=model.removeprefix("llm:"))

    if model.startswith("openai:"):
        return get_model(model.removeprefix("openai:"))

    if model.startswith("simple-openai:"):
        from llmling_models.pyodide_model import SimpleOpenAIModel

        return SimpleOpenAIModel(model=model.removeprefix("simple-openai:"))

    if model.startswith("copilot:"):
        from httpx import AsyncClient
        from pydantic_ai.models.openai import OpenAIModel

        token = os.getenv("GITHUB_COPILOT_API_KEY")
        client = AsyncClient(
            headers={
                "Authorization": f"Bearer {token}",
                "editor-version": "Neovim/0.9.0",
                "Copilot-Integration-Id": "vscode-chat",
            }
        )
        return OpenAIModel(
            model_name=model.removeprefix("copilot:"),
            base_url="https://api.githubcopilot.com",
            api_key=token,
            http_client=client,
        )

    if model.startswith("aisuite:"):
        from llmling_models.aisuite_adapter import AISuiteAdapter

        return AISuiteAdapter(model=model.removeprefix("aisuite:"))
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

        return TestModel(custom_result_text=model.removeprefix("test:"))
    return infer_model_(model)  # type: ignore


def estimate_tokens(messages: list[ModelMessage]) -> int:
    """Estimate token count for messages using available tokenizers.

    Will try to use tiktoken if available (best for OpenAI models),
    falling back to Mistral's tokenizer (good modern default),
    and finally using a simple character-based estimation.
    """
    # Collect all content from relevant message parts
    content = ""
    for message in messages:
        for part in message.parts:
            if isinstance(
                part,
                UserPromptPart | SystemPromptPart | TextPart | ToolReturnPart,
            ):
                content += str(part.content)

    # Try tiktoken (best for OpenAI models)
    if importlib.util.find_spec("tiktoken"):
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(content))

    # Try transformers with Mistral's tokenizer
    if importlib.util.find_spec("transformers"):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        return len(tokenizer.encode(content))

    # Fallback to simple character-based estimation
    return len(content) // 4


def estimate_request_cost(
    costs: dict[str, str] | ModelCosts,
    token_count: int,
) -> Decimal:
    """Estimate input cost for a request.

    Args:
        costs: Cost information (dict or ModelCosts object)
        token_count: Number of tokens in the request

    Returns:
        Decimal: Estimated input cost in USD
    """
    # Extract input cost per token
    if isinstance(costs, dict):
        input_cost = Decimal(costs["input_cost_per_token"])
    else:
        input_cost = Decimal(str(costs.input_cost_per_token))

    estimated_cost = input_cost * token_count
    logger.debug(
        "Estimated cost: %s * %d tokens = %s",
        input_cost,
        token_count,
        estimated_cost,
    )
    return estimated_cost
