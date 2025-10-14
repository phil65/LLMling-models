"""Utility functions for model handling."""

from __future__ import annotations

from decimal import Decimal
import functools
import importlib.util
import inspect
import logging
import os
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, ImportString, TypeAdapter
from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import infer_model as infer_model_
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel

from llmling_models.formatting import format_part


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic_ai import ModelMessage, ModelResponsePart
    from pydantic_ai.models import Model
    from pydantic_ai.models.function import AgentInfo
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


def function_to_model(callback: Callable) -> FunctionModel:
    """Factory to get a text model for Callables with "simpler" signatures.

    Pydantic-AIs FunctionModel requires a very "specific" callable. This function serves
    as  helper to allow creating FunctionModels which take either no arguments or a
    single argument in form of a prompt.
    """
    sig = inspect.signature(callback)
    # Count required parameters (those without defaults)
    required_params = sum(
        1 for param in sig.parameters.values() if param.default is inspect.Parameter.empty
    )
    takes_prompt = required_params > 0

    @functools.wraps(callback)
    async def callback_wrapper(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> ModelResponse:
        try:
            if takes_prompt:
                prompt = format_part(messages[-1].parts[0])
                if inspect.iscoroutinefunction(callback):
                    result = await callback(prompt)
                else:
                    result = callback(prompt)
            elif inspect.iscoroutinefunction(callback):
                result = await callback()
            else:
                result = callback()

            if isinstance(result, str):
                part: ModelResponsePart = TextPart(result)
            else:
                part = ToolCallPart(tool_name="final_result", args=result.model_dump())
            return ModelResponse(parts=[part])
        except Exception as e:
            logger.exception("Processor callback failed")
            name = getattr(callback, "__name__", str(callback))
            msg = f"Processor error in {name!r}: {e}"
            raise RuntimeError(msg) from e

    return FunctionModel(callback_wrapper)


def without_unprocessed_tool_calls(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Clean message history by removing unprocessed tool calls.

    This removes ToolCallPart from the last ModelResponse if it has unprocessed
    tool calls, but preserves all text content and reasoning.
    """
    cleaned_messages = list(messages)  # Make a copy to avoid modifying the original
    # Check if the last message is a ModelResponse with unprocessed tool calls
    if cleaned_messages:
        # Import at runtime to avoid circular imports

        last_message = cleaned_messages[-1]
        if isinstance(last_message, ModelResponse) and last_message.tool_calls:
            # Create a new ModelResponse with the same content but without tool calls
            filtered_parts = [
                part for part in last_message.parts if not isinstance(part, ToolCallPart)
            ]

            # Only replace if we actually removed some tool calls
            if len(filtered_parts) != len(last_message.parts):
                # Create a new ModelResponse with filtered parts
                cleaned_response = ModelResponse(
                    parts=filtered_parts,
                    usage=last_message.usage,
                    model_name=last_message.model_name,
                    timestamp=last_message.timestamp,
                    provider_name=last_message.provider_name,
                    provider_details=last_message.provider_details,
                    provider_response_id=last_message.provider_response_id,
                    finish_reason=last_message.finish_reason,
                )
                cleaned_messages[-1] = cleaned_response

    return cleaned_messages


PydanticAIMessage = ModelRequest | ModelResponse
message_adapter: TypeAdapter[PydanticAIMessage] = TypeAdapter(
    PydanticAIMessage,
    config=ConfigDict(ser_json_bytes="base64", val_json_bytes="base64"),
)


def serialize_message(message: PydanticAIMessage) -> str:
    """Serialize pydantic-ai message.

    The `ctx` field in the `RetryPromptPart` is optionally dict[str, Any],
    which is not always serializable.
    """
    for part in message.parts:
        if isinstance(part, RetryPromptPart) and isinstance(part.content, list):
            for content in part.content:
                content["ctx"] = {
                    k: str(v) for k, v in (content.get("ctx", None) or {}).items()
                }
    return message_adapter.dump_python(message, mode="json")
