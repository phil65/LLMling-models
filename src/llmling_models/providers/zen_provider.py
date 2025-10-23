r"""OpenCode Zen provider implementation for Pydantic AI."""

import os
from typing import ClassVar

from pydantic_ai.providers.openai import OpenAIProvider


class ZenProvider:
    """Factory for creating OpenCode Zen providers with model-specific endpoints."""

    # Model-to-endpoint mapping based on OpenCode Zen documentation
    MODEL_ENDPOINTS: ClassVar = {
        # Claude models use /messages endpoint
        "claude-sonnet-4-5": "/messages",
        "claude-sonnet-4": "/messages",
        "claude-3-5-sonnet": "/messages",
        "claude-3-opus": "/messages",
        "claude-haiku-4-5": "/messages",
        "claude-3-5-haiku": "/messages",
        "claude-opus-4-1": "/messages",
        # GPT models use /responses endpoint
        "gpt-5": "/responses",
        "gpt-5-codex": "/responses",
        "gpt-4": "/responses",
        "gpt-4-turbo": "/responses",
        "gpt-3.5-turbo": "/responses",
        # Other models (Qwen, Kimi, etc.) use standard /chat/completions
        "qwen3-coder": "/chat/completions",
        "qwen2.5-coder": "/chat/completions",
        "kimi-k2": "/chat/completions",
        "grok-code": "/chat/completions",
    }

    @classmethod
    def create_provider(
        cls, api_key: str, model_name: str, base_url: str = "https://opencode.ai/zen/v1"
    ) -> OpenAIProvider:
        """Create an OpenAI provider configured for OpenCode Zen with correct endpoint.

        Args:
            api_key: OpenCode Zen API key
            model_name: Name of the model (determines endpoint)
            base_url: Base URL for OpenCode Zen API (default: https://opencode.ai/zen/v1)

        Returns:
            Configured OpenAIProvider instance
        """
        # For grok-code and other models using /chat/completions, we can use the base URL
        # since the OpenAI client will automatically append /chat/completions
        if model_name in ["grok-code", "qwen3-coder", "qwen2.5-coder", "kimi-k2"]:
            # Use base URL directly - OpenAI client will append /chat/completions
            return OpenAIProvider(api_key=api_key, base_url=base_url)
        # For Claude and GPT models that need different endpoints,
        # we need the full endpoint URL
        endpoint = cls.get_model_endpoint(model_name, base_url)
        return OpenAIProvider(api_key=api_key, base_url=endpoint)

    @classmethod
    def get_model_endpoint(
        cls, model_name: str, base_url: str = "https://opencode.ai/zen/v1"
    ) -> str:
        """Get the appropriate endpoint for a given model.

        Args:
            model_name: Name of the model (e.g., 'claude-sonnet-4', 'gpt-5')
            base_url: Base URL for OpenCode Zen API

        Returns:
            Full endpoint URL for the model
        """
        # Check if model has a specific endpoint mapping
        if model_name in cls.MODEL_ENDPOINTS:
            endpoint = cls.MODEL_ENDPOINTS[model_name]
        else:
            # Default to /chat/completions for unknown models
            endpoint = "/chat/completions"

        return f"{base_url}{endpoint}"

    @classmethod
    def is_model_supported(cls, model_name: str) -> bool:
        """Check if a model is explicitly supported.

        Args:
            model_name: Name of the model to check

        Returns:
            True if model is in our mapping, False otherwise
        """
        return model_name in cls.MODEL_ENDPOINTS

    @classmethod
    def get_supported_models(cls) -> dict[str, str]:
        """Get all supported models and their endpoints.

        Returns:
            Dictionary mapping model names to their endpoints
        """
        return cls.MODEL_ENDPOINTS.copy()


def _create_zen_model(model_name: str):
    """Create OpenCode Zen model."""
    from pydantic_ai.models.openai import OpenAIChatModel

    if not model_name:
        msg = "ZEN_MODEL is required when using OpenCode Zen provider"
        raise ValueError(msg)

    # Create provider instance using the factory method
    api_key = os.getenv("ZEN_API_KEY")
    assert api_key, "ZEN_API_KEY is required when using OpenCode Zen provider"
    provider_instance = ZenProvider.create_provider(
        api_key=api_key, model_name=model_name
    )

    return OpenAIChatModel(
        model_name=model_name,
        provider=provider_instance,
    )
