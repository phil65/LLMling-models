"""Providers package."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from pydantic_ai.providers import infer_provider as _infer_provider
import os

if TYPE_CHECKING:
    from pydantic_ai.providers import Provider


def infer_provider(provider: str) -> Provider[Any]:
    """Infer the provider from the provider name."""
    if provider == "copilot":
        from llmling_models.providers.copilot_provider import CopilotProvider

        return CopilotProvider()
    if provider == "openrouter":
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        return OpenRouterProvider()
    if provider == "grok":
        from pydantic_ai.providers.grok import GrokProvider

        api_key = os.environ.get("X_AI_API_KEY") or os.environ.get("GROK_API_KEY") or ""
        return GrokProvider(api_key=api_key)
    if provider == "perplexity":
        from pydantic_ai.providers.openai import OpenAIProvider

        api_key = os.environ.get("PERPLEXITY_API_KEY") or ""
        return OpenAIProvider(base_url="https://api.perplexity.ai", api_key=api_key)
    if provider == "lm-studio":
        from llmling_models.providers.lm_studio_provider import LMStudioProvider

        return LMStudioProvider()

    return _infer_provider(provider)
