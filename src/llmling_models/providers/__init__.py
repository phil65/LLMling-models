"""Providers package."""

from typing import Any
from pydantic_ai.providers import Provider, infer_provider as _infer_provider


def infer_provider(provider: str) -> Provider[Any]:  # noqa: PLR0911
    """Infer the provider from the provider name."""
    if provider == "copilot":
        from llmling_models.providers.copilot_provider import CopilotProvider

        return CopilotProvider()
    if provider == "openrouter":
        from llmling_models.providers.openrouter_provider import OpenRouterProvider

        return OpenRouterProvider()
    if provider == "grok":
        from llmling_models.providers.grok_provider import GrokProvider

        return GrokProvider()
    if provider == "perplexity":
        from llmling_models.providers.perplexity_provider import PerplexityProvider

        return PerplexityProvider()
    if provider == "lm-studio":
        from llmling_models.providers.lm_studio_provider import LMStudioProvider

        return LMStudioProvider()
    if provider == "together":
        from llmling_models.providers.together_provider import TogetherProvider

        return TogetherProvider()
    if provider == "ovhcloud":
        from llmling_models.providers.ovhcloud_provider import OVHCloudProvider

        return OVHCloudProvider()

    return _infer_provider(provider)
