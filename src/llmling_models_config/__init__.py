"""Model configuration.

This is a lightweight config-only package for fast imports.
For the actual models, use `from llmling_models import ...`.
"""

from llmling_models_config.configs import (
    AnyModelConfig,
    AnthropicModelConfig,
    AugmentedModelConfig,
    BaseModelConfig,
    DelegationModelConfig,
    FallbackModelConfig,
    FunctionModelConfig,
    GeminiModelConfig,
    ImportModelConfig,
    InputModelConfig,
    ModelSettings,
    OpenAIModelConfig,
    OpenRouterModelConfig,
    PrePostPromptConfig,
    RemoteInputConfig,
    RemoteProxyConfig,
    StringModelConfig,
    TestModelConfig,
    UserSelectModelConfig,
)


__all__ = [
    "AnthropicModelConfig",
    "AnyModelConfig",
    "AugmentedModelConfig",
    "BaseModelConfig",
    "DelegationModelConfig",
    "FallbackModelConfig",
    "FunctionModelConfig",
    "GeminiModelConfig",
    "ImportModelConfig",
    "InputModelConfig",
    "ModelSettings",
    "OpenAIModelConfig",
    "OpenRouterModelConfig",
    "PrePostPromptConfig",
    "RemoteInputConfig",
    "RemoteProxyConfig",
    "StringModelConfig",
    "TestModelConfig",
    "UserSelectModelConfig",
]
