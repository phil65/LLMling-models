"""Model configuration.

This is a lightweight config-only package for fast imports.
For the actual models, use `from llmling_models import ...`.
"""

from __future__ import annotations

from llmling_models_config.configs import (
    AnyModelConfig,
    AnthropicModelConfig,
    AugmentedModelConfig,
    BaseModelConfig,
    ClaudeCodeFullModelName,
    ClaudeCodeModelConfig,
    ClaudeCodeModelName,
    ClaudeCodePermissionMode,
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
    "ClaudeCodeFullModelName",
    "ClaudeCodeModelConfig",
    "ClaudeCodeModelName",
    "ClaudeCodePermissionMode",
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
