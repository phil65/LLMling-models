"""Configuration management for LLMling."""

from __future__ import annotations


from llmling_models.multimodels.random import RandomMultiModel
from llmling_models.multimodels.fallback import FallbackMultiModel

__all__ = [
    "FallbackMultiModel",
    "RandomMultiModel",
]
