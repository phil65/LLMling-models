"""Configuration management for LLMling."""

from __future__ import annotations


from llmling_models.multimodels.random import RandomMultiModel
from llmling_models.multimodels.fallback import FallbackMultiModel
from llmling_models.multimodels.delegation import DelegationMultiModel
from llmling_models.multimodels.cost import CostOptimizedMultiModel

__all__ = [
    "CostOptimizedMultiModel",
    "DelegationMultiModel",
    "FallbackMultiModel",
    "RandomMultiModel",
]
