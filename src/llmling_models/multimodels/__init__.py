"""Configuration management for LLMling."""

from __future__ import annotations


from llmling_models.multimodels.fallback import FallbackMultiModel
from llmling_models.multimodels.delegation import DelegationMultiModel
from llmling_models.multimodels.cost import CostOptimizedMultiModel
from llmling_models.multimodels.token import TokenOptimizedMultiModel

__all__ = [
    "CostOptimizedMultiModel",
    "DelegationMultiModel",
    "FallbackMultiModel",
    "TokenOptimizedMultiModel",
]
