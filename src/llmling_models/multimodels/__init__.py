"""Configuration management for LLMling."""

from __future__ import annotations


from llmling_models.multimodels.fallback import FallbackMultiModel
from llmling_models.multimodels.delegation import DelegationMultiModel
from llmling_models.multimodels.cost import CostOptimizedMultiModel
from llmling_models.multimodels.token_optimized import TokenOptimizedMultiModel
from llmling_models.multimodels.userselect import UserSelectModel

__all__ = [
    "CostOptimizedMultiModel",
    "DelegationMultiModel",
    "FallbackMultiModel",
    "TokenOptimizedMultiModel",
    "UserSelectModel",
]
