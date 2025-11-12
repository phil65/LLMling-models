"""Multi-models."""

from __future__ import annotations


from llmling_models.multimodels.delegation import DelegationMultiModel

from llmling_models.multimodels.userselect import UserSelectModel

__all__ = [
    "DelegationMultiModel",
    "UserSelectModel",
]
