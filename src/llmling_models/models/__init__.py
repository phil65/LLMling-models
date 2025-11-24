"""Multi-models."""

from __future__ import annotations


from llmling_models.models.delegation import DelegationMultiModel
from llmling_models.models.augmented import AugmentedModel
from llmling_models.models.userselect import UserSelectModel
from llmling_models.models.input_model import InputModel
from llmling_models.models.pyodide_model import SimpleOpenAIModel
from llmling_models.models.remote_input import RemoteInputModel
from llmling_models.models.remote_model import RemoteProxyModel

__all__ = [
    "AugmentedModel",
    "DelegationMultiModel",
    "InputModel",
    "RemoteInputModel",
    "RemoteProxyModel",
    "SimpleOpenAIModel",
    "UserSelectModel",
]
