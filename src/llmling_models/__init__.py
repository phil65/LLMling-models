"""LLMling-models: main package.

Pydantic-AI models for LLMling-agent.
"""

from __future__ import annotations

from importlib.metadata import version

__version__ = version("llmling-models")
__title__ = "LLMling-models"

__author__ = "Philipp Temminghoff"
__author_email__ = "philipptemminghoff@googlemail.com"
__copyright__ = "Copyright (c) 2024 Philipp Temminghoff"
__license__ = "MIT"
__url__ = "https://github.com/phil65/llmling-models"

from llmling_models.models.multi import MultiModel
from llmling_models.models.input_model import InputModel, DefaultInputHandler
from llmling_models.models import DelegationMultiModel, UserSelectModel
from llmling_models.models.function_to_model import function_to_model
from llmling_models.models.helpers import infer_model
from llmling_models.model_types import AllModels, ModelInput
from llmling_models.toolsets.codemode_toolset import CodeModeToolset

__version__ = version("llmling-models")

__all__ = [
    "AllModels",
    "CodeModeToolset",
    "DefaultInputHandler",
    "DelegationMultiModel",
    "InputModel",
    "ModelInput",
    "MultiModel",
    "UserSelectModel",
    "function_to_model",
    "infer_model",
]
