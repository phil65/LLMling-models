"""Base class for YAML-configurable pydantic-ai models."""

from __future__ import annotations

from abc import ABC

from pydantic import BaseModel, ConfigDict
from pydantic_ai.models import Model


class PydanticModel(Model, BaseModel, ABC):
    """Base for models that can be configured via YAML."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        use_attribute_docstrings=True,
    )
