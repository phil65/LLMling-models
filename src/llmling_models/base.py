"""Base class for YAML-configurable pydantic-ai models."""

from __future__ import annotations

from pydantic import ConfigDict
from pydantic_ai.models import Model
from schemez import Schema


class PydanticModel(Schema, Model):
    """Base for models that can be configured via YAML."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
