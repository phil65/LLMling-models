"""Multi-model implementations."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field, model_validator
from pydantic.config import ConfigDict
from pydantic_ai.models import Model
from schemez import Schema

from llmling_models.log import get_logger
from llmling_models.models.helpers import infer_model


logger = get_logger(__name__)


class MultiModel(Schema, Model):
    """Base for model configurations that combine multiple language models.

    This provides the base interface for YAML-configurable multi-model setups,
    allowing configuration of multiple models through LLMling's config system.
    """

    models: Sequence[str | Model] = Field(min_length=1)
    """List of models to use."""

    _initialized_models: list[Model] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def initialize_models(self) -> MultiModel:
        """Convert string model names to Model instances."""
        models: list[Model] = []
        for model in self.models:
            if isinstance(model, str):
                models.append(infer_model(model))
            else:
                models.append(model)
        self._initialized_models = models
        return self

    @property
    def available_models(self) -> Sequence[Model]:
        """Get initialized model instances."""
        if self._initialized_models is None:
            msg = "Models not initialized"
            raise RuntimeError(msg)
        return self._initialized_models
