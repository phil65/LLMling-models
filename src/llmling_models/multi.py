"""Multi-model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, model_validator
from pydantic_ai.models import Model, infer_model
from typing_extensions import TypeVar

from llmling_models.base import PydanticModel
from llmling_models.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = get_logger(__name__)
TModel = TypeVar("TModel", bound=Model)


class MultiModel[TModel](PydanticModel):
    """Base for model configurations that combine multiple language models.

    This provides the base interface for YAML-configurable multi-model setups,
    allowing configuration of multiple models through LLMling's config system.
    """

    type: str = Field(init=False)
    """Discriminator field for multi-model types"""

    models: list[str | Model] = Field(min_length=1)
    """List of models to use."""

    _initialized_models: list[TModel] | None = None

    @model_validator(mode="after")
    def initialize_models(self) -> MultiModel[TModel]:
        """Convert string model names to Model instances."""
        models: list[TModel] = []
        for model in self.models:
            if isinstance(model, str):
                models.append(infer_model(model))  # type: ignore[arg-type]
            else:
                models.append(model)  # type: ignore
        self._initialized_models = models
        return self

    @property
    def available_models(self) -> Sequence[TModel]:
        """Get initialized model instances."""
        if self._initialized_models is None:
            msg = "Models not initialized"
            raise RuntimeError(msg)
        return self._initialized_models
