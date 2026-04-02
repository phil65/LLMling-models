"""OpenAI-compatible API server for Pydantic-AI models."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import tokonomics

from llmling_models.log import get_logger
from llmling_models.models.helpers import infer_model
from llmling_models.openai_server.models import OpenAIModelInfo


if TYPE_CHECKING:
    from pydantic_ai.models import Model


logger = get_logger(__name__)


class ModelRegistry:
    """Registry of available models."""

    def __init__(self, models: dict[str, str | Model] | None = None) -> None:
        """Initialize model registry.

        Args:
            models: Dictionary mapping model names to models or model identifiers
        """
        self.models: dict[str, Model] = {}
        if models:
            for name, model_or_id in models.items():
                if isinstance(model_or_id, str):
                    self.models[name] = infer_model(model_or_id)
                else:
                    self.models[name] = model_or_id

    @classmethod
    async def create(cls) -> ModelRegistry:
        """Create a model registry populated with all models from tokonomics.

        Returns:
            A new ModelRegistry instance with auto-populated models.
        """
        registry = cls({})  # Empty registry
        try:
            all_models = await tokonomics.get_all_models()
            for model_info in all_models:
                try:
                    # Use the pydantic_model_id directly as the key
                    model_id = model_info.pydantic_ai_id
                    registry.models[model_id] = infer_model(model_id)
                    logger.debug("Auto-registered model: %s", model_id)
                except Exception as e:  # noqa: BLE001
                    msg = "Failed to register model %s: %s"
                    logger.warning(msg, model_info.pydantic_ai_id, str(e))

            logger.info("Auto-populated %d models from tokonomics", len(registry.models))
        except Exception as e:  # noqa: BLE001
            logger.warning("Error auto-populating models: %s", str(e))

        return registry

    def add_model(self, name: str, model_or_id: str | Model) -> None:
        """Add a model to the registry."""
        model = infer_model(model_or_id) if isinstance(model_or_id, str) else model_or_id
        self.models[name] = model

    def get_model(self, name: str) -> Model:
        """Get a model by name."""
        try:
            return self.models[name]
        except KeyError:
            raise ValueError(f"Model {name} not found") from None

    def list_models(self) -> list[OpenAIModelInfo]:
        """List available models."""
        return [
            OpenAIModelInfo(
                id=n,
                created=int(time.time()),
                description=f"Model {n}",
                object="model",
                owned_by="agentpool",
            )
            for n in self.models
        ]
