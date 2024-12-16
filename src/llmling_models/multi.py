"""Multi-model implementations."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Generic, Literal, Self

from pydantic import Field, model_validator
from pydantic_ai.models import AgentModel, Model, infer_model
from typing_extensions import TypeVar

from llmling_models.base import PydanticModel
from llmling_models.log import get_logger


if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.result import Cost
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition

logger = get_logger(__name__)
TModel = TypeVar("TModel", default=Model)


class MultiModel(PydanticModel, Generic[TModel]):
    """Base for model configurations that combine multiple language models.

    This provides the base interface for YAML-configurable multi-model setups,
    allowing configuration of multiple models through LLMling's config system.
    """

    type: str = Field(description="Discriminator field for multi-model types")
    models: list[str | TModel] = Field(
        description="List of models to use",
        min_length=1,
    )
    _initialized_models: list[TModel] | None = None

    @model_validator(mode="after")
    def initialize_models(self) -> MultiModel[TModel]:
        """Convert string model names to Model instances."""
        models: list[TModel] = []
        for model in self.models:
            if isinstance(model, str):
                models.append(infer_model(model))  # type: ignore[arg-type]
            else:
                models.append(model)
        self._initialized_models = models
        return self

    @property
    def available_models(self) -> list[TModel]:
        """Get initialized model instances."""
        if self._initialized_models is None:
            msg = "Models not initialized"
            raise RuntimeError(msg)
        return self._initialized_models


class RandomMultiModel(MultiModel[TModel]):
    """Randomly selects from configured models.

    Example YAML configuration:
        ```yaml
        model:
          type: random
          models:
            - openai:gpt-4
            - openai:gpt-3.5-turbo
        ```
    """

    type: Literal["random"] = "random"

    @model_validator(mode="after")
    def validate_models(self) -> Self:
        """Validate model configuration."""
        if not self.models:
            msg = "At least one model must be provided"
            raise ValueError(msg)
        return self

    def name(self) -> str:
        """Get descriptive model name."""
        return f"multi-random({len(self.models)})"

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create agent model that randomly selects from available models."""
        return RandomAgentModel(
            models=self.available_models,
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )


class RandomAgentModel(AgentModel):
    """AgentModel that randomly selects from available models."""

    def __init__(
        self,
        models: list[Model],
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> None:
        """Initialize with list of models."""
        if not models:
            msg = "At least one model must be provided"
            raise ValueError(msg)
        self.models = models
        self.function_tools = function_tools
        self.allow_text_result = allow_text_result
        self.result_tools = result_tools
        self._initialized_models: list[AgentModel] | None = None

    async def _initialize_models(self) -> list[AgentModel]:
        """Initialize all agent models."""
        if self._initialized_models is None:
            self._initialized_models = []
            for model in self.models:
                agent_model = await model.agent_model(
                    function_tools=self.function_tools,
                    allow_text_result=self.allow_text_result,
                    result_tools=self.result_tools,
                )
                self._initialized_models.append(agent_model)
        return self._initialized_models

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> tuple[ModelResponse, Cost]:
        """Make request using randomly selected model."""
        models = await self._initialize_models()
        selected = random.choice(models)
        logger.debug("Selected model: %s", selected)
        return await selected.request(messages, model_settings)


class FallbackMultiModel(MultiModel[TModel]):
    """Tries models in sequence until one succeeds.

    Example YAML configuration:
        ```yaml
        model:
          type: fallback
          models:
            - openai:gpt-4  # Try this first
            - openai:gpt-3.5-turbo  # Fall back to this if gpt-4 fails
            - ollama:llama2  # Last resort
        ```
    """

    type: Literal["fallback"] = "fallback"

    def name(self) -> str:
        """Get descriptive model name."""
        return f"multi-fallback({len(self.models)})"

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create agent model that implements fallback strategy."""
        return FallbackAgentModel(
            models=self.available_models,
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )


class FallbackAgentModel(AgentModel):
    """AgentModel that implements fallback strategy."""

    def __init__(
        self,
        models: list[Model],
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> None:
        """Initialize with ordered list of models."""
        if not models:
            msg = "At least one model must be provided"
            raise ValueError(msg)
        self.models = models
        self.function_tools = function_tools
        self.allow_text_result = allow_text_result
        self.result_tools = result_tools
        self._initialized_models: list[AgentModel] | None = None

    async def _initialize_models(self) -> list[AgentModel]:
        """Initialize all agent models."""
        if self._initialized_models is None:
            self._initialized_models = []
            for model in self.models:
                agent_model = await model.agent_model(
                    function_tools=self.function_tools,
                    allow_text_result=self.allow_text_result,
                    result_tools=self.result_tools,
                )
                self._initialized_models.append(agent_model)
        return self._initialized_models

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> tuple[ModelResponse, Cost]:
        """Try each model in sequence until one succeeds."""
        models = await self._initialize_models()
        last_error = None

        for model in models:
            try:
                logger.debug("Trying model: %s", model)
                return await model.request(messages, model_settings)
            except Exception as e:  # noqa: BLE001
                last_error = e
                logger.debug("Model %s failed: %s", model, e)
                continue

        msg = f"All models failed. Last error: {last_error}"
        raise RuntimeError(msg) from last_error
