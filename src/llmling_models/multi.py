"""Multi-model implementations."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, model_validator
from pydantic_ai.models import AgentModel, KnownModelName, Model, infer_model

from llmling_models.base import PydanticModel
from llmling_models.log import get_logger


if TYPE_CHECKING:
    from pydantic_ai.messages import Message, ModelAnyResponse
    from pydantic_ai.tools import ToolDefinition


logger = get_logger(__name__)


class MultiModel(PydanticModel):
    """Base for model configurations that combine multiple language models.

    This provides the base interface for YAML-configurable multi-model setups,
    allowing configuration of multiple models through LLMling's config system.
    """

    type: str
    """Discriminator field for multi-model types"""

    models: list[KnownModelName | Model] = Field(default_factory=list)
    """"List of models to use"."""

    @property
    def available_models(self) -> list[Model]:
        """Convert model names/instances to pydantic-ai Model instances."""
        return [
            model if isinstance(model, Model) else infer_model(model)  # type: ignore[arg-type]
            for model in self.models
        ]

    @model_validator(mode="after")
    def validate_models(self) -> MultiModel:
        """Validate model configuration."""
        if not self.models:
            msg = "At least one model must be provided"
            raise ValueError(msg)
        return self


class RandomMultiModel(MultiModel):
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
    def validate_models(self) -> RandomMultiModel:
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
        messages: list[Message],
    ) -> tuple[ModelAnyResponse, Any]:
        """Make request using randomly selected model."""
        models = await self._initialize_models()
        selected = random.choice(models)
        logger.debug("Selected model: %s", selected)
        return await selected.request(messages)


class FallbackMultiModel(MultiModel):
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
        messages: list[Message],
    ) -> tuple[ModelAnyResponse, Any]:
        """Try each model in sequence until one succeeds."""
        models = await self._initialize_models()
        last_error = None

        for model in models:
            try:
                logger.debug("Trying model: %s", model)
                return await model.request(messages)
            except Exception as e:  # noqa: BLE001
                last_error = e
                logger.debug("Model %s failed: %s", model, e)
                continue

        msg = f"All models failed. Last error: {last_error}"
        raise RuntimeError(msg) from last_error
