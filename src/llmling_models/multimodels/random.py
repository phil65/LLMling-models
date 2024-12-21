"""Multi-model implementations."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, model_validator
from pydantic_ai.models import AgentModel, Model
from typing_extensions import TypeVar

from llmling_models.log import get_logger
from llmling_models.multi import MultiModel


if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.result import Usage
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition

logger = get_logger(__name__)
TModel = TypeVar("TModel", bound=Model)


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

    type: Literal["random"] = Field(default="random", init=False)

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
        return RandomAgentModel[TModel](
            models=self.available_models,
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )


class RandomAgentModel[TModel: Model](AgentModel):
    """AgentModel that randomly selects from available models."""

    def __init__(
        self,
        models: list[TModel],
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
    ) -> tuple[ModelResponse, Usage]:
        """Make request using randomly selected model."""
        models = await self._initialize_models()
        selected = random.choice(models)
        logger.debug("Selected model: %s", selected)
        return await selected.request(messages, model_settings)
