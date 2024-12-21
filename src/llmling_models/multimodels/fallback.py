"""Multi-model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field
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

    type: Literal["fallback"] = Field(default="fallback", init=False)

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
        return FallbackAgentModel[TModel](
            models=self.available_models,
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )


class FallbackAgentModel[TModel: Model](AgentModel):
    """AgentModel that implements fallback strategy."""

    def __init__(
        self,
        models: list[TModel],
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
    ) -> tuple[ModelResponse, Usage]:
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
