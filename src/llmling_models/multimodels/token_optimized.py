"""Token-limit optimized model selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

from pydantic import Field
from pydantic_ai.models import AgentModel, Model

from llmling_models.log import get_logger
from llmling_models.multi import MultiModel
from llmling_models.utils import estimate_tokens, get_model_limits


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.result import Usage
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition

logger = get_logger(__name__)
TModel = TypeVar("TModel", bound=Model)


class TokenOptimizedMultiModel[TModel: Model](MultiModel[TModel]):
    """Multi-model that selects based on input token count.

    Example YAML configuration:
        ```yaml
        model:
          type: token-optimized
          models:
            - openai:gpt-4  # 8k context
            - openai:gpt-4-32k  # 32k context
            - openai:gpt-3.5-turbo-16k  # 16k context
          strategy: efficient  # Use smallest sufficient context window
        ```
    """

    type: Literal["token-optimized"] = Field(default="token-optimized", init=False)

    strategy: Literal["efficient", "maximum_context"] = Field(default="efficient")
    """Model selection strategy."""

    def name(self) -> str:
        """Get descriptive model name."""
        return f"token-optimized({len(self.models)})"

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create agent model that implements token-based selection."""
        return TokenOptimizedAgentModel[TModel](
            models=self.available_models,
            strategy=self.strategy,
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )


class TokenOptimizedAgentModel[TModel: Model](AgentModel):
    """AgentModel that implements token-based model selection."""

    def __init__(
        self,
        models: Sequence[TModel],
        strategy: str,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ):
        """Initialize with models and selection strategy."""
        if not models:
            msg = "At least one model must be provided"
            raise ValueError(msg)
        self.models = models
        self.strategy = strategy
        self.function_tools = function_tools
        self.allow_text_result = allow_text_result
        self.result_tools = result_tools
        self._initialized_models: dict[str, AgentModel] = {}

    async def _select_model(
        self,
        messages: list[ModelMessage],
    ) -> AgentModel:
        """Select appropriate model based on token counts."""
        token_estimate = estimate_tokens(messages)
        logger.debug("Estimated token count: %d", token_estimate)

        # Define model capabilities order (smaller number = less capable)
        model_capabilities = {
            "gpt-3.5-turbo": 1,  # Base model
            "gpt-3.5-turbo-16k": 2,  # Same but larger context
            "gpt-4-turbo": 3,  # More capable and largest context
        }

        # Get available models that can handle the token count
        model_options: list[
            tuple[AgentModel, int, int]
        ] = []  # (model, capability, limit)
        for model in self.models:
            model_name = model.name()
            if model_name not in self._initialized_models:
                self._initialized_models[model_name] = await model.agent_model(
                    function_tools=self.function_tools,
                    allow_text_result=self.allow_text_result,
                    result_tools=self.result_tools,
                )

            # Check token limits
            limits = await get_model_limits(model_name)
            if not limits:
                logger.debug("No token limits for %s, skipping", model_name)
                continue

            if token_estimate <= limits.input_tokens:
                capability = model_capabilities.get(model_name, 0)
                model_estimates = (
                    self._initialized_models[model_name],
                    capability,
                    limits.input_tokens,
                )
                model_options.append(model_estimates)
                logger.debug(
                    "Model %s (capability %d) can handle %d tokens (limit: %d)",
                    model_name,
                    capability,
                    token_estimate,
                    limits.input_tokens,
                )

        if not model_options:
            msg = f"No suitable model found for {token_estimate} tokens"
            raise RuntimeError(msg)

        # Sort first by capability, then by limit
        model_options.sort(key=lambda x: (x[1], x[2]))

        if self.strategy == "efficient":
            # Use least capable model that can handle the input
            selected, capability, limit = model_options[0]
        else:  # maximum_context
            # Use most capable model available
            selected, capability, limit = model_options[-1]

        logger.info(
            "Selected %s (capability %d) with %d token limit",
            selected.__class__.__name__,
            capability,
            limit,
        )
        return selected

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> tuple[ModelResponse, Usage]:
        """Process request using token-optimized model selection."""
        selected_model = await self._select_model(messages)
        return await selected_model.request(messages, model_settings)
