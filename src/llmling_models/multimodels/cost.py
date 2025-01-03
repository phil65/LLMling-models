"""Cost-optimized model selection."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Literal, TypeVar

from pydantic import Field
from pydantic_ai.models import AgentModel, Model

from llmling_models.log import get_logger
from llmling_models.multi import MultiModel
from llmling_models.utils import get_model_costs


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.result import Usage
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition
    from tokonomics import ModelCosts

logger = get_logger(__name__)
TModel = TypeVar("TModel", bound=Model)


class CostOptimizedMultiModel[TModel: Model](MultiModel[TModel]):
    """Multi-model that selects based on cost and token limits.

    Example YAML configuration:
        ```yaml
        model:
          type: cost-optimized
          models:
            - openai:gpt-4  # Expensive but powerful
            - openai:gpt-3.5-turbo  # Good balance
            - openai:gpt-3.5-turbo-16k  # For long contexts
          max_cost: 0.5  # Maximum cost in USD per request
          strategy: cheapest_possible
        ```
    """

    type: Literal["cost-optimized"] = Field(default="cost-optimized", init=False)
    max_cost: float = Field(
        description="Maximum allowed cost in USD per request",
        gt=0,
    )
    strategy: Literal["cheapest_possible", "best_within_budget"] = Field(
        default="best_within_budget",
        description="Strategy for model selection",
    )

    def name(self) -> str:
        """Get descriptive model name."""
        return f"cost-optimized({len(self.models)})"

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create agent model that implements cost-based selection."""
        return CostOptimizedAgentModel[TModel](
            models=self.available_models,
            max_cost=Decimal(str(self.max_cost)),
            strategy=self.strategy,
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )


class CostOptimizedAgentModel[TModel: Model](AgentModel):
    """AgentModel that implements cost-based model selection."""

    def __init__(
        self,
        models: Sequence[TModel],
        max_cost: Decimal,
        strategy: str,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> None:
        """Initialize with models and cost settings."""
        if not models:
            msg = "At least one model must be provided"
            raise ValueError(msg)
        self.models = models
        self.max_cost = max_cost
        self.strategy = strategy
        self.function_tools = function_tools
        self.allow_text_result = allow_text_result
        self.result_tools = result_tools
        self._initialized_models: dict[str, AgentModel] = {}

    async def _get_model_info(
        self,
        model: Model,
        token_estimate: int,
    ) -> tuple[AgentModel, ModelCosts | None]:
        """Get initialized model and its cost information."""
        model_name = model.name()
        if model_name not in self._initialized_models:
            self._initialized_models[model_name] = await model.agent_model(
                function_tools=self.function_tools,
                allow_text_result=self.allow_text_result,
                result_tools=self.result_tools,
            )

        costs = await get_model_costs(model_name)
        return self._initialized_models[model_name], costs

    async def _select_model(
        self,
        messages: list[ModelMessage],
    ) -> AgentModel:
        """Select appropriate model based on costs and token counts."""
        from llmling_models.utils import (
            estimate_request_cost,
            estimate_tokens,
            get_model_costs,
            get_model_limits,
        )

        token_estimate = estimate_tokens(messages)
        logger.debug("Estimated token count: %d", token_estimate)

        # Get cost estimates for each model
        model_estimates: list[tuple[AgentModel, Decimal]] = []
        for model in self.models:
            model_name = model.name()
            if model_name not in self._initialized_models:
                self._initialized_models[model_name] = await model.agent_model(
                    function_tools=self.function_tools,
                    allow_text_result=self.allow_text_result,
                    result_tools=self.result_tools,
                )

            costs = await get_model_costs(model_name)
            if not costs:
                logger.debug("No cost info for %s, skipping", model_name)
                continue

            # Check token limits
            limits = await get_model_limits(model_name)
            if limits and token_estimate > limits.total_tokens:
                logger.debug(
                    "Model %s token limit exceeded: %d > %d",
                    model_name,
                    token_estimate,
                    limits.total_tokens,
                )
                continue

            # Estimate total cost
            total_cost = estimate_request_cost(costs, token_estimate)
            if total_cost <= self.max_cost:
                model_estimates.append((self._initialized_models[model_name], total_cost))
                logger.debug(
                    "Model %s estimated cost: $%s",
                    model_name,
                    total_cost,
                )
        if not model_estimates:
            msg = (
                f"No suitable model found within cost limit ${self.max_cost} "
                f"for {token_estimate} tokens"
            )
            raise RuntimeError(msg)

        # Sort by cost
        model_estimates.sort(key=lambda x: x[1])

        if self.strategy == "cheapest_possible":
            selected, cost = model_estimates[0]
        else:  # best_within_budget
            selected, cost = model_estimates[-1]

        logger.info(
            "Selected %s with estimated cost $%s",
            selected.__class__.__name__,  # Use class name instead of .name()
            cost,
        )
        return selected

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> tuple[ModelResponse, Usage]:
        """Process request using cost-optimized model selection."""
        selected_model = await self._select_model(messages)
        return await selected_model.request(messages, model_settings)
