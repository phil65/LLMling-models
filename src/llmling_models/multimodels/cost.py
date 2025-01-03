"""Cost-optimized model selection."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Literal, TypeVar

from pydantic import Field
from pydantic_ai.models import AgentModel, Model

from llmling_models.log import get_logger
from llmling_models.multi import MultiModel
from llmling_models.utils import (
    estimate_request_cost,
    estimate_tokens,
    get_model_costs,
    get_model_limits,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.result import Usage
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition

logger = get_logger(__name__)
TModel = TypeVar("TModel", bound=Model)


class CostOptimizedMultiModel[TModel: Model](MultiModel[TModel]):
    """Multi-model that selects based on cost and token limits."""

    type: Literal["cost-optimized"] = Field(default="cost-optimized", init=False)

    max_input_cost: float = Field(gt=0)
    """Maximum allowed cost in USD per request"""

    strategy: Literal["cheapest_possible", "best_within_budget"] = "best_within_budget"
    """Strategy for model selection."""

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
            max_input_cost=Decimal(str(self.max_input_cost)),
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
        max_input_cost: Decimal,
        strategy: str,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ):
        """Initialize with models and cost settings."""
        if not models:
            msg = "At least one model must be provided"
            raise ValueError(msg)
        self.models = models
        self.max_input_cost = max_input_cost
        self.strategy = strategy
        self.function_tools = function_tools
        self.allow_text_result = allow_text_result
        self.result_tools = result_tools
        self._initialized_models: dict[str, AgentModel] = {}

    async def _select_model(
        self,
        messages: list[ModelMessage],
    ) -> AgentModel:
        """Select appropriate model based on input token costs."""
        token_estimate = estimate_tokens(messages)
        logger.debug("Estimated input tokens: %d", token_estimate)

        # Get cost estimates and check limits for each model
        model_options: list[tuple[AgentModel, Decimal]] = []
        for model in self.models:
            model_name = model.name()
            logger.debug("Checking model: %s", model_name)

            # Initialize model if needed
            if model_name not in self._initialized_models:
                self._initialized_models[model_name] = await model.agent_model(
                    function_tools=self.function_tools,
                    allow_text_result=self.allow_text_result,
                    result_tools=self.result_tools,
                )

            # Check token limits first
            limits = await get_model_limits(model_name)
            if not limits:
                logger.debug("No token limits for %s, skipping", model_name)
                continue

            if token_estimate > limits.input_tokens:
                logger.debug(
                    "Token limit exceeded for %s: %d > %d",
                    model_name,
                    token_estimate,
                    limits.input_tokens,
                )
                continue

            # Check costs
            costs = await get_model_costs(model_name)
            if not costs:
                logger.debug("No cost info for %s, skipping", model_name)
                continue

            # Calculate total estimated cost
            estimated_cost = estimate_request_cost(costs, token_estimate)
            logger.debug(
                "Estimated cost for %s: $%s (max: $%s)",
                model_name,
                estimated_cost,
                self.max_input_cost,
            )

            if estimated_cost <= self.max_input_cost:
                model_options.append((
                    self._initialized_models[model_name],
                    estimated_cost,
                ))
                logger.debug("Added model %s to options", model_name)

        if not model_options:
            msg = (
                f"No suitable model found within input cost limit ${self.max_input_cost} "
                f"for {token_estimate} tokens"
            )
            raise RuntimeError(msg)

        # Sort by cost and select based on strategy
        model_options.sort(key=lambda x: x[1])
        if self.strategy == "cheapest_possible":
            selected, cost = model_options[0]
        else:  # best_within_budget
            selected, cost = model_options[-1]
        msg = "Selected %s with estimated cost $%s"
        logger.info(msg, selected.__class__.__name__, cost)
        return selected

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> tuple[ModelResponse, Usage]:
        """Process request using cost-optimized model selection."""
        selected_model = await self._select_model(messages)
        return await selected_model.request(messages, model_settings)
