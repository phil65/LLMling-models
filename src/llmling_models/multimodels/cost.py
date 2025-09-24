"""Cost-optimized model selection."""

from __future__ import annotations

from contextlib import asynccontextmanager
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field
from pydantic_ai.models import Model

from llmling_models.log import get_logger
from llmling_models.multi import MultiModel
from llmling_models.utils import (
    estimate_request_cost,
    estimate_tokens,
    get_model_costs,
    get_model_limits,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai import RunContext
    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models import ModelRequestParameters, StreamedResponse
    from pydantic_ai.settings import ModelSettings

logger = get_logger(__name__)


class CostOptimizedMultiModel[TModel: Model](MultiModel[TModel]):
    """Multi-model that selects based on cost and token limits."""

    max_input_cost: float = Field(gt=0)
    """Maximum allowed cost in USD per request"""

    strategy: Literal["cheapest_possible", "best_within_budget"] = "best_within_budget"
    """Strategy for model selection."""

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return "cost_optimized"

    @property
    def system(self) -> str:
        """Return the system/provider name."""
        return "multi"

    async def _select_model(
        self,
        messages: list[ModelMessage],
    ) -> Model:
        """Select appropriate model based on input token costs."""
        token_estimate = estimate_tokens(messages)
        logger.debug("Estimated input tokens: %d", token_estimate)

        # Get cost estimates and check limits for each model
        model_options: list[tuple[Model, Decimal]] = []
        max_input_cost = Decimal(str(self.max_input_cost))

        for model in self.available_models:
            model_name = model.model_name
            logger.debug("Checking model: %s", model_name)

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
                max_input_cost,
            )

            if estimated_cost <= max_input_cost:
                model_options.append((model, estimated_cost))
                logger.debug("Added model %s to options", model_name)

        if not model_options:
            msg = (
                f"No suitable model found within input cost limit ${max_input_cost} "
                f"for {token_estimate} tokens"
            )
            raise RuntimeError(msg)

        # Sort by cost and select based on strategy
        model_options.sort(key=lambda x: x[1])
        if self.strategy == "cheapest_possible":
            selected, cost = model_options[0]
        else:  # best_within_budget
            selected, cost = model_options[-1]

        logger.info(
            "Selected %s with estimated cost $%s",
            selected.__class__.__name__,
            cost,
        )
        return selected

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Process request using cost-optimized model selection."""
        selected_model = await self._select_model(messages)
        return await selected_model.request(
            messages,
            model_settings,
            model_request_parameters,
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Stream response using cost-optimized model selection."""
        selected_model = await self._select_model(messages)
        async with selected_model.request_stream(
            messages,
            model_settings,
            model_request_parameters,
            run_context,
        ) as stream:
            yield stream
