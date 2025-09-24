"""Token-limit optimized model selection."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal, TypeVar

from pydantic import Field
from pydantic_ai.models import Model

from llmling_models.log import get_logger
from llmling_models.multi import MultiModel
from llmling_models.utils import estimate_tokens, get_model_limits


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Any

    from pydantic_ai import RunContext
    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models import ModelRequestParameters, StreamedResponse
    from pydantic_ai.settings import ModelSettings

logger = get_logger(__name__)
TModel = TypeVar("TModel", bound=Model)


class TokenOptimizedMultiModel[TModel: Model](MultiModel[TModel]):
    """Multi-model that selects based on input token count."""

    strategy: Literal["efficient", "maximum_context"] = Field(default="efficient")
    """Model selection strategy."""

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return "token-optimized"

    @property
    def system(self) -> str:
        """Return the system/provider name."""
        return "token-optimized"

    async def _select_model(
        self,
        messages: list[ModelMessage],
    ) -> Model:
        """Select appropriate model based on token counts."""
        token_estimate = estimate_tokens(messages)
        logger.debug("Estimated token count: %d", token_estimate)
        model_capabilities = {
            "gpt-3.5-turbo": 1,  # Base model
            "gpt-3.5-turbo-16k": 2,  # Same but larger context
            "gpt-4-turbo": 3,  # More capable and largest context
        }

        model_options: list[tuple[Model, int, int]] = []  # (model, capability, limit)

        for model in self.available_models:
            model_name = model.model_name
            # Check token limits
            limits = await get_model_limits(model_name)
            if not limits:
                logger.debug("No token limits for %s, skipping", model_name)
                continue

            if token_estimate <= limits.input_tokens:
                capability = model_capabilities.get(model_name, 0)
                model_estimates = (model, capability, limits.input_tokens)
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

        model_options.sort(key=lambda x: (x[1], x[2]))
        if self.strategy == "efficient":
            selected, capability, limit = model_options[0]
        else:  # maximum_context
            selected, capability, limit = model_options[-1]
        msg = "Selected %s (capability %d) with %d token limit"
        logger.info(msg, selected.model_name, capability, limit)
        return selected

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Process request using token-optimized model selection."""
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
        """Stream response using token-optimized model selection."""
        selected_model = await self._select_model(messages)
        async with selected_model.request_stream(
            messages,
            model_settings,
            model_request_parameters,
            run_context,
        ) as stream:
            yield stream
