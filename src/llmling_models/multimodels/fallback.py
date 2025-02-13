"""Multi-model implementations."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import Field
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse

from llmling_models.log import get_logger
from llmling_models.multi import MultiModel


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.usage import Usage

logger = get_logger(__name__)
TModel = TypeVar("TModel", bound=Model)


class FallbackMultiModel[TModel: Model](MultiModel[TModel]):
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
    _model_name: str = "fallback"

    def model_post_init(self, __context: dict[str, Any], /) -> None:
        """Initialize model name."""
        self._model_name = f"multi-fallback({len(self.models)})"

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """Return the system/provider name."""
        return "multi"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, Usage]:
        """Try each model in sequence until one succeeds."""
        last_error = None

        for model in self.available_models:
            try:
                logger.debug("Trying model: %s", model.model_name)
                return await model.request(
                    messages,
                    model_settings,
                    model_request_parameters,
                )
            except Exception as e:  # noqa: BLE001
                last_error = e
                logger.debug("Model %s failed: %s", model.model_name, e)
                continue

        msg = f"All models failed. Last error: {last_error}"
        raise RuntimeError(msg) from last_error

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        """Try streaming from each model until one succeeds."""
        last_error = None

        for model in self.available_models:
            try:
                logger.debug("Trying model stream: %s", model.model_name)
                async with model.request_stream(
                    messages,
                    model_settings,
                    model_request_parameters,
                ) as stream:
                    yield stream
                    return  # Exit after first successful stream
            except Exception as e:  # noqa: BLE001
                last_error = e
                logger.debug("Model %s stream failed: %s", model.model_name, e)
                continue

        msg = f"All models failed streaming. Last error: {last_error}"
        raise RuntimeError(msg) from last_error
