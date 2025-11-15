from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field
from pydantic_ai.models import Model

from llmling_models.models.helpers import infer_model


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai import ModelMessage, ModelResponse, ModelSettings, RunContext
    from pydantic_ai.models import ModelRequestParameters, StreamedResponse


@dataclass
class StringModel(Model):
    """Wrapper for string model names."""

    identifier: str
    type: Literal["string"] = Field(default="string", init=False)
    _model_name: str = "string"

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.identifier

    @property
    def system(self) -> str:
        """Return the model name."""
        return "string"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Create and delegate to inferred model."""
        model = infer_model(self.identifier)  # type: ignore
        return await model.request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Stream from inferred model."""
        model = infer_model(self.identifier)  # type: ignore
        async with model.request_stream(
            messages,
            model_settings,
            model_request_parameters,
            run_context,
        ) as stream:
            yield stream
