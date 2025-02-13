from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from pydantic import Field
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import (
    KnownModelName,
    Model,
    ModelRequestParameters,
    StreamedResponse,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import Usage

from llmling_models import PydanticModel, infer_model


AllModels = Literal[
    "delegation",
    "cost_optimized",
    "token_optimized",
    "fallback",
    "input",
    "import",
    "remote_model",
    "remote_input",
    "llm",
    "aisuite",
    "augmented",
    "user_select",
]


class StringModel(PydanticModel):
    """Wrapper for string model names."""

    type: Literal["string"] = Field(default="string", init=False)
    _model_name: str = "string"
    identifier: str

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
    ) -> tuple[ModelResponse, Usage]:
        """Create and delegate to inferred model."""
        model = infer_model(self.identifier)  # type: ignore
        return await model.request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        """Stream from inferred model."""
        model = infer_model(self.identifier)  # type: ignore
        async with model.request_stream(
            messages,
            model_settings,
            model_request_parameters,
        ) as stream:
            yield stream


type ModelInput = str | KnownModelName | Model | PydanticModel
"""Type for internal model handling (after validation)."""
