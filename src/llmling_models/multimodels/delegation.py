"""Dynamic model delegation based on prompt analysis."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic import Field, model_validator
from pydantic_ai import ModelRequest, UserPromptPart
from pydantic_ai.models import Model

from llmling_models.log import get_logger
from llmling_models.multi import MultiModel
from llmling_models.utils import infer_model


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai import ModelMessage, ModelResponse, ModelSettings, RunContext
    from pydantic_ai.models import ModelRequestParameters, StreamedResponse

logger = get_logger(__name__)


class DelegationMultiModel(MultiModel[Model]):
    """Meta-model that dynamically selects models based on a user prompt."""

    selector_model: str | Model
    """Model to use for delegation."""

    selection_prompt: str
    """Instructions for model selection based on task type."""

    model_descriptions: dict[str | Model, str] | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def handle_model_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Handle both list of models and dict[model, description]."""
        if isinstance(data.get("models"), dict):
            data["model_descriptions"] = data["models"]
            data["models"] = list(data["models"].keys())
        return data

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return "delegation"

    @property
    def system(self) -> str:
        """Return the system/provider name."""
        return "delegation"

    def _format_selection_text(self, base_prompt: str) -> str:
        """Format selection text using prompt and optional model descriptions."""
        if not self.model_descriptions:
            return base_prompt

        model_hints = "\n".join(
            f"Pick '{model}' for: {desc}"
            for model, desc in self.model_descriptions.items()
        )
        return f"{model_hints}\n\n{base_prompt}"

    async def _select_model(
        self,
        prompt: str,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> Model:
        """Use selector model to choose appropriate model for prompt."""
        # Initialize selector
        selector = (
            infer_model(self.selector_model)
            if isinstance(self.selector_model, str)
            else self.selector_model
        )

        # Create selection request
        selection_text = (
            f"{self.selection_prompt}\n\n"
            f"Task: {prompt}\n\n"
            "Return only the name of the model to use."
        )
        part = UserPromptPart(content=selection_text)
        selection_msg = ModelRequest(parts=[part])

        response = await selector.request(
            [selection_msg],
            model_settings,
            model_request_parameters,
        )
        selected_name = str(response.parts[0].content).strip()  # type: ignore

        # Find matching model
        for model in self.available_models:
            if model.model_name == selected_name:
                logger.debug("Selected model %s for prompt: %s", model.model_name, prompt)
                return model

        msg = f"Selector returned unknown model: {selected_name}"
        raise ValueError(msg)

    def _get_last_prompt(self, messages: list[ModelMessage]) -> str:
        """Extract the last user prompt from messages."""
        if not messages:
            msg = "No messages provided"
            raise ValueError(msg)

        last_message = messages[-1]
        if not isinstance(last_message, ModelRequest):
            msg = "Last message must be a request"
            raise ValueError(msg)  # noqa: TRY004

        for part in last_message.parts:
            if isinstance(part, UserPromptPart):
                return str(part.content)  # TODO: could also be media content

        msg = "No user prompt found in messages"
        raise ValueError(msg)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Process request using dynamically selected model."""
        # Extract the actual prompt
        prompt = self._get_last_prompt(messages)

        # Select and use appropriate model
        selected_model = await self._select_model(
            prompt,
            model_settings,
            model_request_parameters,
        )
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
        """Stream response using dynamically selected model."""
        # Extract the actual prompt
        prompt = self._get_last_prompt(messages)

        # Select appropriate model
        selected_model = await self._select_model(
            prompt,
            model_settings,
            model_request_parameters,
        )

        # Stream from selected model
        async with selected_model.request_stream(
            messages,
            model_settings,
            model_request_parameters,
            run_context,
        ) as stream:
            yield stream


if __name__ == "__main__":
    import asyncio
    import logging

    from pydantic_ai import Agent

    logging.basicConfig(level=logging.DEBUG)
    PROMPT = (
        "Pick 'openai:gpt-4o-mini' for complex reasoning, math, or coding tasks. "
        "Pick 'openai:gpt-3.5-turbo' for simple queries and chat."
    )

    async def test():
        # Create delegation model
        delegation_model = DelegationMultiModel(
            selector_model="openai:gpt-4o-mini",
            models=["openai:gpt-4o-mini", "openai:gpt-3.5-turbo"],
            selection_prompt=PROMPT,
        )

        agent: Agent[None, str] = Agent(delegation_model)
        result = await agent.run("Find the highest prime number known to mankind")
        print(result.output)

    asyncio.run(test())
