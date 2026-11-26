"""Dynamic model delegation based on prompt analysis."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic import Field, model_validator
from pydantic_ai import ModelRequest, TextPart, UserPromptPart
from pydantic_ai.models import Model, ModelRequestParameters
from tokonomics import ModelName  # noqa: TC002

from llmling_models.log import get_logger
from llmling_models.models.helpers import infer_model
from llmling_models.models.multi import MultiModel


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from pydantic_ai import ModelMessage, ModelResponse, ModelSettings, RunContext
    from pydantic_ai.models import StreamedResponse
    from pydantic_ai.usage import RequestUsage

logger = get_logger(__name__)


class DelegationMultiModel(MultiModel):
    """Meta-model that dynamically selects models based on a user prompt."""

    selector_model: ModelName | str | Model
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

    def _models_by_id(self) -> dict[str, Model]:
        """Map the configured ids ("openai:gpt-4o-mini") to the model instances."""
        models: dict[str, Model] = {}
        for spec, model in zip(self.models, self.available_models, strict=True):
            models.setdefault(spec if isinstance(spec, str) else model.model_name, model)
        return models

    def _format_selection_text(self, base_prompt: str) -> str:
        """Format selection text using prompt and optional model descriptions."""
        if not self.model_descriptions:
            return base_prompt

        model_hints = "\n".join(
            f"Pick '{model if isinstance(model, str) else model.model_name}' for: {desc}"
            for model, desc in self.model_descriptions.items()
        )
        return f"{model_hints}\n\n{base_prompt}"

    async def _select_model(
        self, prompt: str, model_settings: ModelSettings | None
    ) -> tuple[Model, RequestUsage]:
        """Use selector model to choose appropriate model for prompt.

        Returns the chosen model and the selector's usage.
        """
        selector = (
            infer_model(self.selector_model)
            if isinstance(self.selector_model, str)
            else self.selector_model
        )
        models = self._models_by_id()
        selection_text = (
            f"{self._format_selection_text(self.selection_prompt)}\n\n"
            f"Available models: {', '.join(models)}\n\n"
            f"Task: {prompt}\n\n"
            "Return only the name of the model to use."
        )
        selection_msg = ModelRequest(parts=[UserPromptPart(content=selection_text)])
        # Plain parameters: the agent's function and output tools are not for the selector.
        response = await selector.request([selection_msg], None, ModelRequestParameters())
        text = next((p.content for p in response.parts if isinstance(p, TextPart)), "")
        selected_name = text.strip().strip("'\"`").strip()

        model = models.get(selected_name) or next(
            (m for m in models.values() if m.model_name == selected_name), None
        )
        if model is None:
            msg = f"Selector returned unknown model: {selected_name}"
            raise ValueError(msg)
        logger.debug("Selected model %s for prompt: %s", selected_name, prompt)
        return model, response.usage

    def _get_last_prompt(self, messages: list[ModelMessage]) -> str:
        """Extract the last user prompt from messages.

        After a tool call the last request holds only tool returns: the prompt is the
        one of an earlier request.
        """
        if not messages:
            msg = "No messages provided"
            raise ValueError(msg)

        for message in reversed(messages):
            if not isinstance(message, ModelRequest):
                continue
            for part in message.parts:
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

        selected_model, selector_usage = await self._select_model(prompt, model_settings)
        response = await selected_model.request(
            messages,
            model_settings,
            model_request_parameters,
        )
        response.usage = response.usage + selector_usage
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        """Stream response using dynamically selected model."""
        # Extract the actual prompt
        prompt = self._get_last_prompt(messages)

        selected_model, _ = await self._select_model(prompt, model_settings)

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

    async def test() -> None:
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
