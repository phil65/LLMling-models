"""Dynamic model delegation based on prompt analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, model_validator
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import AgentModel, Model, infer_model
from typing_extensions import TypeVar

from llmling_models.log import get_logger
from llmling_models.multi import MultiModel


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.result import Usage
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition

logger = get_logger(__name__)
TModel = TypeVar("TModel", bound=Model)


class DelegationMultiModel(MultiModel[TModel]):
    """Meta-model that dynamically selects models based on a user prompt.

    Example YAML configuration:
        ```yaml
        model:
          type: delegation
          selector_model: openai:gpt-4-turbo
          models:
            - openai:gpt-4
            - openai:gpt-3.5-turbo
          selection_prompt: |
            Pick gpt-4 for complex tasks, gpt-3.5-turbo for simple queries.
        ```
    """

    type: Literal["delegation"] = Field(default="delegation", init=False)
    selector_model: str | Model
    selection_prompt: str = Field(
        description="Instructions for model selection based on task type"
    )
    model_descriptions: dict[str | Model, str] | None = Field(default=None, exclude=True)

    def name(self) -> str:
        """Get descriptive model name."""
        return f"delegation({len(self.models)})"

    @model_validator(mode="before")
    @classmethod
    def handle_model_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Handle both list of models and dict[model, description]."""
        if isinstance(data.get("models"), dict):
            data["_model_descriptions"] = data["models"]
            data["models"] = list(data["models"].keys())
        return data

    def _format_selection_text(self, base_prompt: str) -> str:
        """Format selection text using prompt and optional model descriptions."""
        if not self.model_descriptions:
            return base_prompt

        model_hints = "\n".join(
            f"Pick '{model}' for: {desc}"
            for model, desc in self.model_descriptions.items()
        )
        return f"{model_hints}\n\n{base_prompt}"

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> DelegationAgentModel:
        """Create agent model that implements selection strategy."""
        selector = (
            infer_model(self.selector_model)  # type: ignore
            if isinstance(self.selector_model, str)
            else self.selector_model
        )

        return DelegationAgentModel[TModel](
            selector_model=selector,
            choice_models=self.available_models,
            selection_prompt=self._format_selection_text(self.selection_prompt),
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )


class DelegationAgentModel[TModel: Model](AgentModel):
    """AgentModel that implements dynamic model selection."""

    def __init__(
        self,
        selector_model: Model,
        choice_models: list[TModel],
        selection_prompt: str,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> None:
        """Initialize with models and selection configuration."""
        if not choice_models:
            msg = "At least one choice model must be provided"
            raise ValueError(msg)

        self.selector_model = selector_model
        self.choice_models = choice_models
        self.selection_prompt = selection_prompt
        self.function_tools = function_tools
        self.allow_text_result = allow_text_result
        self.result_tools = result_tools

        # Will be initialized on first use
        self._initialized_selector: AgentModel | None = None
        self._initialized_choices: dict[str, AgentModel] = {}

    async def _initialize_models(self) -> None:
        """Initialize selector and choice models if needed."""
        if self._initialized_selector is None:
            # Initialize selector with basic config (no tools needed)
            self._initialized_selector = await self.selector_model.agent_model(
                function_tools=[],
                allow_text_result=True,
                result_tools=[],
            )

        # Initialize choice models as needed
        for model in self.choice_models:
            model_name = model.name()
            if model_name not in self._initialized_choices:
                self._initialized_choices[model_name] = await model.agent_model(
                    function_tools=self.function_tools,
                    allow_text_result=self.allow_text_result,
                    result_tools=self.result_tools,
                )

    async def _select_model(
        self,
        prompt: str,
        model_settings: ModelSettings | None = None,
    ) -> AgentModel:
        """Use selector model to choose appropriate model for prompt."""
        await self._initialize_models()
        assert self._initialized_selector is not None

        # Create selection request
        selection_text = (
            f"{self.selection_prompt}\n\n"
            f"Task: {prompt}\n\n"
            "Return only the name of the model to use."
        )
        part = UserPromptPart(content=selection_text)
        selection_msg = ModelRequest(parts=[part])

        response, _ = await self._initialized_selector.request(
            [selection_msg],
            model_settings,
        )
        selected_name = str(response.parts[0].content).strip()  # type: ignore

        # Find matching model
        for model in self.choice_models:
            if model.name() == selected_name:
                model_name = model.name()
                logger.debug("Selected model %s for prompt: %s", model_name, prompt)
                return self._initialized_choices[model_name]

        msg = f"Selector returned unknown model: {selected_name}"
        raise ValueError(msg)

    async def request(
        self,
        messages: Sequence[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> tuple[ModelResponse, Usage]:
        """Process request using dynamically selected model."""
        # Extract the actual prompt from messages
        if not messages:
            msg = "No messages provided"
            raise ValueError(msg)

        last_message = messages[-1]
        if not isinstance(last_message, ModelRequest):
            msg = "Last message must be a request"
            raise ValueError(msg)  # noqa: TRY004

        prompt = ""
        for part in last_message.parts:
            if isinstance(part, UserPromptPart):
                prompt = part.content
                break

        if not prompt:
            msg = "No prompt found in messages"
            raise ValueError(msg)

        # Select and use appropriate model
        selected_model = await self._select_model(prompt, model_settings)
        return await selected_model.request(list(messages), model_settings)


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
        delegation_model: DelegationMultiModel[Any] = DelegationMultiModel(
            selector_model="openai:gpt-4o-mini",
            models=["openai:gpt-4o-mini", "openai:gpt-3.5-turbo"],
            selection_prompt=PROMPT,
        )

        agent: Agent[None, str] = Agent(delegation_model)
        result = await agent.run("Find the highest prime number known to mankind")
        print(result.data)

    asyncio.run(test())
