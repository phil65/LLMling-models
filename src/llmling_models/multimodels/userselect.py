"""Model that lets users interactively select from multiple models."""

from __future__ import annotations

from collections.abc import Awaitable
import inspect
from typing import TYPE_CHECKING, Literal

from pydantic import Field, ImportString
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import AgentModel, Model

from llmling_models.input_handlers import DefaultInputHandler, InputHandler
from llmling_models.log import get_logger
from llmling_models.multi import MultiModel


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.result import Usage
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition

logger = get_logger(__name__)


class UserSelectModel(MultiModel[Model]):
    """Model that lets users interactively select from multiple models.

    Example YAML configuration:
        ```yaml
        models:
          interactive:
            type: user-select
            models:
              - openai:gpt-4
              - openai:gpt-3.5-turbo
              - anthropic:claude-3-opus
            prompt_template: "🤖 Choose a model for: {prompt}"
            show_system: true
            input_prompt: "Enter model number (0-{max}): "
            handler: llmling_models.input_handlers:DefaultInputHandler
        ```
    """

    type: Literal["user-select"] = Field(default="user-select", init=False)

    prompt_template: str = Field(default="🤖 Choose a model for: {prompt}")
    """Template for showing the prompt to the user."""

    show_system: bool = Field(default=True)
    """Whether to show system messages."""

    input_prompt: str = Field(default="Enter model number (0-{max}): ")
    """Prompt shown when requesting model selection."""

    handler: ImportString = Field(default=DefaultInputHandler)
    """Input handler class to use."""

    def name(self) -> str:
        """Get descriptive model name."""
        return f"user-select({len(self.models)})"

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create agent model implementation."""
        handler = self.handler()
        return UserSelectAgentModel(
            models=self.available_models,
            prompt_template=self.prompt_template,
            show_system=self.show_system,
            input_prompt=self.input_prompt,
            input_handler=handler,
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )


class UserSelectAgentModel(AgentModel):
    """AgentModel that implements interactive model selection."""

    def __init__(
        self,
        models: Sequence[Model],
        prompt_template: str,
        show_system: bool,
        input_prompt: str,
        input_handler: InputHandler,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ):
        """Initialize with models and input configuration."""
        if not models:
            msg = "At least one model must be provided"
            raise ValueError(msg)

        self.models = models
        self.prompt_template = prompt_template
        self.show_system = show_system
        self.input_prompt = input_prompt
        self.input_handler = input_handler
        self.function_tools = function_tools
        self.allow_text_result = allow_text_result
        self.result_tools = result_tools
        self._initialized_models: dict[str, AgentModel] = {}

    def _get_last_prompt(self, messages: Sequence[ModelMessage]) -> str:
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
                return part.content

        msg = "No user prompt found in messages"
        raise ValueError(msg)

    async def _get_user_selection(
        self, prompt: str, models: Sequence[Model]
    ) -> AgentModel:
        """Get model selection from user."""
        # Format the model list
        model_list = "\n".join(f"[{i}] {model.name()}" for i, model in enumerate(models))
        display = (
            f"{self.prompt_template.format(prompt=prompt)}\n\n"
            f"Available models:\n{model_list}"
        )
        print("\n" + "=" * 80)
        print(display)
        print("-" * 80)

        while True:
            # Get user input
            selection_prompt = self.input_prompt.format(max=len(models) - 1)
            input_method = self.input_handler.get_input
            if inspect.iscoroutinefunction(input_method):
                selection = await input_method(selection_prompt)
            else:
                selection_or_awaitable = input_method(selection_prompt)
                if isinstance(selection_or_awaitable, Awaitable):
                    selection = await selection_or_awaitable
                else:
                    selection = selection_or_awaitable
            # Parse selection
            try:
                index = int(selection)
                if 0 <= index < len(models):
                    model = models[index]
                    model_name = model.name()

                    # Initialize model if needed
                    if model_name not in self._initialized_models:
                        self._initialized_models[model_name] = await model.agent_model(
                            function_tools=self.function_tools,
                            allow_text_result=self.allow_text_result,
                            result_tools=self.result_tools,
                        )
                    return self._initialized_models[model_name]

            except ValueError:
                pass

            print(
                f"Invalid selection. Please enter number between 0 and {len(models) - 1}"
            )

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> tuple[ModelResponse, Usage]:
        """Process request using user-selected model."""
        # Get the prompt and show it to user
        prompt = self._get_last_prompt(messages)

        # Let user select model
        selected_model = await self._get_user_selection(prompt, self.models)
        logger.info("User selected model: %s", selected_model.__class__.__name__)

        # Use selected model
        return await selected_model.request(messages, model_settings)


if __name__ == "__main__":
    import asyncio

    from pydantic_ai import Agent

    async def test():
        model = UserSelectModel(
            models=["openai:gpt-4o-mini", "openai:gpt-3.5-turbo"],
            prompt_template="🤖 Choose a model for: {prompt}",
            show_system=True,
            input_prompt="Enter model number (0-{max}): ",
        )
        prompt = "You are helping test user model selection."
        agent: Agent[None, str] = Agent(model=model, system_prompt=prompt)
        result = await agent.run("What is the meaning of life?")
        print(f"\nSelected model's response: {result.data}")

    asyncio.run(test())
