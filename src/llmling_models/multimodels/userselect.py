"""Model that lets users interactively select from multiple models."""

from __future__ import annotations

from collections.abc import Awaitable
import inspect
from typing import TYPE_CHECKING, Literal

from pydantic import Field, ImportString
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse

from llmling_models.log import get_logger
from llmling_models.multi import MultiModel


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.result import Usage
    from pydantic_ai.settings import ModelSettings

    from llmling_models.input_handlers import InputHandler

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
    _model_name: str = "user-select"

    prompt_template: str = Field(default="🤖 Choose a model for: {prompt}")
    """Template for showing the prompt to the user."""

    show_system: bool = Field(default=True)
    """Whether to show system messages."""

    input_prompt: str = Field(default="Enter model number (0-{max}): ")
    """Prompt shown when requesting model selection."""

    handler: ImportString = Field(
        default="llmling_models:DefaultInputHandler", validate_default=True
    )
    """Input handler class to use."""

    async def _get_user_selection(
        self,
        messages: list[ModelMessage],
        handler: InputHandler,
    ) -> Model:
        """Get model selection from user."""
        # Format the model list
        model_list = "\n".join(
            f"[{i}] {model.model_name}" for i, model in enumerate(self.available_models)
        )

        # Format and display messages
        display_text = handler.format_messages(
            messages,
            prompt_template=self.prompt_template,
            show_system=self.show_system,
        )

        print("\n" + "=" * 80)
        print(display_text)
        print("\nAvailable models:\n" + model_list)
        print("-" * 80)

        while True:
            # Get user input
            selection_prompt = self.input_prompt.format(
                max=len(self.available_models) - 1
            )
            input_method = handler.get_input
            if inspect.iscoroutinefunction(input_method):
                selection = await input_method(selection_prompt)
            else:
                response_or_awaitable = input_method(selection_prompt)
                if isinstance(response_or_awaitable, Awaitable):
                    selection = await response_or_awaitable
                else:
                    selection = response_or_awaitable

            # Parse selection
            try:
                index = int(selection)
                if 0 <= index < len(self.available_models):
                    selected_model = self.available_models[index]
                    logger.info(
                        "User selected model: %s",
                        selected_model.model_name,
                    )
                    return selected_model

            except ValueError:
                pass

            print(
                f"Invalid selection. Please enter number between 0 and "
                f"{len(self.available_models) - 1}"
            )

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, Usage]:
        """Process request using user-selected model."""
        # Initialize handler
        handler = self.handler() if isinstance(self.handler, type) else self.handler

        # Let user select model
        selected_model = await self._get_user_selection(messages, handler)

        # Use selected model
        return await selected_model.request(
            messages,
            model_settings,
            model_request_parameters,
        )

    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        """Stream response using user-selected model."""
        # Initialize handler
        handler = self.handler() if isinstance(self.handler, type) else self.handler

        # Let user select model
        selected_model = await self._get_user_selection(messages, handler)

        # Stream from selected model
        async with selected_model.request_stream(
            messages,
            model_settings,
            model_request_parameters,
        ) as stream:
            yield stream


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
