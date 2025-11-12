"""Models with pre/post prompt processing."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict
from pydantic_ai import ModelRequest, RequestUsage, UserPromptPart
from pydantic_ai.models import Model
from schemez import Schema

from llmling_models.log import get_logger
from llmling_models.utils import infer_model


logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai import ModelMessage, ModelResponse, ModelSettings, RunContext
    from pydantic_ai.models import (
        KnownModelName,
        ModelRequestParameters,
        StreamedResponse,
    )


class PrePostPromptConfig(Schema):
    """Configuration for pre/post prompts."""

    text: str
    model: str | Model

    @property
    def model_instance(self) -> Model:
        """Get model instance."""
        if isinstance(self.model, str):
            return infer_model(self.model)
        return self.model

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AugmentedModel(Model):
    """Model with pre/post prompt processing."""

    def __init__(
        self,
        main_model: KnownModelName | Model,
        pre_prompt: PrePostPromptConfig | None = None,
        post_prompt: PrePostPromptConfig | None = None,
    ) -> None:
        """Initialize the augmented model.

        Args:
            main_model: The main model to use for the augmented model.
            pre_prompt: The pre-prompt configuration for the augmented model.
            post_prompt: The post-prompt configuration for the augmented model.
        """
        self.main_model = main_model
        self.pre_prompt = pre_prompt
        self.post_prompt = post_prompt
        self._initialized_models: dict[str, Model] = {}
        self._main_model = infer_model(self.main_model)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return (
            self.main_model
            if isinstance(self.main_model, str)
            else self.main_model.model_name
        )

    @property
    def system(self) -> str:
        """Return the system/provider name."""
        return "augmented"

    def _get_model(self, key: str) -> Model:
        """Get or initialize a model."""
        if key in self._initialized_models:
            return self._initialized_models[key]

        match key:
            case "main":
                model = self._main_model
            case "pre" if self.pre_prompt:
                model = self.pre_prompt.model_instance
            case "post" if self.post_prompt:
                model = self.post_prompt.model_instance
            case _:
                msg = f"Unknown model key: {key}"
                raise ValueError(msg)

        self._initialized_models[key] = model
        return model

    def _get_last_content(self, messages: list[ModelMessage]) -> str:
        """Extract content from last message."""
        if not messages:
            return ""

        last_msg = messages[-1]
        if isinstance(last_msg, ModelRequest):
            for part in reversed(last_msg.parts):
                if isinstance(part, UserPromptPart):
                    return str(part.content)  # TODO: could also be media content
        return ""

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Process request with pre/post prompting."""
        total_cost = RequestUsage()
        all_messages = messages.copy()

        # Pre-process the question if configured
        if self.pre_prompt:
            pre_model = self._get_model("pre")
            input_question = self._get_last_content(messages)
            pre_prompt = self.pre_prompt.text.format(input=input_question)

            # Get expanded question
            pre_request = ModelRequest(parts=[UserPromptPart(content=pre_prompt)])
            pre_response = await pre_model.request(
                [pre_request],
                model_settings,
                model_request_parameters,
            )
            total_cost += pre_response.usage

            # Replace original question with expanded version
            expanded_question = str(pre_response.parts[0].content)  # type: ignore
            logger.debug("Original question: %s", input_question)
            logger.debug("Expanded question: %s", expanded_question)
            expanded_part = UserPromptPart(content=expanded_question)
            all_messages[-1] = ModelRequest(parts=[expanded_part])

        # Process with main model
        main_model = self._get_model("main")
        main_response = await main_model.request(
            all_messages,
            model_settings,
            model_request_parameters,
        )
        total_cost += main_response.usage
        logger.debug("Main response: %s", str(main_response.parts[0].content))  # type: ignore

        # Post-process if configured
        if self.post_prompt:
            post_model = self._get_model("post")
            post_prompt = self.post_prompt.text.format(
                output=str(main_response.parts[0].content)  # type: ignore
            )

            # Create post-processing request
            post_request = ModelRequest(parts=[UserPromptPart(content=post_prompt)])
            post_response = await post_model.request(
                [post_request],
                model_settings,
                model_request_parameters,
            )
            total_cost += post_response.usage
            logger.debug(
                "Post-processed response: %s",
                str(post_response.parts[0].content),  # type: ignore
            )

            # Add post-prompt messages to the chain
            all_messages.extend([main_response, post_request, post_response])
            post_response.usage = total_cost
            return post_response

        # If no post-processing, add main response to message chain
        all_messages.append(main_response)
        main_response.usage = total_cost
        return main_response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Stream response with pre/post processing."""
        all_messages = messages.copy()

        # Pre-process if configured
        if self.pre_prompt:
            pre_model = self._get_model("pre")
            input_question = self._get_last_content(messages)
            pre_prompt = self.pre_prompt.text.format(input=input_question)

            # Get expanded question
            pre_request = ModelRequest(parts=[UserPromptPart(content=pre_prompt)])
            pre_response = await pre_model.request(
                [pre_request],
                model_settings,
                model_request_parameters,
            )

            # Replace original question
            expanded_question = str(pre_response.parts[0].content)  # type: ignore
            expanded_part = UserPromptPart(content=expanded_question)
            all_messages[-1] = ModelRequest(parts=[expanded_part])

        # Stream from main model
        main_model = self._get_model("main")
        async with main_model.request_stream(
            all_messages,
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

    async def test():
        pre = PrePostPromptConfig(
            text=(
                "Your task is to rewrite '{input}' as a more detailed "
                "philosophical question. Do not answer it. Only return the expanded "
                "question."
            ),
            model="openai:gpt-4o-mini",
        )
        augmented = AugmentedModel("openai:gpt-5-nano", pre_prompt=pre)
        agent: Agent[None, str] = Agent(model=augmented)
        print("\nTesting Pre-Prompt Expansion Pipeline")
        print("=" * 60)
        question = "What is the meaning of life?"
        print(f"Original Question: {question}")
        result = await agent.run(question)
        print(result)

    asyncio.run(test())
