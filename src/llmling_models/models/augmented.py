"""Models with pre/post prompt processing."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict
from pydantic_ai import ModelRequest, RequestUsage, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models import Model, ModelRequestParameters
from schemez import Schema

from llmling_models.log import get_logger
from llmling_models.models.helpers import infer_model


logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from pydantic_ai import ModelMessage, ModelResponse, ModelSettings, RunContext
    from pydantic_ai.models import KnownModelName, StreamedResponse


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
    """Model with pre/post prompt processing.

    The pre-prompt rewrites the last user prompt before the main model sees it; the
    post-prompt rewrites the main model's final answer (not responses with tool calls).

    When streaming (`request_stream`, e.g. `agent.run_stream()` / `agent.iter()`), only the
    pre-prompt is applied: the main model's stream is passed through as it arrives, so the
    post-prompt is not applied. Use `request` (`agent.run()`) if you need the post-prompt.
    """

    def __init__(
        self,
        main_model: KnownModelName | str | Model,
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
        return self.main_model if isinstance(self.main_model, str) else self.main_model.model_name

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
                raise ValueError(f"Unknown model key: {key}")

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

    async def _expand_prompt(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> tuple[list[ModelMessage], RequestUsage]:
        """Run the pre-prompt and put the expanded question into the last request.

        Only the user prompt part is replaced: system prompts and instructions of the
        request stay. A request without a user prompt (tool returns) is left alone.
        """
        last = messages[-1] if messages else None
        if not self.pre_prompt or not isinstance(last, ModelRequest):
            return messages, RequestUsage()
        if not any(isinstance(part, UserPromptPart) for part in last.parts):
            return messages, RequestUsage()

        input_question = self._get_last_content(messages)
        pre_request = ModelRequest(
            parts=[UserPromptPart(content=self.pre_prompt.text.format(input=input_question))]
        )
        pre_response = await self._get_model("pre").request(
            [pre_request], model_settings, ModelRequestParameters()
        )
        expanded_question: str | None = _text_of(pre_response)
        logger.debug("Original question: %s", input_question)
        logger.debug("Expanded question: %s", expanded_question)
        # The expanded question takes the place of the (first) user prompt part.
        new_parts = []
        for part in last.parts:
            if not isinstance(part, UserPromptPart):
                new_parts.append(part)
            elif expanded_question is not None:
                new_parts.append(UserPromptPart(content=expanded_question))
                expanded_question = None
        expanded = [*messages[:-1], replace(last, parts=new_parts)]
        return expanded, pre_response.usage

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Process request with pre/post prompting."""
        all_messages, total_cost = await self._expand_prompt(messages, model_settings)

        # Process with main model
        main_response = await self._get_model("main").request(
            all_messages,
            model_settings,
            model_request_parameters,
        )
        total_cost += main_response.usage

        # Post-process the final answer only: a response with tool calls goes back to
        # the agent unchanged.
        has_tool_calls = any(isinstance(p, ToolCallPart) for p in main_response.parts)
        if self.post_prompt and not has_tool_calls:
            post_prompt = self.post_prompt.text.format(output=_text_of(main_response))
            post_request = ModelRequest(parts=[UserPromptPart(content=post_prompt)])
            post_response = await self._get_model("post").request(
                [post_request], model_settings, ModelRequestParameters()
            )
            total_cost += post_response.usage
            logger.debug("Post-processed response: %s", _text_of(post_response))
            post_response.usage = total_cost
            return post_response

        main_response.usage = total_cost
        return main_response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        """Stream response with pre-processing (the post-prompt is not applied)."""
        all_messages, _ = await self._expand_prompt(messages, model_settings)

        # Stream from main model
        main_model = self._get_model("main")
        async with main_model.request_stream(
            all_messages,
            model_settings,
            model_request_parameters,
            run_context,
        ) as stream:
            yield stream


def _text_of(response: ModelResponse) -> str:
    return "".join(part.content for part in response.parts if isinstance(part, TextPart))


if __name__ == "__main__":
    import asyncio
    import logging

    from pydantic_ai import Agent

    logging.basicConfig(level=logging.DEBUG)

    async def test() -> None:
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
