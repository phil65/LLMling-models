"""Model that delegates responses to human input."""

from __future__ import annotations

from collections.abc import Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
import inspect
from typing import TYPE_CHECKING, Any

from pydantic import Field, ImportString  # noqa: TC002
from pydantic_ai import ModelResponse, RequestUsage, TextPart
from pydantic_ai.models import ModelRequestParameters, StreamedResponse

from llmling_models.base import PydanticModel
from llmling_models.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai import (
        ModelMessage,
        ModelResponseStreamEvent,
        RunContext,
    )
    from pydantic_ai.settings import ModelSettings


logger = get_logger(__name__)


@dataclass(kw_only=True)
class InputStreamedResponse(StreamedResponse):
    """Stream implementation for input model."""

    stream: AsyncIterator[str]
    _timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        """Initialize usage tracking."""
        self._usage = RequestUsage()

    @property
    def provider_name(self) -> str | None:
        """Get the provider name."""
        return "input"

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream characters as events."""
        try:
            while True:
                try:
                    char = await self.stream.__anext__()
                    # Emit text delta event for each character
                    event = self._parts_manager.handle_text_delta(
                        vendor_part_id="content",
                        content=char,
                    )
                    if event is not None:
                        yield event
                except StopAsyncIteration:
                    break

        except Exception as e:
            msg = f"Stream error: {e}"
            raise RuntimeError(msg) from e

    @property
    def timestamp(self) -> datetime:
        """Get response timestamp."""
        return self._timestamp

    @property
    def model_name(self) -> str:
        """Get response model_name."""
        return "input"


class InputModel(PydanticModel):
    """Model that delegates responses to human input."""

    prompt_template: str = Field(default="👤 Please respond to: {prompt}")
    """Template for showing the prompt to the human."""

    show_system: bool = Field(default=True)
    """Whether to show system messages to the human."""

    input_prompt: str = Field(default="Your response: ")
    """Prompt to show when requesting input."""

    handler: ImportString = Field(
        default="llmling_models:DefaultInputHandler", validate_default=True
    )
    """Input handler class to use."""

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return "input"

    @property
    def system(self) -> str:
        """Return the system/provider name."""
        return "input"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Get response from human input."""
        # Format and display messages using handler
        handler = self.handler() if isinstance(self.handler, type) else self.handler
        display_text = handler.format_messages(
            messages,
            prompt_template=self.prompt_template,
            show_system=self.show_system,
        )
        print("\n" + "=" * 80)
        print(display_text)
        print("-" * 80)

        # Get input using configured handler
        input_method = handler.get_input
        if inspect.iscoroutinefunction(input_method):
            response = await input_method(self.input_prompt)
        else:
            response_or_awaitable = input_method(self.input_prompt)
            if isinstance(response_or_awaitable, Awaitable):
                response = await response_or_awaitable
            else:
                response = response_or_awaitable

        return ModelResponse(parts=[TextPart(response)], timestamp=datetime.now(UTC))

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Stream responses character by character."""
        # Format and display messages using handler
        handler = self.handler() if isinstance(self.handler, type) else self.handler
        display_text = handler.format_messages(
            messages,
            prompt_template=self.prompt_template,
            show_system=self.show_system,
        )
        print("\n" + "=" * 80)
        print(display_text)
        print("-" * 80)

        # Get streaming input using configured handler
        stream_method = handler.stream_input
        if inspect.iscoroutinefunction(stream_method):
            char_stream = await stream_method(self.input_prompt)
        else:
            stream_or_awaitable = stream_method(self.input_prompt)
            if isinstance(stream_or_awaitable, Awaitable):
                char_stream = await stream_or_awaitable
            else:
                char_stream = stream_or_awaitable

        yield InputStreamedResponse(
            model_request_parameters=ModelRequestParameters(), stream=char_stream
        )


if __name__ == "__main__":
    import asyncio
    import logging

    from pydantic_ai import Agent

    # Set up logging
    logging.basicConfig(level=logging.DEBUG)

    async def test():
        # Test basic input
        model = InputModel(
            prompt_template="🤖 Question: {prompt}",
            show_system=True,
            input_prompt="Your answer: ",
        )
        agent: Agent[None, str] = Agent(
            model=model,
            system_prompt="You are helping test user input.",
        )

        # Test regular input
        print("\nTesting regular input:")
        result = await agent.run("What's your favorite color?")
        print(f"\nYour response was: {result.output}")

        # Test streaming input
        print("\nTesting streaming input:")
        async with agent.run_stream("Tell me a story") as stream:
            async for chunk in stream.stream_text(delta=True):
                print(chunk, end="", flush=True)
        print("\nStreaming complete!")

    asyncio.run(test())
