"""Model that delegates responses to human input."""

from __future__ import annotations

from collections.abc import Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
import inspect
from typing import TYPE_CHECKING, Literal, cast

from pydantic import Field, ImportString
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import AgentModel, StreamedResponse
from pydantic_ai.result import Usage

from llmling_models.base import PydanticModel
from llmling_models.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition

    from llmling_models.input_handlers import InputHandler

logger = get_logger(__name__)


@dataclass
class InputStreamedResponse(StreamedResponse):
    """Stream implementation for input model."""

    _stream: AsyncIterator[str]
    _timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        """Initialize usage tracking."""
        self._usage = Usage()

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream characters as events."""
        try:
            while True:
                try:
                    char = await self._stream.__anext__()
                    # Emit text delta event for each character
                    yield self._parts_manager.handle_text_delta(
                        vendor_part_id="content",
                        content=char,
                    )
                except StopAsyncIteration:
                    break

        except Exception as e:
            msg = f"Stream error: {e}"
            raise RuntimeError(msg) from e

    def timestamp(self) -> datetime:
        """Get response timestamp."""
        return self._timestamp


class InputModel(PydanticModel):
    """Model that delegates responses to human input."""

    type: Literal["input"] = Field(default="input", init=False)

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

    def name(self) -> str:
        """Get model name."""
        return "input"

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create agent model implementation."""
        handler = self.handler() if isinstance(self.handler, type) else self.handler
        return InputAgentModel(
            prompt_template=self.prompt_template,
            show_system=self.show_system,
            input_handler=handler,
            input_prompt=self.input_prompt,
        )


class InputAgentModel(AgentModel):
    """AgentModel implementation that requests human input."""

    def __init__(
        self,
        prompt_template: str,
        show_system: bool,
        input_handler: InputHandler,
        input_prompt: str,
    ):
        """Initialize with configuration."""
        self.prompt_template = prompt_template
        self.show_system = show_system
        self.input_handler = input_handler
        self.input_prompt = input_prompt

    def _format_messages(self, messages: list[ModelMessage]) -> str:
        """Format messages for human display."""
        formatted: list[str] = []

        for message in messages:
            for part in message.parts:
                match part:
                    case SystemPromptPart() if self.show_system:
                        formatted.append(f"🔧 System: {part.content}")
                    case UserPromptPart():
                        formatted.append(self.prompt_template.format(prompt=part.content))
                    case TextPart():
                        formatted.append(f"Assistant: {part.content}")
                    case _:
                        continue

        return "\n\n".join(formatted)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> tuple[ModelResponse, Usage]:
        """Get response from human input."""
        # Format and display messages using handler
        display_text = self.input_handler.format_messages(
            messages,
            prompt_template=self.prompt_template,
            show_system=self.show_system,
        )
        print("\n" + "=" * 80)
        print(display_text)
        print("-" * 80)

        # Get input using configured handler
        input_method = self.input_handler.get_input
        if inspect.iscoroutinefunction(input_method):
            response = await input_method(self.input_prompt)
        else:
            response_or_awaitable = input_method(self.input_prompt)
            if isinstance(response_or_awaitable, Awaitable):
                response = await response_or_awaitable
            else:
                response = response_or_awaitable

        parts = cast(list[ModelResponsePart], [TextPart(response)])
        return ModelResponse(
            parts=parts,
            timestamp=datetime.now(UTC),
        ), Usage()

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Stream responses character by character."""
        # Format and display messages using handler
        display_text = self.input_handler.format_messages(
            messages,
            prompt_template=self.prompt_template,
            show_system=self.show_system,
        )
        print("\n" + "=" * 80)
        print(display_text)
        print("-" * 80)

        # Get streaming input using configured handler
        stream_method = self.input_handler.stream_input
        if inspect.iscoroutinefunction(stream_method):
            char_stream = await stream_method(self.input_prompt)
        else:
            stream_or_awaitable = stream_method(self.input_prompt)
            if isinstance(stream_or_awaitable, Awaitable):
                char_stream = await stream_or_awaitable
            else:
                char_stream = stream_or_awaitable

        yield InputStreamedResponse(char_stream)


if __name__ == "__main__":
    import asyncio

    from pydantic_ai import Agent

    async def test_conversation():
        """Test the input model with a simple conversation."""
        model = InputModel(
            prompt_template="🤖 Question: {prompt}",
            show_system=True,
            input_prompt="Your answer: ",
        )

        agent: Agent[None, str] = Agent(
            model=model,
            system_prompt="You are helping test an input model. Be concise.",
        )

        # First question
        result = await agent.run("What's your favorite color?")
        print(f"\nFirst response: {result.data}")

        # Follow-up question using previous context
        result = await agent.run("Why do you like that color?")
        print(f"\nSecond response: {result.data}")

    asyncio.run(test_conversation())
