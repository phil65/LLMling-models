"""Model that delegates responses to human input."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from pydantic import Field
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import AgentModel, EitherStreamedResponse, StreamTextResponse
from pydantic_ai.result import Usage

from llmling_models.base import PydanticModel
from llmling_models.input_handlers import InputConfig, InputHandler
from llmling_models.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition

logger = get_logger(__name__)


@dataclass
class InputStreamResponse(StreamTextResponse):
    """Stream implementation for input model."""

    _stream: AsyncIterator[str]
    _timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    _buffer: list[str] = field(default_factory=list)

    async def __anext__(self) -> None:
        """Get next character from stream."""
        char = await self._stream.__anext__()
        self._buffer.append(char)

    def get(self, *, final: bool = False) -> Iterable[str]:
        """Get accumulated characters."""
        chars = self._buffer.copy()
        self._buffer.clear()
        return chars

    def usage(self) -> Usage:
        """Get usage stats."""
        return Usage()

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

    input: InputConfig = Field(default_factory=InputConfig)
    """Input handler configuration."""

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
        return InputAgentModel(
            prompt_template=self.prompt_template,
            show_system=self.show_system,
            input_handler=self.input.get_handler(),
            input_prompt=self.input.prompt,
        )


class InputAgentModel(AgentModel):
    """AgentModel implementation that requests human input."""

    def __init__(
        self,
        prompt_template: str,
        show_system: bool,
        input_handler: InputHandler,
        input_prompt: str,
    ) -> None:
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
        # Format and display messages
        display_text = self._format_messages(messages)
        print("\n" + "=" * 80)
        print(display_text)
        print("-" * 80)

        # Get input using configured handler
        response = self.input_handler.get_input(self.input_prompt)

        return ModelResponse(
            parts=[TextPart(response)],
            timestamp=datetime.now(UTC),
        ), Usage()

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> AsyncIterator[EitherStreamedResponse]:
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
        char_stream = self.input_handler.stream_input(self.input_prompt)

        yield InputStreamResponse(char_stream)


if __name__ == "__main__":
    import asyncio

    from pydantic_ai import Agent

    async def test_conversation():
        """Test the input model with a simple conversation."""
        model = InputModel(
            prompt_template="🤖 Question: {prompt}",
            show_system=True,
            input=InputConfig(prompt="Your answer: "),
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
