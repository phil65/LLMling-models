"""Input handling for interactive models."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Annotated, Protocol, runtime_checkable

from pydantic import ImportString
from pydantic_ai.messages import (
    ModelMessage,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable


@runtime_checkable
class InputHandler(Protocol):
    """Protocol for input handlers."""

    def get_input(self, prompt: str) -> str | Awaitable[str]:
        """Get single input response. Can be sync or async."""
        ...

    def stream_input(
        self, prompt: str
    ) -> AsyncIterator[str] | Awaitable[AsyncIterator[str]]:
        """Stream input character by character. Can be sync or async."""
        ...

    def format_messages(
        self,
        messages: list[ModelMessage],
        *,
        prompt_template: str,
        show_system: bool,
    ) -> str:
        """Format messages for display."""
        ...


class DefaultInputHandler:
    """Default input handler using standard input."""

    def get_input(self, prompt: str) -> str:
        """Get input using basic console input."""
        return input(prompt).strip()

    async def stream_input(self, prompt: str) -> AsyncIterator[str]:
        """Simulate streaming input using standard input."""
        print(prompt, end="", flush=True)

        async def char_iterator():
            while True:
                char = sys.stdin.read(1)
                if char == "\n":
                    break
                yield char

        return char_iterator()

    def format_messages(
        self,
        messages: list[ModelMessage],
        *,
        prompt_template: str,
        show_system: bool,
    ) -> str:
        """Format messages for display."""
        formatted: list[str] = []

        for message in messages:
            for part in message.parts:
                match part:
                    case SystemPromptPart() if show_system:
                        formatted.append(f"🔧 System: {part.content}")
                    case UserPromptPart():
                        formatted.append(prompt_template.format(prompt=part.content))
                    case TextPart():
                        formatted.append(f"Assistant: {part.content}")
                    case _:
                        continue

        return "\n\n".join(formatted)


HandlerType = Annotated[type[InputHandler], ImportString]
