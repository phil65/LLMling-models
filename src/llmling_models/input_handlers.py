"""Input handling for interactive models."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Annotated, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ImportString
from pydantic_ai.messages import ModelMessage, SystemPromptPart, TextPart, UserPromptPart


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@runtime_checkable
class InputHandler(Protocol):
    """Protocol for input handlers."""

    def get_input(self, prompt: str) -> str:
        """Get single input response."""
        ...

    def stream_input(self, prompt: str) -> AsyncIterator[str]:
        """Stream input character by character."""
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

    def stream_input(self, prompt: str) -> AsyncIterator[str]:
        """Simulate streaming input using standard input.

        Yields each character as it's typed. Empty line ends input.
        """
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


class InputConfig(BaseModel):
    """Configuration for input handling."""

    prompt: str = Field(
        default="Your response: ",
        description="Prompt to show when requesting input",
    )
    handler: HandlerType = Field(
        default=DefaultInputHandler,
        description="Input handler class to use",
    )

    def get_handler(self) -> InputHandler:
        """Get configured input handler."""
        return self.handler()


if __name__ == "__main__":
    cfg = InputConfig(prompt="Your response: ")
    handler = cfg.get_handler()
    print(handler.get_input("What is your favorite color?"))
