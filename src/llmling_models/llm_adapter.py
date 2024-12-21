"""Adapter to use LLM library models with Pydantic-AI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import llm
from pydantic import Field
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    AgentModel,
    EitherStreamedResponse,
    StreamTextResponse,
)
from pydantic_ai.result import Usage

from llmling_models.base import PydanticModel


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from pydantic_ai.settings import ModelSettings


class LLMAdapter(PydanticModel):
    """Adapter to use LLM library models with Pydantic-AI."""

    model_name: str = Field(description="Name of the LLM model to use")
    needs_key: str | None = None
    key_env_var: str | None = None
    can_stream: bool = False

    _async_model: llm.AsyncModel | None = None
    _sync_model: llm.Model | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        # Try async first
        try:
            self._async_model = llm.get_async_model(self.model_name)
            # If we got an async model, get its properties
            self.needs_key = self._async_model.needs_key
            self.key_env_var = self._async_model.key_env_var
            self.can_stream = self._async_model.can_stream
        except llm.UnknownModelError:
            pass
        else:
            return

        # Fall back to sync model if async not available
        try:
            self._sync_model = llm.get_model(self.model_name)
            self.needs_key = self._sync_model.needs_key
            self.key_env_var = self._sync_model.key_env_var
            self.can_stream = self._sync_model.can_stream
        except llm.UnknownModelError as e:
            msg = f"No sync or async model found for {self.model_name}"
            raise ValueError(msg) from e

    async def agent_model(
        self,
        *,
        function_tools: list[Any],
        allow_text_result: bool,
        result_tools: list[Any],
    ) -> AgentModel:
        """Create an agent model - tools are ignored for now."""
        return LLMAgentModel(
            async_model=self._async_model,
            sync_model=self._sync_model,
        )

    def name(self) -> str:
        """Return the model name."""
        return f"llm:{self.model_name}"


@dataclass
class LLMAgentModel(AgentModel):
    """AgentModel implementation for LLM models."""

    async_model: llm.AsyncModel | None
    sync_model: llm.Model | None

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
    ) -> tuple[ModelResponse, Usage]:
        """Make a request to the model."""
        prompt, system = self._build_prompt(messages)

        if self.async_model:
            response = await self.async_model.prompt(prompt, system=system, stream=False)
            text = await response.text()
            usage = await self._map_async_usage(response)
        elif self.sync_model:
            response = self.sync_model.prompt(prompt, system=system, stream=False)
            text = response.text()
            usage = self._map_sync_usage(response)
        else:
            msg = "No model available"
            raise RuntimeError(msg)

        return ModelResponse(
            parts=[TextPart(text)],
            timestamp=datetime.now(UTC),
        ), usage

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
    ) -> AsyncIterator[EitherStreamedResponse]:
        """Make a streaming request to the model."""
        prompt, system = self._build_prompt(messages)

        if self.async_model:
            response = await self.async_model.prompt(prompt, system=system, stream=True)
        elif self.sync_model and self.sync_model.can_stream:
            response = self.sync_model.prompt(prompt, system=system, stream=True)
        else:
            msg = (
                "No streaming capable model available. "
                "Either async model is missing or sync model doesn't support streaming."
            )
            raise RuntimeError(msg)

        yield LLMStreamTextResponse(response)

    @staticmethod
    def _build_prompt(messages: list[ModelMessage]) -> tuple[str, str | None]:
        """Build a prompt and optional system prompt from messages.

        Returns:
            Tuple of (prompt, system_prompt) where system_prompt may be None
        """
        prompt_parts = []
        system = None

        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        system = part.content
                    elif isinstance(part, UserPromptPart):
                        prompt_parts.append(part.content)

        return "\n".join(prompt_parts), system

    @staticmethod
    async def _map_async_usage(response: llm.AsyncResponse) -> Usage:
        """Map async LLM usage to Pydantic-AI usage."""
        await response._force()  # Ensure usage is available
        return Usage(
            request_tokens=response.input_tokens,
            response_tokens=response.output_tokens,
            total_tokens=((response.input_tokens or 0) + (response.output_tokens or 0)),
            details=response.token_details,
        )

    @staticmethod
    def _map_sync_usage(response: llm.Response) -> Usage:
        """Map sync LLM usage to Pydantic-AI usage."""
        response._force()  # Ensure usage is available
        return Usage(
            request_tokens=response.input_tokens,
            response_tokens=response.output_tokens,
            total_tokens=((response.input_tokens or 0) + (response.output_tokens or 0)),
            details=response.token_details,
        )


@dataclass
class LLMStreamTextResponse(StreamTextResponse):
    """Stream implementation for LLM responses."""

    _response: llm.Response | llm.AsyncResponse
    _timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    _buffer: list[str] = field(default_factory=list)
    _complete: bool = field(default=False)
    _accumulated_text: str = field(default="")
    _usage: Usage = field(default_factory=Usage)

    async def __anext__(self) -> None:
        """Process the next chunk without returning it.

        Chunks are accumulated and made available via get().
        """
        try:
            if isinstance(self._response, llm.AsyncResponse):
                chunk = await self._response.__anext__()
            else:
                chunk = next(iter(self._response))

            self._buffer.append(chunk)
            self._accumulated_text += chunk

            # Update usage if available
            if hasattr(self._response, "usage"):
                self._usage = Usage(
                    request_tokens=self._response.input_tokens,
                    response_tokens=self._response.output_tokens,
                    total_tokens=(
                        (self._response.input_tokens or 0)
                        + (self._response.output_tokens or 0)
                    ),
                    details=self._response.token_details,
                )

        except (StopIteration, StopAsyncIteration):
            self._complete = True
            raise StopAsyncIteration  # noqa: B904

    def get(self, *, final: bool = False) -> Iterable[str]:
        """Get accumulated chunks since last get call.

        Args:
            final: If True, this is the final call and should return any remaining text.
        """
        # If final and not complete, raise an error
        if final and not self._complete:
            msg = "Called get(final=True) before stream was complete"
            raise RuntimeError(msg)

        # Return buffered chunks and clear buffer
        chunks = self._buffer
        self._buffer = []
        return chunks

    def usage(self) -> Usage:
        """Get usage stats - only complete after stream is finished."""
        return self._usage

    def timestamp(self) -> datetime:
        """Get response timestamp."""
        return self._timestamp


if __name__ == "__main__":
    from pydantic_ai import Agent

    model = LLMAdapter(model_name="gpt-4o-mini")
    agent: Agent[None, str] = Agent(model)
    response = agent.run_sync("Say hello!")
    print(response)
