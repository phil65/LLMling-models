"""Adapter to use LLM library models with Pydantic-AI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import llm
from pydantic import Field
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.result import Usage

from llmling_models.base import PydanticModel
from llmling_models.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.settings import ModelSettings

logger = get_logger(__name__)


class LLMAdapter(PydanticModel):
    """Adapter to use LLM library models with Pydantic-AI."""

    type: Literal["llm"] = Field(default="llm", init=False)
    model: str = Field(description="Name of the LLM model to use")
    needs_key: str | None = None
    key_env_var: str | None = None
    can_stream: bool = False

    _async_model: llm.AsyncModel | None = None
    _sync_model: llm.Model | None = None

    def __init__(self, **data: Any):
        super().__init__(**data)

        # Try async first
        try:
            self._async_model = llm.get_async_model(self.model)
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
            self._sync_model = llm.get_model(self.model)
            self.needs_key = self._sync_model.needs_key
            self.key_env_var = self._sync_model.key_env_var
            self.can_stream = self._sync_model.can_stream
        except llm.UnknownModelError as e:
            msg = f"No sync or async model found for {self.model}"
            raise ValueError(msg) from e

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.model

    @property
    def system(self) -> str:
        """Return the system/provider name."""
        return "llm"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, Usage]:
        """Make a request to the model."""
        prompt, system = self._build_prompt(messages)

        if self._async_model:
            response = await self._async_model.prompt(prompt, system=system, stream=False)
            text = await response.text()
            usage = await self._map_async_usage(response)
        elif self._sync_model:
            response = self._sync_model.prompt(prompt, system=system, stream=False)
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
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streaming request to the model."""
        prompt, system = self._build_prompt(messages)

        if self._async_model:
            response = await self._async_model.prompt(prompt, system=system, stream=True)
        elif self._sync_model and self._sync_model.can_stream:
            response = self._sync_model.prompt(prompt, system=system, stream=True)
        else:
            msg = (
                "No streaming capable model available. "
                "Either async model is missing or sync model doesn't support streaming."
            )
            raise RuntimeError(msg)

        yield LLMStreamedResponse(response=response)

    @staticmethod
    def _build_prompt(messages: list[ModelMessage]) -> tuple[str, str | None]:
        """Build a prompt and optional system prompt from messages."""
        prompt_parts = []
        system = None

        for message in messages:
            if isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, TextPart | ToolReturnPart):
                        prompt_parts.append(f"Assistant: {part.content}")  # noqa: PERF401
            else:  # ModelRequest
                for part in message.parts:  # type: ignore
                    if isinstance(part, SystemPromptPart):
                        system = part.content
                    elif isinstance(part, UserPromptPart | RetryPromptPart):
                        prompt_parts.append(f"Human: {part.content}")

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


@dataclass(kw_only=True)
class LLMStreamedResponse(StreamedResponse):
    """Stream implementation for LLM responses."""

    response: llm.Response | llm.AsyncResponse
    _timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    _model_name: str = "llm"

    def __post_init__(self):
        """Initialize usage."""
        self._usage = Usage()

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream response chunks as events."""
        try:
            while True:
                try:
                    if isinstance(self.response, llm.AsyncResponse):
                        chunk = await self.response.__anext__()
                    else:
                        chunk = next(iter(self.response))

                    # Update usage if available
                    if hasattr(self.response, "usage"):
                        self._usage = Usage(
                            request_tokens=self.response.input_tokens,
                            response_tokens=self.response.output_tokens,
                            total_tokens=(
                                (self.response.input_tokens or 0)
                                + (self.response.output_tokens or 0)
                            ),
                            details=self.response.token_details,
                        )

                    # Emit text delta event
                    yield self._parts_manager.handle_text_delta(
                        vendor_part_id="content",
                        content=chunk,
                    )

                except (StopIteration, StopAsyncIteration):
                    break

        except Exception as e:
            msg = f"Stream error: {e}"
            raise RuntimeError(msg) from e

    @property
    def model_name(self) -> str:
        """Get response model_name."""
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        """Get response timestamp."""
        return self._timestamp


if __name__ == "__main__":
    import asyncio

    from pydantic_ai import Agent

    async def test():
        # Test with both sync and async models
        adapter = LLMAdapter(model_name="gpt-4o-mini")
        agent: Agent[None, str] = Agent(model=adapter)

        print("\nTesting sync request:")
        response = await agent.run("Say hello!")
        print(f"Response: {response.data}")

        print("\nTesting streaming:")
        async with agent.run_stream("Tell me a story") as stream:
            async for chunk in stream.stream_text(delta=True):
                print(chunk, end="", flush=True)
        print()

    asyncio.run(test())
