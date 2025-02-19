"""Custom OpenAI model implementation using only httpx."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import os
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from pydantic import Field, TypeAdapter
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.result import Usage

from llmling_models.base import PydanticModel
from llmling_models.log import get_logger


class StreamChunk(TypedDict):
    """OpenAI stream chunk format."""

    choices: list[dict[str, Any]]
    usage: dict[str, int] | None


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import httpx
    from pydantic_ai.settings import ModelSettings


logger = get_logger(__name__)
json_ta = TypeAdapter[Any](Any)


def convert_messages(messages: list[ModelMessage]) -> list[dict[str, str]]:
    """Convert pydantic-ai messages to OpenAI format."""
    result = []

    for message in messages:
        if isinstance(message, ModelResponse):
            content = ""
            for part in message.parts:
                if isinstance(part, TextPart | ToolReturnPart):
                    content += str(part.content)
            if content:
                result.append({"role": "assistant", "content": content})
        else:
            for part in message.parts:  # type: ignore
                if isinstance(part, SystemPromptPart):
                    result.append({"role": "system", "content": part.content})
                elif isinstance(part, UserPromptPart):
                    result.append({"role": "user", "content": part.content})

    return result


@dataclass(kw_only=True)
class OpenAIStreamedResponse(StreamedResponse):
    """Stream implementation for OpenAI responses."""

    response: httpx.Response
    _timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    _model_name: str

    def __post_init__(self):
        """Initialize usage tracking and parts manager."""
        self._usage = Usage()
        self._parts_manager = ModelResponsePartsManager()
        self._has_yielded_start = False
        self._buffer = ""
        self._stream = None

    def get(self) -> ModelResponse:
        """Get current state of response."""
        return ModelResponse(
            parts=self._parts_manager.get_parts(),
            model_name=self._model_name,
            timestamp=self._timestamp,
        )

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream response chunks."""
        try:
            content_id = "content"  # OpenAI uses a single content stream
            accumulated_text = ""

            # Get lines from the response
            async for line in self.response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue

                if line == "data: [DONE]":
                    break

                try:
                    data = json.loads(line.removeprefix("data: "))
                except json.JSONDecodeError:
                    continue

                if data.get("error"):
                    msg = f"OpenAI error: {data['error']}"
                    raise RuntimeError(msg)  # noqa: TRY301

                choices = data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                if not delta:
                    continue

                # Extract content
                if content := delta.get("content"):
                    accumulated_text += content

                    # First chunk - emit start event
                    if not self._has_yielded_start:
                        self._has_yielded_start = True
                        event = self._parts_manager.handle_text_delta(
                            vendor_part_id=content_id,
                            content=accumulated_text,
                        )
                        yield event
                    else:
                        # Subsequent chunks - emit delta events
                        event = self._parts_manager.handle_text_delta(
                            vendor_part_id=content_id,
                            content=content,
                        )
                        yield event

                # Update usage if available
                if usage := data.get("usage"):
                    self._usage = Usage(
                        request_tokens=usage.get("prompt_tokens", 0),
                        response_tokens=usage.get("completion_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                    )

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
        return self._model_name


class SimpleOpenAIModel(PydanticModel):
    """OpenAI model implementation using only httpx."""

    type: Literal["openai-simple"] = Field(default="openai-simple", init=False)
    model: str
    """OpenAI model identifier."""

    api_key: str | None = None
    """OpenAI API key."""

    base_url: str | None = None
    """Base URL for API requests."""

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.model

    @property
    def system(self) -> str:
        """Return the system/provider name."""
        return "openai-simple"

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            msg = "OpenAI API key not provided"
            raise ValueError(msg)

        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def _build_request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build request payload."""
        req: dict[str, Any] = {
            "model": self.model,
            "messages": convert_messages(messages),
            "stream": stream,
        }

        # Add model settings if provided
        if model_settings:
            if temperature := model_settings.get("temperature"):
                req["temperature"] = temperature
            if max_tokens := model_settings.get("max_tokens"):
                req["max_tokens"] = max_tokens

        return req

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, Usage]:
        """Make request to OpenAI API."""
        import httpx

        headers = self._get_headers()
        payload = self._build_request(messages, model_settings)
        base_url = self.base_url or "https://api.openai.com/v1"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

                # Extract content
                content = data["choices"][0]["message"]["content"]

                # Extract usage
                usage_data = data.get("usage", {})
                usage = Usage(
                    request_tokens=usage_data.get("prompt_tokens", 0),
                    response_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                )

                return ModelResponse(
                    parts=[TextPart(content)],
                    timestamp=datetime.now(UTC),
                ), usage

            except httpx.HTTPError as e:
                msg = f"OpenAI request failed: {e}"
                raise RuntimeError(msg) from e

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        """Stream response from OpenAI API."""
        import httpx

        headers = self._get_headers()
        payload = self._build_request(messages, model_settings, stream=True)
        base_url = self.base_url or "https://api.openai.com/v1"

        client = httpx.AsyncClient(timeout=30.0)
        try:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

            yield OpenAIStreamedResponse(
                response=response,
                _model_name=self.model_name,
            )

        except httpx.HTTPError as e:
            msg = f"OpenAI stream request failed: {e}"
            raise RuntimeError(msg) from e
        finally:
            await client.aclose()


if __name__ == "__main__":
    import asyncio

    from pydantic_ai import Agent

    async def test():
        # Create model instance
        model = SimpleOpenAIModel(
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Test with agent
        agent: Agent[None, str] = Agent(model=model)
        result = await agent.run("Say hello!")
        print(f"\nResponse: {result.data}")

        # Test streaming
        print("\nStreaming response:")
        async with agent.run_stream("Tell me a short story") as stream:
            async for chunk in stream.stream_text(delta=True):
                print(chunk, end="", flush=True)
        print("\nStreaming complete!")

    asyncio.run(test())
