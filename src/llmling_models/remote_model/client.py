"""Remote model implementation that supports full message protocol."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
import json
from typing import TYPE_CHECKING, Literal

import httpx
from pydantic import Field, TypeAdapter
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    ModelResponsePart,
)
from pydantic_ai.models import AgentModel, EitherStreamedResponse, StreamTextResponse
from pydantic_ai.result import Usage
import websockets

from llmling_models.base import PydanticModel
from llmling_models.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition
    from websockets import ClientConnection

logger = get_logger(__name__)


class RemoteProxyModel(PydanticModel):
    """Model that proxies requests to a remote model server.

    Example YAML configuration:
        ```yaml
        models:
          remote-gpt4:
            type: remote-proxy
            url: ws://model-server:8000/v1/completion  # or http://
            protocol: websocket  # or rest
            api_key: your-api-key
        ```
    """

    type: Literal["remote-proxy"] = Field(default="remote-proxy", init=False)
    """Discriminator field for model type."""

    url: str
    """URL of the remote model server."""

    protocol: Literal["rest", "websocket"] = "websocket"
    """Protocol to use for communication."""

    api_key: str
    """API key for authentication."""

    def name(self) -> str:
        """Get model name."""
        return f"remote-proxy({self.url})"

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create agent model implementation."""
        if self.protocol == "websocket":
            return WebSocketProxyAgent(url=self.url, api_key=self.api_key)
        return RestProxyAgent(url=self.url, api_key=self.api_key)


class RestProxyAgent(AgentModel):
    """Agent implementation using REST API."""

    def __init__(self, url: str, api_key: str) -> None:
        """Initialize with configuration."""
        headers = {"Authorization": f"Bearer {api_key}"}
        self.client = httpx.AsyncClient(base_url=url, headers=headers)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> tuple[ModelResponse, Usage]:
        """Make request to remote model."""
        try:
            # Serialize complete message history
            payload = ModelMessagesTypeAdapter.dump_json(messages)
            headers = {"Content-Type": "application/json"}
            response = await self.client.post(
                "/completion", content=payload, headers=headers
            )
            response.raise_for_status()

            # Deserialize response
            data = response.json()
            model_response = ModelResponse.from_text(
                data["content"],
                timestamp=datetime.now(UTC),
            )
            usage = Usage(**data.get("usage", {}))
        except httpx.HTTPError as e:
            msg = f"HTTP error: {e}"
            raise RuntimeError(msg) from e
        else:
            return model_response, usage


class WebSocketStreamResponse(StreamTextResponse):
    """Stream implementation for WebSocket responses."""

    response_part_adapter = TypeAdapter(ModelResponsePart)

    def __init__(self, websocket: ClientConnection) -> None:
        """Initialize with active WebSocket."""
        self.websocket = websocket
        self._buffer: list[str] = []
        self._complete = False
        self._timestamp = datetime.now(UTC)
        self._accumulated_parts: list[ModelResponsePart] = []

    async def __anext__(self) -> None:
        """Get next response chunk."""
        if self._complete:
            raise StopAsyncIteration

        try:
            raw_data = await self.websocket.recv()
            data = json.loads(raw_data)

            if data.get("error"):
                msg = f"Server error: {data['error']}"
                raise RuntimeError(msg)

            if data["done"]:
                self._complete = True
                raise StopAsyncIteration

            # Handle streamed response parts
            if "part" in data:
                part = self.response_part_adapter.validate_python(data["part"])
                self._accumulated_parts.append(part)
                # Extract text content if available
                if hasattr(part, "content"):
                    self._buffer.append(str(part.content))
            else:
                self._buffer.append(data["chunk"])

        except (websockets.ConnectionClosed, ValueError, KeyError) as e:
            msg = f"Stream error: {e}"
            raise RuntimeError(msg) from e

    def get(self, *, final: bool = False) -> Iterable[str]:
        """Get accumulated response chunks."""
        chunks = self._buffer.copy()
        self._buffer.clear()
        return chunks

    def usage(self) -> Usage:
        """Get usage statistics."""
        return Usage()

    def timestamp(self) -> datetime:
        """Get response timestamp."""
        return self._timestamp


class WebSocketProxyAgent(AgentModel):
    """Agent implementation using WebSocket connection."""

    response_part_adapter = TypeAdapter(ModelResponsePart)

    def __init__(self, url: str, api_key: str) -> None:
        """Initialize with configuration."""
        self.url = url
        self.api_key = api_key

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> tuple[ModelResponse, Usage]:
        """Make request using WebSocket connection."""
        async with websockets.connect(
            self.url,
            extra_headers={"Authorization": f"Bearer {self.api_key}"},
        ) as websocket:
            try:
                # Serialize and send messages
                await websocket.send(ModelMessagesTypeAdapter.dump_json(messages))

                # Accumulate response parts
                parts: list[ModelResponsePart] = []
                usage = Usage()

                while True:
                    raw_data = await websocket.recv()
                    data = json.loads(raw_data)

                    if data.get("error"):
                        msg = f"Server error: {data['error']}"
                        raise RuntimeError(msg)

                    if data.get("usage"):
                        usage = Usage(**data["usage"])

                    if "part" in data:
                        part = self.response_part_adapter.validate_python(data["part"])
                        parts.append(part)

                    if data.get("done", False):
                        break

                return ModelResponse(parts=parts), usage

            except (websockets.ConnectionClosed, ValueError, KeyError) as e:
                msg = f"WebSocket error: {e}"
                raise RuntimeError(msg) from e

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> AsyncIterator[EitherStreamedResponse]:
        """Stream responses using WebSocket connection."""
        websocket = await websockets.connect(
            self.url,
            extra_headers={"Authorization": f"Bearer {self.api_key}"},
        )

        try:
            # Send messages
            await websocket.send(ModelMessagesTypeAdapter.dump_json(messages))
            yield WebSocketStreamResponse(websocket)

        except websockets.ConnectionClosed as e:
            msg = f"WebSocket error: {e}"
            raise RuntimeError(msg) from e
        finally:
            await websocket.close()


if __name__ == "__main__":
    import asyncio

    from pydantic_ai import Agent

    async def test():
        model = RemoteProxyModel(
            url="ws://localhost:8000/v1/completion",
            protocol="websocket",
            api_key="test-key",
        )
        agent: Agent[None, str] = Agent(model=model)
        response = await agent.run("Hello! How are you?")
        print(f"Response: {response.data}")

    asyncio.run(test())
