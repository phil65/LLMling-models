"""Client implementation for remote human-in-the-loop conversations."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

import httpx
from pydantic import Field
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
    TextPart,
)
from pydantic_ai.models import AgentModel, StreamedResponse
from pydantic_ai.result import Usage
import websockets

from llmling_models.base import PydanticModel
from llmling_models.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition
    from websockets import ClientConnection

logger = get_logger(__name__)


class RemoteInputModel(PydanticModel):
    """Model that connects to a remote human operator.

    Example YAML configuration:
        ```yaml
        models:
          remote-human:
            type: remote-input
            url: ws://localhost:8000/v1/chat/stream  # or http://localhost:8000/v1/chat
            api_key: your-api-key
        ```
    """

    type: Literal["remote-input"] = Field(default="remote-input", init=False)
    """Discriminator field for model type."""

    url: str = "ws://localhost:8000/v1/chat/stream"
    """URL of the remote input server."""

    api_key: str | None = None
    """API key for authentication."""

    def name(self) -> str:
        """Get model name."""
        return f"remote-input({self.url})"

    @property
    def protocol(self) -> Literal["rest", "websocket"]:
        """Infer protocol from URL."""
        scheme = urlparse(self.url).scheme.lower()
        return "websocket" if scheme in ("ws", "wss") else "rest"

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create agent model implementation."""
        if self.protocol == "websocket":
            return WebSocketRemoteAgent(url=self.url, api_key=self.api_key)
        return RestRemoteAgent(url=self.url, api_key=self.api_key)


def extract_conversation(messages: list[ModelMessage]) -> list[dict[str, str]]:
    """Extract simple conversation history from messages."""
    history = []

    for message in messages:
        role = "assistant" if isinstance(message, ModelResponse) else "user"
        content = ""

        for part in message.parts:
            if hasattr(part, "content"):
                content += str(part.content)  # pyright: ignore

        if content:
            history.append({"role": role, "content": content})

    return history


class RestRemoteAgent(AgentModel):
    """Agent implementation using REST API."""

    def __init__(self, url: str, api_key: str | None = None):
        """Initialize with configuration."""
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(base_url=url, headers=headers)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> tuple[ModelResponse, Usage]:
        """Make request to remote operator."""
        try:
            # Get current prompt from last message
            prompt = ""
            if messages:
                last_message = messages[-1]
                for part in last_message.parts:
                    if hasattr(part, "content"):
                        prompt += str(part.content)  # pyright: ignore

            # Extract conversation history
            conversation = []
            if len(messages) > 1:  # Only if there's history
                for message in messages[:-1]:  # Exclude the current prompt
                    for part in message.parts:
                        if hasattr(part, "content"):
                            role = (
                                "assistant"
                                if isinstance(message, ModelResponse)
                                else "user"
                            )
                            conversation.append({
                                "role": role,
                                "content": str(part.content),  # pyright: ignore
                            })

            # Log request data for debugging
            request_data = {"prompt": prompt, "conversation": conversation}
            logger.debug("Sending request data: %s", request_data)

            # Make request
            response = await self.client.post(
                "/v1/chat/completions",
                json=request_data,
                timeout=30.0,  # Add timeout
            )
            response.raise_for_status()

            response_data = response.json()
            logger.debug("Received response: %s", response_data)

            part = TextPart(response_data["content"])
            return ModelResponse(parts=[part]), Usage()

        except httpx.HTTPError as e:
            # Log the full error response if available
            if hasattr(e, "response") and e.response is not None:  # type: ignore
                logger.exception("Error response: %s", e.response.text)  # type: ignore
            msg = f"HTTP error: {e}"
            raise RuntimeError(msg) from e


@dataclass
class WebSocketStreamedResponse(StreamedResponse):
    """Stream implementation for WebSocket responses."""

    websocket: ClientConnection
    _timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        """Initialize usage tracking."""
        self._usage = Usage()

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream responses as events."""
        try:
            while True:
                try:
                    raw_data = await self.websocket.recv()
                    data = json.loads(raw_data)

                    if data.get("error"):
                        msg = f"Server error: {data['error']}"
                        raise RuntimeError(msg)

                    if data["done"]:
                        break

                    # Emit text delta event for each chunk
                    yield self._parts_manager.handle_text_delta(
                        vendor_part_id="content",
                        content=data["chunk"],
                    )

                except (websockets.ConnectionClosed, ValueError, KeyError) as e:
                    msg = f"Stream error: {e}"
                    raise RuntimeError(msg) from e

        except Exception as e:
            msg = f"Stream error: {e}"
            raise RuntimeError(msg) from e

    def timestamp(self) -> datetime:
        """Get response timestamp."""
        return self._timestamp


class WebSocketRemoteAgent(AgentModel):
    """Agent implementation using WebSocket connection."""

    def __init__(self, url: str, api_key: str | None = None):
        """Initialize with configuration."""
        self.url = url
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(headers=self.headers)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> tuple[ModelResponse, Usage]:
        """Make request using WebSocket connection."""
        async with websockets.connect(self.url, extra_headers=self.headers) as websocket:
            try:
                # Get current prompt and history
                prompt = ""
                if messages:
                    last_message = messages[-1]
                    for part in last_message.parts:
                        if hasattr(part, "content"):
                            prompt += str(part.content)  # pyright: ignore

                conversation = extract_conversation(messages[:-1])
                data = json.dumps({"prompt": prompt, "conversation": conversation})
                # Send request
                await websocket.send(data)

                # Accumulate response characters
                response_text = ""
                while True:
                    raw_data = await websocket.recv()
                    dct = json.loads(raw_data)
                    if dct.get("error"):
                        msg = f"Server error: {dct['error']}"
                        raise RuntimeError(msg)

                    if dct["done"]:
                        break

                    response_text += dct["chunk"]
                part = TextPart(response_text)
                return (ModelResponse(parts=[part]), Usage())

            except (websockets.ConnectionClosed, ValueError, KeyError) as e:
                msg = f"WebSocket error: {e}"
                raise RuntimeError(msg) from e

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Stream responses from operator."""
        websocket = await websockets.connect(self.url, extra_headers=self.headers)

        try:
            # Send prompt and history
            prompt = ""
            if messages:
                last_message = messages[-1]
                for part in last_message.parts:
                    if hasattr(part, "content"):
                        prompt += str(part.content)  # pyright: ignore

            conversation = extract_conversation(messages[:-1])
            data = json.dumps({"prompt": prompt, "conversation": conversation})
            await websocket.send(data)

            yield WebSocketStreamedResponse(websocket)

        except websockets.ConnectionClosed as e:
            msg = f"WebSocket error: {e}"
            raise RuntimeError(msg) from e
        finally:
            await websocket.close()

    async def __aenter__(self):
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        await self.client.aclose()


if __name__ == "__main__":
    import asyncio
    import logging

    from pydantic_ai import Agent

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    async def test():
        # Test both protocols
        print("\nTesting REST protocol:")
        model = RemoteInputModel(
            url="http://localhost:8000",  # Base URL only
            api_key="test-key",
        )
        agent: Agent[None, str] = Agent(
            model=model, system_prompt="You are a helpful assistant."
        )
        response = await agent.run("Hello! How are you?")
        print(f"\nResponse: {response.data}")

    asyncio.run(test())
