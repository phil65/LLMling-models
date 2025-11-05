"""Client implementation for remote human-in-the-loop conversations."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

import httpx
from pydantic_ai import ModelResponse, RequestUsage, TextPart
from pydantic_ai.models import ModelRequestParameters, StreamedResponse

from llmling_models.base import PydanticModel
from llmling_models.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai import (
        ModelMessage,
        ModelResponseStreamEvent,
        ModelSettings,
        RunContext,
    )
    from websockets import ClientConnection

logger = get_logger(__name__)


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


class RemoteInputModel(PydanticModel):
    """Model that connects to a remote human operator."""

    url: str = "ws://localhost:8000/v1/chat/stream"
    """URL of the remote input server."""

    api_key: str | None = None
    """API key for authentication."""

    @property
    def protocol(self) -> Literal["rest", "websocket"]:
        """Infer protocol from URL."""
        scheme = urlparse(self.url).scheme.lower()
        return "websocket" if scheme in ("ws", "wss") else "rest"

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return "remote-input"

    @property
    def system(self) -> str:
        """Return the system/provider name."""
        return "remote-operator"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make request to remote operator."""
        if self.protocol == "websocket":
            return await self._request_websocket(messages)
        return await self._request_rest(messages)

    async def _request_rest(self, messages: list[ModelMessage]) -> ModelResponse:
        """Make REST request to remote operator."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(headers=headers) as client:
            try:
                # Get current prompt and history
                prompt = ""
                if messages:
                    last_message = messages[-1]
                    for part in last_message.parts:
                        if hasattr(part, "content"):
                            prompt += str(part.content)  # pyright: ignore

                conversation = extract_conversation(messages[:-1])

                # Log request data for debugging
                request_data = {"prompt": prompt, "conversation": conversation}
                logger.debug("Sending request data: %s", request_data)

                # Make request
                response = await client.post(
                    f"{self.url}/v1/chat/completions",
                    json=request_data,
                    timeout=30.0,
                )
                response.raise_for_status()

                response_data = response.json()
                logger.debug("Received response: %s", response_data)

                return ModelResponse(
                    parts=[TextPart(response_data["content"])],
                    timestamp=datetime.now(UTC),
                    usage=RequestUsage(),
                )

            except httpx.HTTPError as e:
                if hasattr(e, "response") and e.response is not None:  # type: ignore
                    logger.exception("Error response: %s", e.response.text)  # type: ignore
                msg = f"HTTP error: {e}"
                raise RuntimeError(msg) from e

    async def _request_websocket(self, messages: list[ModelMessage]) -> ModelResponse:
        """Make WebSocket request to remote operator."""
        import anyenv
        import websockets

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        async with websockets.connect(self.url, extra_headers=headers) as websocket:
            try:
                # Get current prompt and history
                prompt = ""
                if messages:
                    last_message = messages[-1]
                    for part in last_message.parts:
                        if hasattr(part, "content"):
                            prompt += str(part.content)  # pyright: ignore

                conversation = extract_conversation(messages[:-1])
                data = anyenv.dump_json({"prompt": prompt, "conversation": conversation})

                # Send request
                await websocket.send(data)

                # Accumulate response characters
                response_text = ""
                while True:
                    raw_data = await websocket.recv()
                    dct = anyenv.load_json(raw_data, return_type=dict)
                    if dct.get("error"):
                        msg = f"Server error: {dct['error']}"
                        raise RuntimeError(msg)

                    if dct["done"]:
                        break

                    response_text += dct["chunk"]

                return ModelResponse(
                    parts=[TextPart(response_text)],
                    timestamp=datetime.now(UTC),
                    usage=RequestUsage(),
                )

            except (websockets.ConnectionClosed, ValueError, KeyError) as e:
                msg = f"WebSocket error: {e}"
                raise RuntimeError(msg) from e

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Stream responses from operator."""
        if self.protocol != "websocket":
            msg = "Streaming is only supported with WebSocket protocol"
            raise RuntimeError(msg)

        import anyenv
        import websockets

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        websocket = await websockets.connect(self.url, extra_headers=headers)

        try:
            # Send prompt and history
            prompt = ""
            if messages:
                last_message = messages[-1]
                for part in last_message.parts:
                    if hasattr(part, "content"):
                        prompt += str(part.content)  # pyright: ignore

            conversation = extract_conversation(messages[:-1])
            data = anyenv.dump_json({"prompt": prompt, "conversation": conversation})
            await websocket.send(data)

            yield RemoteInputStreamedResponse(
                model_request_parameters=ModelRequestParameters(),
                websocket=websocket,
            )

        except websockets.ConnectionClosed as e:
            msg = f"WebSocket error: {e}"
            raise RuntimeError(msg) from e
        finally:
            await websocket.close()


@dataclass(kw_only=True)
class RemoteInputStreamedResponse(StreamedResponse):
    """Stream implementation for remote input responses."""

    websocket: ClientConnection
    _timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        """Initialize usage tracking."""
        self._usage = RequestUsage()

    @property
    def model_name(self) -> str:
        """Get response model_name."""
        return "remote-input"

    @property
    def provider_name(self) -> str:
        """Get response provider name."""
        return "remote-input"

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream responses as events."""
        import anyenv
        import websockets

        try:
            while True:
                try:
                    raw_data = await self.websocket.recv()
                    data = anyenv.load_json(raw_data, return_type=dict)

                    if data.get("error"):
                        msg = f"Server error: {data['error']}"
                        raise RuntimeError(msg)

                    if data["done"]:
                        break

                    # Emit text delta event for each chunk
                    event = self._parts_manager.handle_text_delta(
                        vendor_part_id="content",
                        content=data["chunk"],
                    )
                    if event is not None:
                        yield event

                except (websockets.ConnectionClosed, ValueError, KeyError) as e:
                    msg = f"Stream error: {e}"
                    raise RuntimeError(msg) from e

        except Exception as e:
            msg = f"Stream error: {e}"
            raise RuntimeError(msg) from e

    @property
    def timestamp(self) -> datetime:
        """Get response timestamp."""
        return self._timestamp


if __name__ == "__main__":
    import asyncio
    import logging

    from pydantic_ai import Agent

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    async def test():
        model = RemoteInputModel(url="http://localhost:8000", api_key="test-key")
        agent: Agent[None, str] = Agent(model=model)
        response = await agent.run("Hello! How are you?")
        print(f"\nResponse: {response.output}")

    asyncio.run(test())
