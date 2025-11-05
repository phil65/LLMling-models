"""Remote model implementation that supports full message protocol."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

import httpx
from pydantic_ai import (
    ModelMessagesTypeAdapter,
    ModelResponse,
    RequestUsage,
    TextPart,
)
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


class RemoteProxyModel(PydanticModel):
    """Model that proxies requests to a remote model server."""

    url: str = "ws://localhost:8000/v1/completion/stream"
    """URL of the remote model server."""

    api_key: str | None = None
    """API key for authentication."""

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return "remote-proxy"

    @property
    def system(self) -> str:
        """Return the system/provider name."""
        return "remote-proxy"

    @property
    def protocol(self) -> Literal["rest", "websocket"]:
        """Infer protocol from URL."""
        scheme = urlparse(self.url).scheme.lower()
        return "websocket" if scheme in ("ws", "wss") else "rest"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make request to remote model."""
        if self.protocol == "websocket":
            return await self._request_websocket(messages)
        return await self._request_rest(messages)

    async def _request_rest(self, messages: list[ModelMessage]) -> ModelResponse:
        """Make REST request to remote model."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(headers=headers) as client:
            try:
                # Serialize complete message history
                payload = ModelMessagesTypeAdapter.dump_json(messages)

                logger.debug("Sending request to %s", self.url)
                logger.debug("Request payload: %s", payload)

                response = await client.post(
                    f"{self.url}/v1/completion",
                    content=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30.0,
                )
                response.raise_for_status()

                # Deserialize response
                data = response.json()
                logger.debug("Received response: %s", data)

                model_response = ModelResponse(
                    parts=[TextPart(data["content"])],
                    timestamp=datetime.now(UTC),
                    usage=RequestUsage(**data.get("usage", {})),
                )
            except httpx.HTTPError as e:
                if hasattr(e, "response") and e.response is not None:  # type: ignore
                    logger.exception("Error response: %s", e.response.text)  # type: ignore
                msg = f"HTTP error: {e}"
                raise RuntimeError(msg) from e
            else:
                return model_response

    async def _request_websocket(self, messages: list[ModelMessage]) -> ModelResponse:
        """Make WebSocket request to remote model."""
        import anyenv
        import websockets

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        async with websockets.connect(self.url, extra_headers=headers) as websocket:
            try:
                # Serialize and send messages
                payload = ModelMessagesTypeAdapter.dump_json(messages)
                logger.debug("Sending WebSocket request: %s", payload)
                await websocket.send(payload)

                # Accumulate response chunks
                chunks: list[str] = []
                usage = RequestUsage()

                while True:
                    raw_data = await websocket.recv()
                    data = anyenv.load_json(raw_data, return_type=dict)
                    logger.debug("Received WebSocket data: %s", data)

                    if data.get("error"):
                        msg = f"Server error: {data['error']}"
                        raise RuntimeError(msg)

                    if data.get("usage"):
                        usage = RequestUsage(**data["usage"])

                    chunk = data.get("chunk")
                    if chunk is not None:  # Include empty strings but not None
                        chunks.append(chunk)

                    if data.get("done", False):
                        break

                content = "".join(chunks)
                if not content:
                    msg = "Received empty response from server"
                    raise RuntimeError(msg)
                ts = datetime.now(UTC)
                return ModelResponse(parts=[TextPart(content)], timestamp=ts, usage=usage)

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
        """Stream responses using WebSocket connection."""
        if self.protocol != "websocket":
            msg = "Streaming is only supported with WebSocket protocol"
            raise RuntimeError(msg)

        import websockets

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        websocket = await websockets.connect(self.url, extra_headers=headers)

        try:
            payload = ModelMessagesTypeAdapter.dump_json(messages)
            await websocket.send(payload)
            yield RemoteProxyStreamedResponse(
                model_request_parameters=ModelRequestParameters(),
                websocket=websocket,
            )

        except websockets.ConnectionClosed as e:
            msg = f"WebSocket error: {e}"
            raise RuntimeError(msg) from e
        finally:
            await websocket.close()


@dataclass(kw_only=True)
class RemoteProxyStreamedResponse(StreamedResponse):
    """Stream implementation for remote proxy responses."""

    websocket: ClientConnection
    _timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        """Initialize usage tracking."""
        self._usage = RequestUsage()

    @property
    def model_name(self) -> str:
        """Get response model_name."""
        return "remote-proxy"

    @property
    def provider_name(self) -> str | None:
        """Get the provider name."""
        return "remote-proxy"

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream responses as events."""
        import anyenv
        import websockets

        try:
            while True:
                try:
                    raw_data = await self.websocket.recv()
                    data = anyenv.load_json(raw_data, return_type=dict)
                    logger.debug("Stream received: %s", data)

                    if data.get("error"):
                        msg = f"Server error: {data['error']}"
                        raise RuntimeError(msg)

                    if data.get("usage"):
                        self._usage = RequestUsage(**data["usage"])

                    if data.get("done", False):
                        break

                    chunk = data.get("chunk")
                    if chunk:  # Only emit non-empty chunks
                        event = self._parts_manager.handle_text_delta(
                            vendor_part_id="content",
                            content=chunk,
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

    logging.basicConfig(level=logging.DEBUG)

    async def test():
        model = RemoteProxyModel(url="ws://localhost:8000/v1/completion/stream")
        agent: Agent[None, str] = Agent(model=model)

        # Test streaming
        logger.info("\nTesting streaming...")
        print("Streaming response:")
        chunk_count = 0

        async with agent.run_stream("Tell me a story about a brave knight") as response:
            # Use stream_text with delta=True instead of stream()
            async for chunk in response.stream_text(delta=True):
                chunk_count += 1
                print(chunk, end="", flush=True)

        print(f"\nStreaming complete! Received {chunk_count} chunks")

    asyncio.run(test())
