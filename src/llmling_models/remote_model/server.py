"""FastAPI server implementation for full model protocol support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, TypeAdapter
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    ModelResponsePart,
    TextPart,
)

from llmling_models.log import get_logger


logger = get_logger(__name__)
security = HTTPBearer()


@dataclass
class Usage:
    """LLM usage tracking."""

    requests: int = 0
    request_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert usage to dict for serialization."""
        return {
            "requests": self.requests,
            "request_tokens": self.request_tokens,
            "response_tokens": self.response_tokens,
            "total_tokens": self.total_tokens,
            "details": self.details,
        }


class StreamResponse(BaseModel):
    """Streaming response chunk."""

    part: ModelResponsePart | None = None
    """Response part if available."""

    chunk: str | None = None
    """Text chunk if not a full part."""

    done: bool = False
    """Whether this is the final chunk."""

    error: str | None = None
    """Optional error message."""

    usage: dict[str, Any] | None = None
    """Optional usage information."""


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self) -> None:
        """Initialize connection store."""
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> str:
        """Accept and store new connection."""
        await websocket.accept()
        connection_id = str(uuid4())
        self.active_connections[connection_id] = websocket
        return connection_id

    def disconnect(self, connection_id: str) -> None:
        """Remove stored connection."""
        self.active_connections.pop(connection_id, None)

    async def send_error(
        self,
        websocket: WebSocket,
        error: str,
        code: int = status.WS_1011_INTERNAL_ERROR,
    ) -> None:
        """Send error message and close connection."""
        try:
            await websocket.send_json(
                StreamResponse(
                    error=error,
                    done=True,
                ).model_dump(exclude_none=True),
            )
            await websocket.close(code=code)
        except WebSocketDisconnect:
            pass


def format_conversation(messages: list[ModelMessage]) -> str:
    """Format conversation history for display."""
    lines = []

    for message in messages:
        for part in message.parts:
            if hasattr(part, "content"):
                prefix = "🤖" if isinstance(message, ModelResponse) else "👤"
                lines.append(f"{prefix} {part.content}")  # pyright: ignore

    return "\n".join(lines)


class ModelServer:
    """FastAPI server with full model protocol support."""

    response_part_adapter = TypeAdapter(ModelResponsePart)

    def __init__(
        self,
        title: str = "Remote Model Server",
        description: str | None = None,
    ) -> None:
        """Initialize server with configuration."""
        self.app = FastAPI(title=title, description=description or "No description")
        self.manager = ConnectionManager()
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Configure API routes."""

        @self.app.post("/v1/completion")
        async def create_completion(
            messages: Annotated[list[ModelMessage], ModelMessagesTypeAdapter],
            auth: Annotated[HTTPAuthorizationCredentials, security],
        ) -> dict[str, Any]:
            """Handle completion requests via REST."""
            try:
                # Display conversation to operator
                print("\n" + "=" * 80)
                print("Conversation history:")
                print(format_conversation(messages))
                print("-" * 80)
                print("Your response: ")

                # Get operator's response
                response = input().strip()

                # Track basic usage
                usage = Usage(requests=1)

                return {
                    "content": response,
                    "usage": usage.to_dict(),
                }

            except Exception as e:
                logger.exception("Error processing completion request")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e),
                ) from e

        @self.app.websocket("/v1/completion/stream")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            """Handle streaming conversation via WebSocket."""
            connection_id = None
            usage = Usage(requests=1)

            try:
                connection_id = await self.manager.connect(websocket)
                logger.info("New WebSocket connection: %s", connection_id)

                while True:
                    try:
                        # Receive and parse messages
                        raw_messages = await websocket.receive_text()
                        messages = ModelMessagesTypeAdapter.validate_json(raw_messages)

                        # Display conversation
                        print("\n" + "=" * 80)
                        print("Conversation history:")
                        print(format_conversation(messages))
                        print("-" * 80)
                        print("Type your response (press Enter twice when done):")

                        # Get response character by character
                        response: list[str] = []
                        while True:
                            line = input()
                            if not line and response:  # Empty line after content
                                break
                            if line:
                                response.append(line)
                                # Create and send part
                                part = TextPart(content=line + "\n")
                                await websocket.send_json(
                                    StreamResponse(
                                        part=self.response_part_adapter.dump_python(part),
                                    ).model_dump(exclude_none=True),
                                )

                        # Send final message with usage
                        await websocket.send_json(
                            StreamResponse(
                                done=True,
                                usage=usage.to_dict(),
                            ).model_dump(exclude_none=True),
                        )

                    except ValueError as e:
                        await self.manager.send_error(
                            websocket,
                            f"Invalid request: {e}",
                            status.WS_1003_UNSUPPORTED_DATA,
                        )

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected: %s", connection_id)

            finally:
                if connection_id:
                    self.manager.disconnect(connection_id)

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        **kwargs: Any,
    ) -> None:
        """Start the server."""
        import uvicorn

        uvicorn.run(self.app, host=host, port=port, **kwargs)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    server = ModelServer(
        title="Remote Model Server",
        description="Server that supports full model protocol",
    )
    server.run(port=8000)
