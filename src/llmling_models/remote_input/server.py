"""FastAPI server for remote human-in-the-loop conversations."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from llmling_models.log import get_logger


logger = get_logger(__name__)
security = HTTPBearer()


class Message(BaseModel):
    """Single conversation message."""

    role: Literal["user", "assistant"]
    content: str


class CompletionRequest(BaseModel):
    """Request for completion."""

    prompt: str
    conversation: list[Message] | None = None


class CompletionResponse(BaseModel):
    """Response from completion."""

    content: str


class StreamResponse(BaseModel):
    """Streaming response chunk."""

    chunk: str
    done: bool = False
    error: str | None = None


def format_conversation(
    prompt: str,
    conversation: list[Message] | None = None,
) -> str:
    """Format conversation for display to operator."""
    lines = []

    if conversation:
        for msg in conversation:
            prefix = "👤" if msg.role == "user" else "🤖"
            lines.append(f"{prefix} {msg.content}")

    lines.append("\n>>> Current prompt:")
    lines.append(f"👤 {prompt}")
    lines.append("\nYour response: ")

    return "\n".join(lines)


class ModelServer:
    """Server that delegates to human operator."""

    def __init__(self, title: str = "Input Server", description: str | None = None):
        """Initialize server with configuration."""
        self.app = FastAPI(title=title, description=description or "No description")
        self._setup_routes()

    def _setup_routes(self):
        """Configure API routes."""

        @self.app.post(
            "/v1/chat/completions",
            response_model=CompletionResponse,
        )
        async def create_completion(
            request: CompletionRequest,
            auth: Annotated[HTTPAuthorizationCredentials, security],
        ) -> CompletionResponse:
            """Handle completion requests via REST."""
            try:
                # Display conversation and prompt to operator
                print("\n" + "=" * 80)
                print(format_conversation(request.prompt, request.conversation))

                # Get operator's response
                response = input().strip()

                return CompletionResponse(content=response)

            except Exception as e:
                logger.exception("Error processing completion request")
                code = status.HTTP_500_INTERNAL_SERVER_ERROR
                raise HTTPException(code, detail=str(e)) from e

        @self.app.websocket("/v1/chat/stream")
        async def websocket_endpoint(websocket: WebSocket):
            """Handle streaming conversation via WebSocket."""
            await websocket.accept()

            try:
                while True:
                    # Receive and parse request
                    raw_message = await websocket.receive_text()
                    request = CompletionRequest.model_validate_json(raw_message)

                    # Display to operator
                    print("\n" + "=" * 80)
                    print(format_conversation(request.prompt, request.conversation))

                    # Get response character by character
                    print("Type your response (press Enter when done):")
                    buffer = []
                    while True:
                        char = input()  # This is synchronous - see note below
                        if not char:  # Enter pressed
                            break

                        buffer.append(char)
                        # Send character as stream chunk
                        await websocket.send_json(StreamResponse(chunk=char).model_dump())

                    # Send completion
                    data = StreamResponse(chunk="", done=True).model_dump()
                    await websocket.send_json(data)

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs: Any):
        """Start the server."""
        import uvicorn

        uvicorn.run(self.app, host=host, port=port, **kwargs)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    server = ModelServer(
        title="Remote Input Server",
        description="Server that delegates to human operator",
    )
    server.run(port=8000)
