"""FastAPI server that serves a pydantic-ai model."""

from __future__ import annotations

import contextlib
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Header, HTTPException, WebSocketDisconnect, status
from pydantic_ai import ModelMessagesTypeAdapter, ModelResponse
from pydantic_ai.models import ModelRequestParameters

from llmling_models.log import get_logger
from llmling_models.utils import infer_model


if TYPE_CHECKING:
    from fastapi import WebSocket
    from pydantic_ai import ModelMessage
    from pydantic_ai.models import Model

logger = get_logger(__name__)


class ModelServer:
    """FastAPI server that serves a pydantic-ai model."""

    def __init__(
        self,
        model: Model | str,
        *,
        title: str = "Model Server",
        description: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize server with a pydantic-ai model.

        Args:
            model: The model to serve
            title: Server title for OpenAPI docs
            description: Server description
            api_key: Optional API key for authentication
        """
        self.app = FastAPI(title=title, description=description or "")
        self.model = infer_model(model)
        self.api_key = api_key
        self._setup_routes()

    def _verify_auth(self, auth: str | None) -> None:
        """Verify authentication header if API key is set."""
        if not self.api_key:
            return
        if not auth or not auth.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication header",
            )
        token = auth.removeprefix("Bearer ")
        if token != self.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

    def _setup_routes(self):
        """Configure API routes."""

        @self.app.post("/v1/completion")
        async def create_completion(
            messages: list[ModelMessage],
            auth: str | None = Header(None, alias="Authorization"),
        ) -> dict[str, Any]:
            """Handle completion requests via REST."""
            try:
                self._verify_auth(auth)

                # Create model parameters
                model_params = ModelRequestParameters(
                    function_tools=[],
                    allow_text_output=True,
                    output_tools=[],
                )

                # Get response
                response = await self.model.request(
                    messages,
                    model_settings=None,
                    model_request_parameters=model_params,
                )
                content = (
                    str(response.parts[0].content)  # type: ignore
                    if hasattr(response.parts[0], "content")
                    else ""
                )
                return {"content": content, "usage": asdict(response.usage)}

            except Exception as e:
                logger.exception("Error processing completion request")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e),
                ) from e

        @self.app.websocket("/v1/completion/stream")
        async def websocket_endpoint(websocket: WebSocket):
            """Handle streaming conversation via WebSocket."""
            try:
                # Check auth
                auth = websocket.headers.get("Authorization")
                self._verify_auth(auth)

                # Accept connection
                await websocket.accept()
                logger.debug("WebSocket connection accepted")

                # Create model parameters
                model_params = ModelRequestParameters(
                    function_tools=[],
                    allow_text_output=True,
                    output_tools=[],
                )

                while True:
                    try:
                        data = await websocket.receive()
                        logger.debug("Received request data: %s", data)

                        if data["type"] == "websocket.disconnect":
                            break
                        if data["type"] != "websocket.receive":
                            continue

                        if "bytes" in data:
                            raw_messages = data["bytes"].decode("utf-8")
                        elif "text" in data:
                            raw_messages = data["text"]
                        else:
                            continue

                        messages = ModelMessagesTypeAdapter.validate_json(raw_messages)
                        logger.debug("Starting stream for messages: %s", messages)

                        # Use actual streaming from the model
                        async with self.model.request_stream(
                            messages,
                            model_settings=None,
                            model_request_parameters=model_params,
                        ) as stream:
                            logger.debug("Stream started")

                            # Stream chunks
                            async for _ in stream:
                                chunks = stream.get()
                                if isinstance(chunks, ModelResponse):
                                    # Handle ModelResponse
                                    if chunks.parts and hasattr(
                                        chunks.parts[0], "content"
                                    ):
                                        await websocket.send_json({
                                            "chunk": str(chunks.parts[0].content),  # type: ignore
                                            "done": False,
                                        })
                                else:
                                    # Handle Iterable[str]
                                    for chunk in chunks:  # pyright: ignore
                                        if chunk:  # Only send non-empty chunks
                                            await websocket.send_json({
                                                "chunk": chunk,
                                                "done": False,
                                            })

                            # Send completion with usage
                            await websocket.send_json({
                                "chunk": "",
                                "done": True,
                                "usage": asdict(stream.usage()),
                            })

                    except WebSocketDisconnect:
                        logger.info("WebSocket disconnected")
                        break

            except Exception as e:
                logger.exception("Error in WebSocket connection")
                with contextlib.suppress(WebSocketDisconnect):
                    await websocket.send_json({
                        "error": str(e),
                        "done": True,
                    })

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs: Any):
        """Start the server."""
        import uvicorn

        kwargs.pop("reload", None)
        kwargs.pop("workers", None)
        uvicorn.run(self.app, host=host, port=port, **kwargs)


if __name__ == "__main__":
    import logging

    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Starting model server...")

    # Create server with a model
    server = ModelServer(
        model="openai:gpt-5-nano",
        api_key="test-key",  # Enable authentication
        title="Test Model Server",
        description="Test server serving GPT-4-mini",
    )

    # Run server
    logger.info("Server running at http://localhost:8000")
    server.run(host="localhost", port=8000)
