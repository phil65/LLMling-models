from __future__ import annotations

import contextlib
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse

from llmling_models.log import get_logger


if TYPE_CHECKING:
    from pydantic_ai.models import Model


logger = get_logger(__name__)


class ModelServer:
    """FastAPI server that serves a pydantic-ai model."""

    def __init__(
        self,
        model: Model,
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
        self.model = model
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
                # Initialize model if needed
                agent_model = await self.model.agent_model(
                    function_tools=[],
                    allow_text_result=True,
                    result_tools=[],
                )
                # Get response
                response, usage = await agent_model.request(
                    messages,
                    model_settings=None,  # Add this parameter
                )
                content = (
                    str(response.parts[0].content)  # pyright: ignore
                    if hasattr(response.parts[0], "content")
                    else ""
                )
                return {"content": content, "usage": asdict(usage)}
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

                # Initialize model
                agent_model = await self.model.agent_model(
                    function_tools=[],
                    allow_text_result=True,
                    result_tools=[],
                )

                while True:
                    # Receive and parse messages
                    try:
                        data = await websocket.receive()
                        logger.debug("Received WebSocket data: %s", data)

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
                        logger.debug("Parsed messages: %s", messages)

                        # Get streaming response
                        async with agent_model.request_stream(
                            messages,
                            model_settings=None,
                        ) as stream:
                            content_sent = False

                            # Get initial response chunks
                            chunks = stream.get()
                            if chunks:
                                content_sent = True
                                chunk_text = (
                                    str(chunks.parts[0].content)  # type: ignore
                                    if isinstance(chunks, ModelResponse)
                                    else "".join(chunks)
                                )
                                await websocket.send_json({
                                    "chunk": chunk_text,
                                    "done": False,
                                })

                            # Stream remaining chunks
                            async for chunk in stream:
                                if chunk:  # Skip empty chunks
                                    content_sent = True
                                    await websocket.send_json({
                                        "chunk": chunk,
                                        "done": False,
                                    })

                            # Get final chunks if any
                            final_chunks = stream.get(final=True)
                            if final_chunks:
                                content_sent = True
                                final_text = (
                                    str(final_chunks.parts[0].content)  # type: ignore
                                    if isinstance(final_chunks, ModelResponse)
                                    else "".join(final_chunks)
                                )
                                await websocket.send_json({
                                    "chunk": final_text,
                                    "done": False,
                                })
                            # If no content was sent, send error
                            if not content_sent:
                                await websocket.send_json({
                                    "error": "Model returned no content",
                                    "done": True,
                                })
                                continue

                            # Send completion with usage info
                            await websocket.send_json({
                                "chunk": "",
                                "done": True,
                                "usage": asdict(stream.usage()),
                            })

                    except Exception as e:
                        logger.exception("Error processing message")
                        await websocket.send_json({
                            "error": str(e),
                            "done": True,
                        })
                        break

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
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

    from pydantic_ai.models import infer_model

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting model server...")

    # Create server with a model
    server = ModelServer(
        model=infer_model("openai:gpt-4o-mini"),
        api_key="test-key",  # Enable authentication
        title="Test Model Server",
        description="Test server serving GPT-4-mini",
    )

    # Run server
    logger.info("Server running at http://localhost:8000")
    server.run(host="localhost", port=8000)
