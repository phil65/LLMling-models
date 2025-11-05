"""OpenAI-compatible API server for Pydantic-AI models."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Annotated, Any, cast

import anyenv
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import create_model
from pydantic_ai import (
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
import tokonomics

from llmling_models.log import get_logger
from llmling_models.openai_server.helpers import (
    convert_tools,
    openai_to_pydantic_messages,
    pydantic_response_to_openai,
)
from llmling_models.openai_server.models import ChatCompletionRequest, OpenAIModelInfo
from llmling_models.utils import infer_model


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fastapi import WebSocket
    from pydantic_ai import (
        ModelMessage,
    )
    from pydantic_ai.models import Model


logger = get_logger(__name__)


class ModelRegistry:
    """Registry of available models."""

    def __init__(self, models: dict[str, str | Model] | None = None):
        """Initialize model registry.

        Args:
            models: Dictionary mapping model names to models or model identifiers
        """
        self.models: dict[str, Model] = {}
        if models:
            for name, model_or_id in models.items():
                if isinstance(model_or_id, str):
                    self.models[name] = infer_model(model_or_id)
                else:
                    self.models[name] = model_or_id

    @classmethod
    async def create(cls) -> ModelRegistry:
        """Create a model registry populated with all models from tokonomics.

        Returns:
            A new ModelRegistry instance with auto-populated models.
        """
        registry = cls({})  # Empty registry
        try:
            all_models = await tokonomics.get_all_models()
            for model_info in all_models:
                try:
                    # Use the pydantic_model_id directly as the key
                    model_id = model_info.pydantic_ai_id
                    registry.models[model_id] = infer_model(model_id)
                    logger.debug("Auto-registered model: %s", model_id)
                except Exception as e:  # noqa: BLE001
                    msg = "Failed to register model %s: %s"
                    logger.warning(msg, model_info.pydantic_ai_id, str(e))

            logger.info("Auto-populated %d models from tokonomics", len(registry.models))
        except Exception as e:  # noqa: BLE001
            logger.warning("Error auto-populating models: %s", str(e))

        return registry

    def add_model(self, name: str, model_or_id: str | Model) -> None:
        """Add a model to the registry."""
        model = infer_model(model_or_id) if isinstance(model_or_id, str) else model_or_id
        self.models[name] = model

    def get_model(self, name: str) -> Model:
        """Get a model by name."""
        try:
            return self.models[name]
        except KeyError:
            msg = f"Model {name} not found"
            raise ValueError(msg) from None

    def list_models(self) -> list[OpenAIModelInfo]:
        """List available models."""
        return [
            OpenAIModelInfo(id=n, created=int(time.time()), description=f"Model {n}")
            for n in self.models
        ]


class OpenAIServer:
    """OpenAI-compatible API server backed by Pydantic-AI models."""

    def __init__(
        self,
        registry: ModelRegistry,
        api_key: str | None = None,
        title: str = "LLMling OpenAI-Compatible API",
        description: str | None = None,
    ):
        """Initialize the server.

        Args:
            registry: Model registry
            api_key: API key for authentication (None to disable auth)
            title: API title
            description: API description
        """
        self.registry = registry
        self.api_key = api_key
        self.app = FastAPI(title=title, description=description or "")

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.setup_routes()

    def verify_api_key(
        self, authorization: Annotated[str | None, Header(alias="Authorization")] = None
    ) -> None:
        """Verify API key if configured."""
        if not self.api_key:
            return

        if not authorization:
            raise HTTPException(401, "Missing API key")
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "Invalid authorization format")

        token = authorization.removeprefix("Bearer ")
        if token != self.api_key:
            raise HTTPException(401, "Invalid API key")

    def setup_routes(self) -> None:
        """Configure API routes."""
        # List models endpoint
        self.app.get(
            "/v1/models",
            dependencies=[Depends(self.verify_api_key)] if self.api_key else None,
        )(self.list_models)
        deps = [Depends(self.verify_api_key)] if self.api_key else None
        self.app.post("/v1/chat/completions", dependencies=deps)(
            self.create_chat_completion
        )

        # WebSocket endpoint for chat completions
        if self.api_key:
            self.app.websocket("/v1/chat/completions/ws")(self.websocket_chat_completion)

        # Add common OpenAI endpoints (stubs)
        self.app.get("/v1/dashboard/billing/subscription")(self.stub_billing)
        self.app.get("/v1/dashboard/billing/usage")(self.stub_usage)

        # Health check endpoint
        self.app.get("/health")(self.health_check)

    async def list_models(self) -> dict[str, Any]:
        """List available models."""
        models = self.registry.list_models()
        return {"object": "list", "data": models}

    async def create_chat_completion(self, request: ChatCompletionRequest) -> Response:
        """Handle chat completion requests."""
        try:
            # Get model
            try:
                model = self.registry.get_model(request.model)
            except ValueError:
                raise HTTPException(404, f"Model {request.model} not found") from None

            # Convert messages
            messages = openai_to_pydantic_messages(request.messages)

            # Create settings
            settings_data: dict[str, Any] = {}
            if request.temperature is not None:
                settings_data["temperature"] = request.temperature
            if request.max_tokens is not None:
                settings_data["max_tokens"] = request.max_tokens

            settings = (
                create_model("ModelSettings", **settings_data)()
                if settings_data
                else None
            )

            # Handle function/tool calls
            function_tools = []
            if request.tools:
                function_tools = convert_tools(request.tools)

            # Determine if we should force tool usage
            allow_text_output = True
            if request.tool_choice and request.tool_choice != "auto":
                allow_text_output = False

            # Prepare request parameters
            request_params = ModelRequestParameters(
                function_tools=function_tools,
                allow_text_output=allow_text_output,
                output_tools=[],  # Not used in OpenAI API
            )

            # Check if streaming is requested
            if request.stream:
                return StreamingResponse(
                    self._stream_response(
                        model, messages, settings, request_params, request.model
                    ),
                    media_type="text/event-stream",
                )

            # Non-streaming response
            response = await model.request(
                messages,
                model_settings=cast(ModelSettings, settings),
                model_request_parameters=request_params,
            )

            # Convert to OpenAI format
            openai_response = pydantic_response_to_openai(
                response, request.model, allow_tools=bool(request.tools)
            )

            # Add usage information
            usage = response.usage
            if openai_response.usage and usage:
                openai_response.usage.update({
                    "prompt_tokens": usage.input_tokens or 0,
                    "completion_tokens": usage.output_tokens or 0,
                    "total_tokens": usage.total_tokens or 0,
                })

            return Response(
                content=openai_response.model_dump_json(),
                media_type="application/json",
            )

        except Exception as e:
            logger.exception("Error processing chat completion")
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "param": None,
                    "code": "internal_error",
                }
            }
            return Response(
                content=anyenv.dump_json(error_response),
                status_code=500,
                media_type="application/json",
            )

    async def _stream_response(
        self,
        model: Model,
        messages: list[ModelMessage],
        settings: Any,
        request_params: ModelRequestParameters,
        model_name: str,
    ) -> AsyncGenerator[str]:
        """Stream response in OpenAI format."""
        response_id = f"chatcmpl-{int(time.time() * 1000)}"

        try:
            async with model.request_stream(
                messages,
                model_settings=cast(ModelSettings, settings),
                model_request_parameters=request_params,
            ) as stream:
                # First chunk with role
                choice = {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
                first_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [choice],
                }
                yield f"data: {anyenv.dump_json(first_chunk)}\n\n"

                # Process stream events
                content_buffer = ""
                async for event in stream:
                    if isinstance(event, PartStartEvent):
                        if isinstance(event.part, TextPart):
                            new_content = str(event.part.content)
                            if new_content != content_buffer:
                                delta = new_content[len(content_buffer) :]
                                content_buffer = new_content

                                if delta:
                                    choice = {
                                        "index": 0,
                                        "delta": {"content": delta},
                                        "finish_reason": None,
                                    }
                                    chunk_data = {
                                        "id": response_id,
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model_name,
                                        "choices": [choice],
                                    }
                                    yield f"data: {anyenv.dump_json(chunk_data)}\n\n"

                    elif isinstance(event, PartDeltaEvent) and isinstance(
                        event.delta, TextPartDelta
                    ):
                        delta = event.delta.content_delta
                        content_buffer += delta

                        if delta:
                            choice = {
                                "index": 0,
                                "delta": {"content": delta},
                                "finish_reason": None,
                            }
                            chunk_data = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model_name,
                                "choices": [choice],
                            }
                            yield f"data: {anyenv.dump_json(chunk_data)}\n\n"

                # Final chunk
                final_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {anyenv.dump_json(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        except Exception as e:
            logger.exception("Error during streaming response")
            choice = {
                "index": 0,
                "delta": {"content": f"Error: {e!s}"},
                "finish_reason": "error",
            }
            error_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [choice],
            }
            yield f"data: {anyenv.dump_json(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    async def websocket_chat_completion(self, websocket: WebSocket) -> None:
        """Handle WebSocket chat completions."""
        await websocket.accept()

        try:
            # Verify authentication
            auth = websocket.headers.get("Authorization")
            try:
                self.verify_api_key(auth)
            except HTTPException as e:
                await websocket.send_json({"error": e.detail})
                await websocket.close(code=e.status_code)
                return

            while True:
                # Receive request
                data = await websocket.receive_text()
                request = ChatCompletionRequest.model_validate_json(data)

                # Get model
                try:
                    model = self.registry.get_model(request.model)
                except ValueError:
                    await websocket.send_json({
                        "error": f"Model {request.model} not found"
                    })
                    continue

                # Convert messages
                messages = openai_to_pydantic_messages(request.messages)

                # Create settings
                settings_data: dict[str, Any] = {}
                if request.temperature is not None:
                    settings_data["temperature"] = request.temperature
                if request.max_tokens is not None:
                    settings_data["max_tokens"] = request.max_tokens

                settings = (
                    create_model("ModelSettings", **settings_data)()
                    if settings_data
                    else None
                )

                # Handle function/tool calls
                function_tools = []
                if request.tools:
                    function_tools = convert_tools(request.tools)

                # Determine if we should force tool usage
                allow_text_output = True
                if request.tool_choice and request.tool_choice != "auto":
                    allow_text_output = False

                # Prepare request parameters
                request_params = ModelRequestParameters(
                    function_tools=function_tools,
                    allow_text_output=allow_text_output,
                    output_tools=[],  # Not used in OpenAI API
                )

                # Process request with streaming
                response_id = f"chatcmpl-{int(time.time() * 1000)}"

                try:
                    # Stream response
                    if request.stream:
                        async with model.request_stream(
                            messages,
                            model_settings=cast(ModelSettings, settings),
                            model_request_parameters=request_params,
                        ) as stream:
                            choice = {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                            first_chunk = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [choice],
                            }
                            await websocket.send_json(first_chunk)

                            # Process stream events
                            content_buffer = ""
                            async for event in stream:
                                if isinstance(event, PartStartEvent):
                                    # Handle new part
                                    if isinstance(event.part, TextPart):
                                        new_content = str(event.part.content)
                                        if new_content != content_buffer:
                                            delta = new_content[len(content_buffer) :]
                                            content_buffer = new_content

                                            if delta:
                                                choice = {
                                                    "index": 0,
                                                    "delta": {"content": delta},
                                                    "finish_reason": None,
                                                }
                                                chunk_data = {
                                                    "id": response_id,
                                                    "object": "chat.completion.chunk",
                                                    "created": int(time.time()),
                                                    "model": request.model,
                                                    "choices": [choice],
                                                }
                                                await websocket.send_json(chunk_data)

                                elif isinstance(event, PartDeltaEvent) and isinstance(
                                    event.delta, TextPartDelta
                                ):
                                    delta = event.delta.content_delta
                                    content_buffer += delta

                                    if delta:
                                        choice = {
                                            "index": 0,
                                            "delta": {"content": delta},
                                            "finish_reason": None,
                                        }
                                        chunk_data = {
                                            "id": response_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": request.model,
                                            "choices": [choice],
                                        }
                                        await websocket.send_json(chunk_data)

                            # Final chunk
                            final_chunk = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [
                                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                                ],
                            }
                            await websocket.send_json(final_chunk)
                            await websocket.send_json({"done": True})

                    # Non-streaming response
                    else:
                        response = await model.request(
                            messages,
                            model_settings=cast(ModelSettings, settings),
                            model_request_parameters=request_params,
                        )

                        # Convert to OpenAI format
                        openai_response = pydantic_response_to_openai(
                            response, request.model, allow_tools=bool(request.tools)
                        )

                        # Add usage information
                        usage = response.usage
                        if openai_response.usage and usage:
                            openai_response.usage.update({
                                "prompt_tokens": usage.input_tokens or 0,
                                "completion_tokens": usage.output_tokens or 0,
                                "total_tokens": usage.total_tokens or 0,
                            })

                        # Send response
                        await websocket.send_json(openai_response.model_dump())

                except Exception as e:
                    logger.exception("Error processing WebSocket request")
                    error_response = {
                        "error": {
                            "message": str(e),
                            "type": "server_error",
                            "param": None,
                            "code": "internal_error",
                        }
                    }
                    await websocket.send_json(error_response)

        except Exception as e:
            logger.exception("WebSocket error")
            with contextlib.suppress(RuntimeError):
                await websocket.send_json({"error": str(e)})
        finally:
            with contextlib.suppress(RuntimeError):
                await websocket.close()

    async def stub_billing(self) -> dict[str, Any]:
        """Stub billing endpoint."""
        return {
            "object": "billing_subscription",
            "has_payment_method": True,
            "canceled": False,
            "canceled_at": None,
            "delinquent": None,
            "access_until": int(time.time() + 31536000),  # 1 year from now
            "soft_limit": 10000,
            "hard_limit": 100000,
            "system_hard_limit": 100000,
        }

    async def stub_usage(self) -> dict[str, Any]:
        """Stub usage endpoint."""
        return {"object": "list", "data": [], "total_usage": 0}

    async def health_check(self) -> dict[str, bool]:
        """Health check endpoint."""
        return {"status": True}


async def run_server(
    models: dict[str, str | Model],
    host: str = "0.0.0.0",
    port: int = 8000,
    api_key: str | None = None,
) -> None:
    """Run the OpenAI-compatible API server."""
    import uvicorn

    logger.info("Starting OpenAI-compatible API server...")
    logger.info("Available models: %s", list(models.keys()))

    registry = ModelRegistry(models)
    server = OpenAIServer(
        registry=registry,
        api_key=api_key,
        title="LLMling OpenAI-Compatible API",
        description="OpenAI-compatible API server powered by LLMling models",
    )

    config = uvicorn.Config(app=server.app, host=host, port=port, log_level="info")
    server_instance = uvicorn.Server(config)
    await server_instance.serve()


if __name__ == "__main__":
    import logging

    import uvicorn

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

    async def run_with_auto_discovery():
        """Run the server with auto-discovered models from tokonomics."""
        registry = await ModelRegistry.create()
        server = OpenAIServer(
            registry=registry,
            api_key="test-key",
            title="LLMling OpenAI-Compatible API",
            description="OpenAI-compatible API server powered by LLMling models",
        )
        config = uvicorn.Config(
            app=server.app, host="0.0.0.0", port=8000, log_level="info"
        )
        server_instance = uvicorn.Server(config)
        await server_instance.serve()

    asyncio.run(run_with_auto_discovery())
