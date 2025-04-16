"""OpenAI-compatible API server for Pydantic-AI models."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast
import uuid

import anyenv
from fastapi import Depends, FastAPI, Header, HTTPException, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, create_model
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    PartDeltaEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from llmling_models.log import get_logger
from llmling_models.utils import infer_model


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


logger = get_logger(__name__)


# OpenAI API Models
class OpenAIModelInfo(BaseModel):
    """OpenAI model info format."""

    id: str
    object: str = "model"
    owned_by: str = "llmling"
    created: int
    description: str | None = None
    permissions: list[str] = []


class FunctionCall(BaseModel):
    """Function call information."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call information."""

    id: str
    type: str = "function"
    function: FunctionCall


class OpenAIMessage(BaseModel):
    """OpenAI chat message format."""

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str | None  # Content can be null in function calls
    name: str | None = None
    function_call: FunctionCall | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool responses


class FunctionDefinition(BaseModel):
    """Function definition."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolDefinitionSchema(BaseModel):
    """Tool definition schema."""

    type: str = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request."""

    model: str
    messages: list[OpenAIMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[ToolDefinitionSchema] | None = None
    tool_choice: str | dict[str, Any] | None = Field(default="auto")
    response_format: dict[str, str] | None = None


class Choice(BaseModel):
    """Choice in a completion response."""

    index: int = 0
    message: OpenAIMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: dict[str, int] | None = None


class ChatCompletionChunk(BaseModel):
    """Chunk of a streaming chat completion."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[dict[str, Any]]


# Conversion functions
def openai_to_pydantic_messages(messages: list[OpenAIMessage]) -> list[ModelMessage]:
    """Convert OpenAI messages to Pydantic-AI messages."""
    result: list[ModelMessage] = []

    for message in messages:
        if message.role == "system":
            result.append(
                ModelRequest(parts=[SystemPromptPart(content=message.content or "")])
            )

        elif message.role == "user":
            result.append(
                ModelRequest(parts=[UserPromptPart(content=message.content or "")])
            )

        elif message.role == "assistant":
            parts: list[ModelResponsePart] = []
            if message.content:
                parts.append(TextPart(content=message.content))

            if message.function_call:
                parts.append(
                    ToolCallPart(
                        tool_name=message.function_call.name,
                        args=message.function_call.arguments,
                        tool_call_id=str(uuid.uuid4()),
                    )
                )

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    parts.append(  # noqa: PERF401
                        ToolCallPart(
                            tool_name=tool_call.function.name,
                            args=tool_call.function.arguments,
                            tool_call_id=tool_call.id,
                        )
                    )

            if parts:
                result.append(ModelResponse(parts=parts))

        elif message.role in ("tool", "function"):
            if not message.tool_call_id:
                logger.warning("Tool message without tool_call_id, skipping: %s", message)
                continue
            # This is a fix - assuming we know the tool_name from context
            # Since OpenAI API doesn't provide tool_name in tool response
            # we'll use the content as tool_name as a fallback
            tool_name = message.name or f"tool_{message.tool_call_id}"
            part = ToolReturnPart(
                tool_name=tool_name,
                content=message.content or "",
                tool_call_id=message.tool_call_id,
            )
            result.append(ModelRequest(parts=[part]))

    return result


def convert_tools(tools: list[ToolDefinitionSchema]) -> list[ToolDefinition]:
    """Convert OpenAI tool definitions to Pydantic-AI tool definitions."""
    result = []

    for tool in tools:
        if tool.type != "function":
            logger.warning("Skipping unsupported tool type: %s", tool.type)
            continue
        defn = ToolDefinition(
            name=tool.function.name,
            description=tool.function.description,
            parameters_json_schema=tool.function.parameters,
        )
        result.append(defn)

    return result


def pydantic_response_to_openai(
    response: ModelResponse, model_name: str, allow_tools: bool = True
) -> ChatCompletionResponse:
    """Convert Pydantic-AI response to OpenAI format."""
    # Extract content and tool calls
    content_parts = []
    tool_call_parts = []

    for part in response.parts:
        if isinstance(part, TextPart):
            content_parts.append(part)
        elif isinstance(part, ToolCallPart) and allow_tools:
            tool_call_parts.append(part)

    # Combine content parts
    content = (
        "".join(str(part.content) for part in content_parts) if content_parts else None
    )

    # Create message
    message = OpenAIMessage(role="assistant", content=content)

    # Add tool calls if present
    if tool_call_parts:
        tool_calls = []
        for part in tool_call_parts:
            fn = FunctionCall(name=part.tool_name, arguments=part.args_as_json_str())
            id_ = part.tool_call_id or str(uuid.uuid4())
            call = ToolCall(id=id_, type="function", function=fn)
            tool_calls.append(call)
        message.tool_calls = tool_calls

    # Create completion response
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time() * 1000)}",
        created=int(time.time()),
        model=model_name,
        choices=[Choice(message=message)],
        usage={
            "prompt_tokens": 0,  # These will be populated later
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )


async def generate_stream_chunks(
    response_id: str,
    model_name: str,
    stream: AsyncGenerator[str, None],
    allow_tools: bool = True,
) -> AsyncGenerator[str, None]:
    """Generate streaming response chunks in OpenAI format."""
    created = int(time.time())

    # First chunk with role
    first_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {anyenv.dump_json(first_chunk)}\n\n"

    # Tool call tracking
    tool_calls: dict[str, dict[str, Any]] = {}
    sending_tool_calls = False
    content_complete = False

    # Process content chunks
    async for chunk in stream:
        if not chunk and content_complete:
            continue

        # If we've started sending tool calls and get more content,
        # we need to finish the tool calls first
        if sending_tool_calls:
            if tool_calls:
                # Send final tool call chunk
                for i, tool_call in enumerate(tool_calls.values()):
                    chunk_data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": i,
                                            "id": tool_call["id"],
                                            "type": "function",
                                            "function": tool_call["function"],
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {anyenv.dump_json(chunk_data)}\n\n"
                tool_calls = {}

            sending_tool_calls = False

        # Regular content chunk
        if chunk:
            content_complete = False
            chunk_data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": {"content": chunk}, "finish_reason": None}
                ],
            }
            yield f"data: {anyenv.dump_json(chunk_data)}\n\n"
        else:
            content_complete = True

    # Final chunk
    final_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {anyenv.dump_json(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


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
            import tokonomics

            all_models = await tokonomics.get_all_models()

            for model_info in all_models:
                try:
                    # Use the pydantic_model_id directly as the key
                    model_id = model_info.pydantic_ai_id
                    registry.models[model_id] = infer_model(model_id)
                    logger.debug("Auto-registered model: %s", model_id)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Failed to register model %s: %s",
                        model_info.pydantic_ai_id,
                        str(e),
                    )

            logger.info("Auto-populated %d models from tokonomics", len(registry.models))

        except ImportError:
            logger.warning("tokonomics not available, no models auto-populated")
        except Exception as e:  # noqa: BLE001
            logger.warning("Error auto-populating models: %s", str(e))

        return registry

    def add_model(self, name: str, model_or_id: str | Model) -> None:
        """Add a model to the registry."""
        if isinstance(model_or_id, str):
            self.models[name] = infer_model(model_or_id)
        else:
            self.models[name] = model_or_id

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

        # Chat completions endpoints
        self.app.post(
            "/v1/chat/completions",
            dependencies=[Depends(self.verify_api_key)] if self.api_key else None,
        )(self.create_chat_completion)

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
            response, usage = await model.request(
                messages,
                model_settings=cast(ModelSettings, settings),
                model_request_parameters=request_params,
            )

            # Convert to OpenAI format
            openai_response = pydantic_response_to_openai(
                response, request.model, allow_tools=bool(request.tools)
            )

            # Add usage information
            if openai_response.usage and usage:
                openai_response.usage.update({
                    "prompt_tokens": usage.request_tokens or 0,
                    "completion_tokens": usage.response_tokens or 0,
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
    ) -> AsyncGenerator[str, None]:
        """Stream response in OpenAI format."""
        response_id = f"chatcmpl-{int(time.time() * 1000)}"

        try:
            async with model.request_stream(
                messages,
                model_settings=cast(ModelSettings, settings),
                model_request_parameters=request_params,
            ) as stream:
                # First chunk with role
                first_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {anyenv.dump_json(first_chunk)}\n\n"

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
                                    chunk_data = {
                                        "id": response_id,
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model_name,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"content": delta},
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                    yield f"data: {anyenv.dump_json(chunk_data)}\n\n"

                    elif isinstance(event, PartDeltaEvent) and isinstance(
                        event.delta, TextPartDelta
                    ):
                        delta = event.delta.content_delta
                        content_buffer += delta

                        if delta:
                            chunk_data = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": delta},
                                        "finish_reason": None,
                                    }
                                ],
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
            error_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"Error: {e!s}"},
                        "finish_reason": "error",
                    }
                ],
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
                            # First chunk with role
                            first_chunk = {
                                "id": response_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"role": "assistant"},
                                        "finish_reason": None,
                                    }
                                ],
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
                                                chunk_data = {
                                                    "id": response_id,
                                                    "object": "chat.completion.chunk",
                                                    "created": int(time.time()),
                                                    "model": request.model,
                                                    "choices": [
                                                        {
                                                            "index": 0,
                                                            "delta": {"content": delta},
                                                            "finish_reason": None,
                                                        }
                                                    ],
                                                }
                                                await websocket.send_json(chunk_data)

                                elif isinstance(event, PartDeltaEvent) and isinstance(
                                    event.delta, TextPartDelta
                                ):
                                    delta = event.delta.content_delta
                                    content_buffer += delta

                                    if delta:
                                        chunk_data = {
                                            "id": response_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": request.model,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {"content": delta},
                                                    "finish_reason": None,
                                                }
                                            ],
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
                        response, usage = await model.request(
                            messages,
                            model_settings=cast(ModelSettings, settings),
                            model_request_parameters=request_params,
                        )

                        # Convert to OpenAI format
                        openai_response = pydantic_response_to_openai(
                            response, request.model, allow_tools=bool(request.tools)
                        )

                        # Add usage information
                        if openai_response.usage and usage:
                            openai_response.usage.update({
                                "prompt_tokens": usage.request_tokens or 0,
                                "completion_tokens": usage.response_tokens or 0,
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
