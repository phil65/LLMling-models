"""OpenAI-compatible API server for Pydantic-AI models."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
import uuid

import anyenv
from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import ToolDefinition

from llmling_models.log import get_logger
from llmling_models.openai_server.models import (
    ChatCompletionResponse,
    Choice,
    FunctionCall,
    OpenAIMessage,
    ToolCall,
)


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from pydantic_ai import ModelMessage, ModelResponsePart

    from llmling_models.openai_server.models import (
        ToolDefinitionSchema,
    )


logger = get_logger(__name__)


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
    stream: AsyncGenerator[str],
    allow_tools: bool = True,
) -> AsyncGenerator[str]:
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
