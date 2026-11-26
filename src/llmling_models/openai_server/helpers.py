"""OpenAI-compatible API server for Pydantic-AI models."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
import uuid

import anyenv
from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
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
    from pydantic_ai import ModelMessage, ModelResponsePart, ModelResponseStreamEvent

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
            result.append(ModelRequest(parts=[SystemPromptPart(content=message.content or "")]))

        elif message.role == "user":
            result.append(ModelRequest(parts=[UserPromptPart(content=message.content or "")]))

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
    content = "".join(str(part.content) for part in content_parts) if content_parts else None

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
        choices=[
            Choice(message=message, finish_reason="tool_calls" if tool_call_parts else "stop")
        ],
        usage={
            "prompt_tokens": 0,  # These will be populated later
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )


class StreamDeltaConverter:
    """Turn pydantic-ai stream events into OpenAI `chat.completion.chunk` deltas.

    Keeps per-part state: each text part is tracked by its part index (a later text part
    must not be sliced by an earlier one's length), and each tool call part gets its own
    OpenAI `tool_calls` index.
    """

    def __init__(self, allow_tools: bool = True) -> None:
        self.allow_tools = allow_tools
        self._text_lengths: dict[int, int] = {}
        self._tool_indices: dict[int, int] = {}

    @property
    def finish_reason(self) -> str:
        return "tool_calls" if self._tool_indices else "stop"

    def convert(self, event: ModelResponseStreamEvent) -> dict[str, Any] | None:
        """Return the OpenAI delta for an event, or None if there is nothing to send."""
        if isinstance(event, PartStartEvent):
            part = event.part
            if isinstance(part, TextPart):
                seen = self._text_lengths.get(event.index, 0)
                self._text_lengths[event.index] = len(part.content)
                new = part.content[seen:]
                return {"content": new} if new else None
            if isinstance(part, ToolCallPart) and self.allow_tools:
                index = self._tool_indices.setdefault(event.index, len(self._tool_indices))
                args = part.args_as_json_str() if part.args else ""
                call = {
                    "index": index,
                    "id": part.tool_call_id or str(uuid.uuid4()),
                    "type": "function",
                    "function": {"name": part.tool_name, "arguments": args},
                }
                return {"tool_calls": [call]}
        elif isinstance(event, PartDeltaEvent):
            delta = event.delta
            if isinstance(delta, TextPartDelta):
                self._text_lengths[event.index] = self._text_lengths.get(event.index, 0) + len(
                    delta.content_delta
                )
                return {"content": delta.content_delta} if delta.content_delta else None
            if isinstance(delta, ToolCallPartDelta) and event.index in self._tool_indices:
                function: dict[str, str] = {}
                if delta.tool_name_delta:
                    function["name"] = delta.tool_name_delta
                if delta.args_delta:
                    args_delta = delta.args_delta
                    function["arguments"] = (
                        args_delta if isinstance(args_delta, str) else anyenv.dump_json(args_delta)
                    )
                if not function:
                    return None
                call = {"index": self._tool_indices[event.index], "function": function}
                return {"tool_calls": [call]}
        return None
