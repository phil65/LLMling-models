"""OpenAI-compatible API server for Pydantic-AI models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field
from schemez import Schema

from llmling_models.log import get_logger


logger = get_logger(__name__)


class OpenAIModelInfo(Schema):
    """OpenAI model info format."""

    id: str
    object: str = "model"
    owned_by: str = "llmling"
    created: int
    description: str | None = None
    permissions: list[str] = Field(default_factory=list)


class FunctionCall(Schema):
    """Function call information."""

    name: str
    arguments: str


class ToolCall(Schema):
    """Tool call information."""

    id: str
    type: str = "function"
    function: FunctionCall


class OpenAIMessage(Schema):
    """OpenAI chat message format."""

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str | None  # Content can be null in function calls
    name: str | None = None
    function_call: FunctionCall | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool responses


class FunctionDefinition(Schema):
    """Function definition."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolDefinitionSchema(Schema):
    """Tool definition schema."""

    type: str = "function"
    function: FunctionDefinition


class ChatCompletionRequest(Schema):
    """OpenAI chat completion request."""

    model: str
    messages: list[OpenAIMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[ToolDefinitionSchema] | None = None
    tool_choice: str | dict[str, Any] | None = Field(default="auto")
    response_format: dict[str, str] | None = None


class Choice(Schema):
    """Choice in a completion response."""

    index: int = 0
    message: OpenAIMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(Schema):
    """OpenAI chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: dict[str, int] | None = None


class ChatCompletionChunk(Schema):
    """Chunk of a streaming chat completion."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[dict[str, Any]]
