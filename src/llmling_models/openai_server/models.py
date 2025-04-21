"""OpenAI-compatible API server for Pydantic-AI models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from llmling_models.log import get_logger


logger = get_logger(__name__)


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
