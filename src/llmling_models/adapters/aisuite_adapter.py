"""Adapter to use AISuite library models with Pydantic-AI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic_ai import RequestUsage
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, StreamedResponse


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai import RunContext
    from pydantic_ai.messages import (
        ModelMessage,
        ModelResponseStreamEvent,
        UserContent,
    )
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition


def convert_content_item(item: UserContent) -> dict[str, Any]:  # noqa: PLR0911
    """Convert a single content item to AISuite format."""
    import base64

    match item:
        case str():
            return {"type": "text", "text": item}
        case ImageUrl():
            return {
                "type": "image_url",
                "image_url": {"url": item.url, "format": item.media_type},
            }
        case AudioUrl():
            # AudioUrl has media_type, not format
            audio_format = "mp3" if item.media_type == "audio/mpeg" else "wav"
            return {
                "type": "input_audio",
                "input_audio": {"url": item.url, "format": audio_format},
            }
        case DocumentUrl():
            return {
                "type": "image_url",
                "image_url": {"url": item.url, "format": item.media_type},
            }
        case BinaryContent():
            if item.is_image:
                encoded = base64.b64encode(item.data).decode("utf-8")
                base64_url = f"data:{item.media_type};base64,{encoded}"
                return {
                    "type": "image_url",
                    "image_url": {"url": base64_url, "format": item.media_type},
                }
            if item.is_audio:
                encoded = base64.b64encode(item.data).decode("utf-8")
                # Extract audio format from media_type
                audio_format = item.format  # BinaryContent does have format property
                return {
                    "type": "input_audio",
                    "input_audio": {"data": encoded, "format": audio_format},
                }
            if item.is_document:
                encoded = base64.b64encode(item.data).decode("utf-8")
                base64_url = f"data:{item.media_type};base64,{encoded}"
                return {
                    "type": "file",
                    "file": {
                        "filename": f"document.{item.format}",
                        "file_data": base64_url,
                    },
                }
            return {
                "type": "text",
                "text": f"[Unsupported binary content: {item.media_type}]",
            }
        case _:
            msg = f"Unsupported content type: {item}"
            raise ValueError(msg)


def convert_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert Pydantic-AI tool definitions to AISuite tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_json_schema,
            },
        }
        for tool in tools
    ]


@dataclass(kw_only=True)
class AISuiteStreamedResponse(StreamedResponse):
    """Stream implementation for AISuite."""

    _timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    """Timestamp of when the response was created."""

    def __post_init__(self):
        """Initialize usage."""
        self._usage = RequestUsage()  # Initialize with empty usage

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Not supported yet."""
        msg = "Streaming not supported by AISuite adapter"
        raise NotImplementedError(msg) from None
        # Need to yield even though we raise an error
        # to satisfy the async iterator protocol
        if False:  # pragma: no cover
            yield None  # type: ignore

    @property
    def timestamp(self) -> datetime:
        """Get response timestamp."""
        return self._timestamp

    @property
    def model_name(self) -> str:
        """Get response model_name."""
        return "aisuite"


@dataclass
class AISuiteAdapter(Model):
    """Adapter to use AISuite library models with Pydantic-AI."""

    model: str
    """Model identifier in provider:model format"""

    config: dict[str, dict[str, Any]] = field(default_factory=dict)
    """"Provider configurations."""

    def __post_init__(self):
        """Initialize the client."""
        import aisuite

        self._client = aisuite.Client(self.config)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.model

    @property
    def system(self) -> str:
        """Return the system/provider name."""
        return "aisuite"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a request to the model."""
        assert self._client
        formatted_messages = []

        # Convert messages to AISuite format
        for message in messages:
            match message:
                case ModelResponse():
                    content = ""
                    tool_calls = []
                    for part in message.parts:
                        match part:
                            case TextPart():
                                content += str(part.content)
                            case ToolCallPart():
                                arg_str = part.args_as_json_str()
                                fn = {"name": part.tool_name, "arguments": arg_str}
                                call = {
                                    "id": part.tool_call_id,
                                    "type": "function",
                                    "function": fn,
                                }
                                tool_calls.append(call)
                    msg: dict[str, Any] = {"role": "assistant"}
                    if content:
                        msg["content"] = content
                    if tool_calls:
                        msg["tool_calls"] = tool_calls
                    formatted_messages.append(msg)
                case _:  # ModelRequest
                    for req_part in message.parts:
                        match req_part:
                            case SystemPromptPart():
                                formatted_messages.append({
                                    "role": "system",
                                    "content": req_part.content,
                                })
                            case UserPromptPart():
                                if isinstance(req_part.content, str):
                                    formatted_messages.append({
                                        "role": "user",
                                        "content": req_part.content,
                                    })
                                else:
                                    content_items = [
                                        convert_content_item(item)
                                        for item in req_part.content
                                    ]
                                    formatted_messages.append({
                                        "role": "user",
                                        "content": content_items,
                                    })
                            case ToolReturnPart():
                                formatted_messages.append({
                                    "role": "tool",
                                    "tool_call_id": req_part.tool_call_id,
                                    "content": req_part.model_response_str(),
                                })

        kwargs = {}
        if model_settings:
            if hasattr(model_settings, "temperature"):
                kwargs["temperature"] = model_settings.temperature  # type: ignore
            if hasattr(model_settings, "max_tokens"):
                kwargs["max_tokens"] = model_settings.max_tokens  # type: ignore

        tools = []
        if model_request_parameters.function_tools:
            tools.extend(convert_tools(model_request_parameters.function_tools))
        if model_request_parameters.output_tools:
            tools.extend(convert_tools(model_request_parameters.output_tools))

        if tools:
            kwargs["tools"] = tools
            # Set tool_choice based on allow_text_output
            if not model_request_parameters.allow_text_output:
                kwargs["tool_choice"] = {"type": "function", "function": {"name": "auto"}}
            else:
                kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            **kwargs,
        )

        parts: list[Any] = []
        if (
            hasattr(response.choices[0].message, "tool_calls")
            and response.choices[0].message.tool_calls
        ):
            for tool_call in response.choices[0].message.tool_calls:
                if hasattr(tool_call, "type") and tool_call.type == "function":
                    function_call = tool_call.function
                    part = ToolCallPart(
                        tool_name=function_call.name,
                        args=function_call.arguments,
                        tool_call_id=tool_call.id,
                    )
                    parts.append(part)

        # Extract text content if present
        if (
            hasattr(response.choices[0].message, "content")
            and response.choices[0].message.content
        ):
            content = response.choices[0].message.content
            if content:  # Only add if not empty
                parts.append(TextPart(content))

        # If no parts were added, add an empty text part
        if not parts:
            parts.append(TextPart(""))

        # AISuite doesn't provide token counts yet
        ts = datetime.now(UTC)
        return ModelResponse(parts=parts, timestamp=ts, usage=RequestUsage())

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Streaming is not supported yet."""
        msg = "Streaming not supported by AISuite adapter"
        raise NotImplementedError(msg) from None
        # Need to yield even though we raise an error
        # to satisfy the async context manager protocol
        if False:  # pragma: no cover
            yield AISuiteStreamedResponse()


if __name__ == "__main__":
    import asyncio

    from pydantic_ai import Agent

    async def test():
        adapter = AISuiteAdapter(model="openai:gpt-4o-mini")
        agent: Agent[None, str] = Agent(model=adapter)
        response = await agent.run("Say hello!")
        print(response.output)

        @agent.tool_plain
        def calculate(a: int, b: int, operation: str) -> int:
            """Perform a simple calculation."""
            if operation == "add":
                return a + b
            if operation == "multiply":
                return a * b
            if operation == "subtract":
                return a - b
            if operation == "divide":
                return a // b
            return 0

        tool_response = await agent.run("What is 42 multiplied by 56?")
        print(f"Tool response: {tool_response.output}")

    asyncio.run(test())
