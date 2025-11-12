"""Custom OpenAI model implementation using only httpx."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
import os
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from pydantic import Field, TypeAdapter
from pydantic_ai import (
    ModelResponse,
    RequestUsage,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters, StreamedResponse

from llmling_models.base import PydanticModel
from llmling_models.log import get_logger


class StreamChunk(TypedDict):
    """OpenAI stream chunk format."""

    choices: list[dict[str, Any]]
    usage: dict[str, int] | None


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import httpx
    from pydantic_ai import (
        ModelMessage,
        ModelResponsePart,
        ModelResponseStreamEvent,
        RunContext,
    )
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition


logger = get_logger(__name__)
json_ta = TypeAdapter[Any](Any)


class FunctionDefinition(TypedDict):
    """OpenAI function definition format."""

    name: str
    description: str
    parameters: dict[str, Any]


def convert_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """Convert pydantic-ai messages to OpenAI format."""
    result = []
    for message in messages:
        if isinstance(message, ModelResponse):
            text = ""
            tool_calls = []
            for part in message.parts:
                match part:
                    case TextPart(content=content):
                        text += str(content)
                    case ToolCallPart(tool_call_id=tool_call_id, tool_name=tool_name):
                        fn_dct = {"name": tool_name, "arguments": part.args_as_json_str()}
                        tool_calls.append({
                            "id": tool_call_id,
                            "type": "function",
                            "function": fn_dct,
                        })
            msg: dict[str, Any] = {"role": "assistant"}
            if text:
                msg["content"] = text
            if tool_calls:
                msg["tool_calls"] = tool_calls
            result.append(msg)
        else:
            for request_part in message.parts:
                match request_part:
                    case SystemPromptPart(content=content):
                        result.append({"role": "system", "content": content})
                    case UserPromptPart(content=content):
                        result.append({"role": "user", "content": content})
                    case ToolReturnPart(tool_call_id=tool_call_id):
                        result.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": request_part.model_response_str(),
                        })

    return result


def convert_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert tool definitions to OpenAI format."""
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
class OpenAIStreamedResponse(StreamedResponse):
    """Stream implementation for OpenAI responses."""

    response: httpx.Response
    _timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    _model_name: str

    def __post_init__(self):
        """Initialize usage tracking and parts manager."""
        self._usage = RequestUsage()
        self._has_yielded_start = False

    @property
    def provider_name(self) -> str | None:
        """Get the provider name."""
        return "pyodide"

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream response chunks."""
        import anyenv

        try:
            content_id = "content"  # OpenAI uses a single content stream
            tool_calls: dict[str, dict[str, Any]] = {}

            async for line in self.response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue

                if line == "data: [DONE]":
                    break

                try:
                    data = anyenv.load_json(line.removeprefix("data: "), return_type=dict)
                except anyenv.JsonLoadError:
                    continue

                if data.get("error"):
                    msg = f"OpenAI error: {data['error']}"
                    raise RuntimeError(msg)  # noqa: TRY301

                choices = data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                if not delta:
                    continue

                # Handle text content
                if content := delta.get("content"):
                    if not self._has_yielded_start:
                        self._has_yielded_start = True
                        event = self._parts_manager.handle_text_delta(
                            vendor_part_id=content_id,
                            content=content,
                        )
                        if event is not None:
                            yield event
                    else:
                        event = self._parts_manager.handle_text_delta(
                            vendor_part_id=content_id,
                            content=content,
                        )
                        if event is not None:
                            yield event

                # Handle tool calls
                if tool_call_delta := delta.get("tool_calls", []):
                    for tool_delta in tool_call_delta:
                        index = str(tool_delta["index"])
                        if index not in tool_calls:
                            tool_calls[index] = {
                                "id": tool_delta.get("id", ""),
                                "function": {"name": "", "arguments": ""},
                            }

                        # Update tool call data
                        if "id" in tool_delta:
                            tool_calls[index]["id"] = tool_delta["id"]
                        if func := tool_delta.get("function", {}):
                            if "name" in func:
                                tool_calls[index]["function"]["name"] = func["name"]
                            if "arguments" in func:
                                tool_calls[index]["function"]["arguments"] += func[
                                    "arguments"
                                ]

                        # Generate event if we have complete tool call
                        call = tool_calls[index]
                        if call["id"] and call["function"]["name"]:
                            event_ = self._parts_manager.handle_tool_call_delta(
                                vendor_part_id=index,
                                tool_name=call["function"]["name"],
                                args=call["function"]["arguments"],
                                tool_call_id=call["id"],
                            )
                            if event_:
                                yield event_

                # Update usage if available
                if usage := data.get("usage"):
                    self._usage = RequestUsage(
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                    )

        except Exception as e:
            msg = f"Stream error: {e}"
            raise RuntimeError(msg) from e

    @property
    def timestamp(self) -> datetime:
        """Get response timestamp."""
        return self._timestamp

    @property
    def model_name(self) -> str:
        """Get response model_name."""
        return self._model_name


class SimpleOpenAIModel(PydanticModel):
    """OpenAI-compatible model using HTTPX."""

    type: Literal["openai-simple"] = Field(default="openai-simple", init=False)
    model: str
    """OpenAI model identifier."""

    api_key: str | None = None
    """OpenAI API key."""

    base_url: str | None = None
    """Base URL for API requests."""

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.model

    @property
    def system(self) -> str:
        """Return the system/provider name."""
        return "openai-simple"

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            msg = "OpenAI API key not provided"
            raise ValueError(msg)

        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def _build_request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build request payload."""
        req: dict[str, Any] = {
            "model": self.model,
            "messages": convert_messages(messages),
            "stream": stream,
        }

        # Add tools if provided
        tools = []
        if model_request_parameters.function_tools:
            tools.extend(convert_tools(model_request_parameters.function_tools))
        if model_request_parameters.output_tools:
            tools.extend(convert_tools(model_request_parameters.output_tools))

        if tools:
            req["tools"] = tools
            if not model_request_parameters.allow_text_output:
                req["tool_choice"] = "required"
            else:
                req["tool_choice"] = "auto"

        # Add model settings
        if model_settings:
            if temperature := model_settings.get("temperature"):
                req["temperature"] = temperature
            if max_tokens := model_settings.get("max_tokens"):
                req["max_tokens"] = max_tokens

        return req

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make request to OpenAI API."""
        import httpx

        headers = self._get_headers()
        payload = self._build_request(
            messages,
            model_settings,
            model_request_parameters,
        )
        base_url = self.base_url or "https://api.openai.com/v1"

        async with httpx.AsyncClient() as client:
            try:
                url = f"{base_url}/chat/completions"
                response = await client.post(
                    url, headers=headers, json=payload, timeout=30.0
                )
                response.raise_for_status()
                data = response.json()

                # Extract choice data
                choice = data["choices"][0]["message"]

                # Handle response parts
                parts: list[ModelResponsePart] = []

                # Add text content if present
                if content := choice.get("content"):
                    parts.append(TextPart(content))

                # Add tool calls if present
                if tool_calls := choice.get("tool_calls"):
                    for call in tool_calls:
                        tool_part = ToolCallPart(
                            tool_name=call["function"]["name"],
                            args=call["function"]["arguments"],
                            tool_call_id=call["id"],
                        )
                        parts.append(tool_part)

                # Extract usage
                usage_data = data.get("usage", {})
                usage = RequestUsage(
                    input_tokens=usage_data.get("prompt_tokens", 0),
                    output_tokens=usage_data.get("completion_tokens", 0),
                )

                ts = datetime.now(UTC)
                return ModelResponse(parts=parts, timestamp=ts, usage=usage)

            except httpx.HTTPError as e:
                msg = f"OpenAI request failed: {e}"
                raise RuntimeError(msg) from e

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Stream response from OpenAI API."""
        import httpx

        headers = self._get_headers()
        payload = self._build_request(
            messages,
            model_settings,
            model_request_parameters,
            stream=True,
        )
        base_url = self.base_url or "https://api.openai.com/v1"
        client = httpx.AsyncClient(timeout=30.0)
        try:
            url = f"{base_url}/chat/completions"
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            yield OpenAIStreamedResponse(
                model_request_parameters=ModelRequestParameters(),
                response=response,
                _model_name=self.model_name,
            )

        except httpx.HTTPError as e:
            msg = f"OpenAI stream request failed: {e}"
            raise RuntimeError(msg) from e
        finally:
            await client.aclose()


if __name__ == "__main__":
    import asyncio

    from pydantic_ai import Agent

    async def test():
        # Create model instance
        model = SimpleOpenAIModel(model="gpt-5-nano")
        agent: Agent[None, str] = Agent(model=model)
        result = await agent.run("Hello!")
        print(f"\nResponse: {result.output}")
        print("\nStreaming response:")
        async with agent.run_stream("Tell me a short story") as stream:
            async for chunk in stream.stream():
                print(chunk)
        print("\nStreaming complete!")

    asyncio.run(test())
