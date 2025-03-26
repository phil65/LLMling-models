"""Adapter to use LiteLLM with Pydantic-AI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.result import Usage

from llmling_models.base import PydanticModel
from llmling_models.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition

logger = get_logger(__name__)


def convert_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """Convert pydantic-ai messages to LiteLLM format."""
    result = []
    for message in messages:
        if isinstance(message, ModelResponse):
            content = ""
            tool_calls = []
            for part in message.parts:
                if isinstance(part, TextPart):
                    content += str(part.content)
                elif isinstance(part, ToolCallPart):
                    tool_calls.append({
                        "id": part.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": part.tool_name,
                            "arguments": part.args_as_json_str(),
                        },
                    })

            msg: dict[str, Any] = {"role": "assistant"}
            if content:
                msg["content"] = content
            if tool_calls:
                msg["tool_calls"] = tool_calls
            result.append(msg)  # type: ignore
        else:  # ModelRequest
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    result.append({"role": "system", "content": part.content})  # type: ignore
                elif isinstance(part, UserPromptPart):
                    # TODO: Handle multi-modal content if needed
                    result.append({"role": "user", "content": str(part.content)})  # type: ignore
                elif isinstance(part, ToolReturnPart):
                    result.append({  # type: ignore
                        "role": "tool",
                        "tool_call_id": part.tool_call_id,
                        "content": part.model_response_str(),
                    })

    return result  # type: ignore


def convert_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert tool definitions to LiteLLM format."""
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


class LiteLLMAdapter(PydanticModel):
    """Adapter to use LiteLLM library models with Pydantic-AI."""

    type: Literal["litellm"] = Field(default="litellm", init=False)
    """LiteLLM model type."""

    model: str
    """Model identifier in provider/model format"""

    api_key: str | None = None
    """API key for the model provider."""

    litellm_params: dict[str, Any] = Field(default_factory=dict)
    """Additional parameters to pass to LiteLLM."""

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.model

    @property
    def system(self) -> str:
        """Return the system/provider name."""
        return "litellm"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, Usage]:
        """Make a request to the model."""
        from litellm import Choices, ModelResponse as LiteLLMModelResponse, acompletion

        formatted_messages = convert_messages(messages)
        params = self.litellm_params.copy()

        if model_settings:
            if hasattr(model_settings, "temperature"):
                params["temperature"] = model_settings.temperature  # type: ignore
            if hasattr(model_settings, "max_tokens"):
                params["max_tokens"] = model_settings.max_tokens  # type: ignore

        tools = []
        if model_request_parameters.function_tools:
            tools.extend(convert_tools(model_request_parameters.function_tools))
        if model_request_parameters.result_tools:
            tools.extend(convert_tools(model_request_parameters.result_tools))

        if tools:
            params["tools"] = tools
            if not model_request_parameters.allow_text_result:
                params["tool_choice"] = "required"
            else:
                params["tool_choice"] = "auto"

        # Make the request
        try:
            response = await acompletion(
                model=self.model,
                messages=formatted_messages,
                api_key=self.api_key,
                **params,
            )
            assert isinstance(response, LiteLLMModelResponse)
            parts: list[Any] = []
            choice = response.choices[0]
            assert isinstance(choice, Choices)
            if content := choice.message.get("content"):
                parts.append(TextPart(content))
            if tool_calls := choice.message.get("tool_calls"):
                for call in tool_calls:
                    part = ToolCallPart(
                        tool_name=call["function"]["name"],
                        args=call["function"]["arguments"],
                        tool_call_id=call["id"],
                    )
                    parts.append(part)

            usage_data = getattr(response, "usage", {})
            usage_obj = Usage(
                request_tokens=usage_data.get("prompt_tokens", 0),
                response_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

            return ModelResponse(parts=parts, timestamp=datetime.now(UTC)), usage_obj

        except Exception as e:
            msg = f"LiteLLM request failed: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        """Stream responses from the model."""
        import litellm

        formatted_messages = convert_messages(messages)
        params = self.litellm_params.copy()
        if model_settings:
            if hasattr(model_settings, "temperature"):
                params["temperature"] = model_settings.temperature  # type: ignore
            if hasattr(model_settings, "max_tokens"):
                params["max_tokens"] = model_settings.max_tokens  # type: ignore
        tools = []
        if model_request_parameters.function_tools:
            tools.extend(convert_tools(model_request_parameters.function_tools))
        if model_request_parameters.result_tools:
            tools.extend(convert_tools(model_request_parameters.result_tools))
        if tools:
            params["tools"] = tools
            if not model_request_parameters.allow_text_result:
                params["tool_choice"] = "required"
            else:
                params["tool_choice"] = "auto"

        try:
            stream = await litellm.acompletion(
                model=self.model,
                messages=formatted_messages,
                api_key=self.api_key,
                stream=True,
                **params,
            )

            yield LiteLLMStreamedResponse(response=stream, _model_name=self.model)

        except Exception as e:
            msg = f"LiteLLM streaming request failed: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e


@dataclass(kw_only=True)
class LiteLLMStreamedResponse(StreamedResponse):
    """Stream implementation for LiteLLM."""

    response: Any  # litellm streaming response
    _model_name: str
    _timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        """Initialize usage."""
        self._usage = Usage()
        self._content_part_id = "content"  # Default part ID for content
        self._tool_calls: dict[str, dict[str, Any]] = {}  # Track tool calls by index

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream chunks as events."""
        try:
            async for chunk in self.response:
                if hasattr(chunk, "usage") and chunk.usage:
                    self._usage = Usage(
                        request_tokens=chunk.usage.get("prompt_tokens", 0),
                        response_tokens=chunk.usage.get("completion_tokens", 0),
                        total_tokens=chunk.usage.get("total_tokens", 0),
                    )

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta
                if content := delta.get("content"):
                    yield self._parts_manager.handle_text_delta(
                        vendor_part_id=self._content_part_id,
                        content=content,
                    )

                # Handle tool calls
                if tool_calls := delta.get("tool_calls", []):
                    for tool_delta in tool_calls:
                        index = str(tool_delta.get("index", 0))
                        if index not in self._tool_calls:
                            self._tool_calls[index] = {
                                "id": tool_delta.get("id", ""),
                                "function": {"name": "", "arguments": ""},
                            }
                        if "id" in tool_delta:
                            self._tool_calls[index]["id"] = tool_delta["id"]

                        if func := tool_delta.get("function", {}):
                            if "name" in func:
                                self._tool_calls[index]["function"]["name"] = func["name"]
                            if "arguments" in func:
                                self._tool_calls[index]["function"]["arguments"] += func[
                                    "arguments"
                                ]

                        call = self._tool_calls[index]
                        if call["id"] and call["function"]["name"]:
                            event = self._parts_manager.handle_tool_call_delta(
                                vendor_part_id=index,
                                tool_name=call["function"]["name"],
                                args=call["function"]["arguments"],
                                tool_call_id=call["id"],
                            )
                            if event:
                                yield event

        except Exception as e:
            msg = f"Stream error: {e}"
            raise RuntimeError(msg) from e

    @property
    def model_name(self) -> str:
        """Get response model_name."""
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        """Get response timestamp."""
        return self._timestamp


if __name__ == "__main__":
    import asyncio

    from pydantic_ai import Agent

    async def test():
        model = LiteLLMAdapter(model="gpt-4o-mini")
        agent: Agent[None, str] = Agent(model=model)
        response = await agent.run("Tell me a brief story about a clever fox")
        print(f"Response: {response.data}")
        print("\nTesting streaming:")
        async with agent.run_stream("List 3 interesting science facts") as stream:
            async for chunk in stream.stream_text(delta=True):
                print(chunk, end="", flush=True)

        @agent.tool_plain
        def multiply(a: int, b: int) -> int:
            """Calculate a simple math operation."""
            return a * b

        print("\nTesting with tools:")
        tool_agent: Agent[None, str] = Agent(model=model, tools=[multiply])
        tool_response = await tool_agent.run("What is 42 multiplied by 56?")
        print(f"Tool response: {tool_response.data}")

    asyncio.run(test())
