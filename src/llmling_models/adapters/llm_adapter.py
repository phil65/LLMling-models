"""Adapter to use LLM library models with Pydantic-AI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
import inspect
import json
import re
from typing import TYPE_CHECKING, Any
import uuid

from pydantic_ai import BinaryContent, ImageUrl, RequestUsage
from pydantic_ai.messages import (
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse

from llmling_models.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import llm
    from pydantic_ai import RunContext
    from pydantic_ai.messages import (
        ModelMessage,
        ModelResponseStreamEvent,
    )
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition

logger = get_logger(__name__)


async def _map_async_usage(response: llm.AsyncResponse) -> RequestUsage:
    """Map async LLM usage to Pydantic-AI usage."""
    await response._force()  # Ensure usage is available
    return RequestUsage(
        input_tokens=response.input_tokens or 0,
        output_tokens=response.output_tokens or 0,
        details=response.token_details or {},
    )


def _map_sync_usage(response: llm.Response) -> RequestUsage:
    """Map sync LLM usage to Pydantic-AI usage."""
    response._force()
    return RequestUsage(
        input_tokens=response.input_tokens or 0,
        output_tokens=response.output_tokens or 0,
    )


def _build_prompt(
    messages: list[ModelMessage], attachments: list[Any] | None = None
) -> tuple[str, str | None, list[Any]]:
    """Build a prompt and optional system prompt from messages, with attachments."""
    import llm

    prompt_parts = []
    system = None
    llm_attachments = []

    # Process any provided attachments
    if attachments:
        llm_attachments.extend(attachments)

    for message in messages:
        if isinstance(message, ModelResponse):
            for rsp_part in message.parts:
                if isinstance(rsp_part, TextPart):
                    prompt_parts.append(f"Assistant: {rsp_part.content}")
                elif isinstance(rsp_part, ToolCallPart):
                    # Include tool calls in conversation
                    call_info = f"{rsp_part.tool_name}({rsp_part.args_as_json_str()})"
                    prompt_parts.append(f"Assistant called tool: {call_info}")
        else:  # ModelRequest
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    system = part.content
                elif isinstance(part, UserPromptPart | RetryPromptPart):
                    if isinstance(part.content, str):
                        prompt_parts.append(f"Human: {part.content}")
                    else:
                        # Handle multi-modal content - convert to text for now
                        text_parts = []
                        for item in part.content:
                            if isinstance(item, str):
                                text_parts.append(item)
                            elif isinstance(item, ImageUrl):
                                # Convert ImageURL to LLM Attachment
                                llm_attachments.append(llm.Attachment(url=item.url))
                                text_parts.append("[Image attached]")
                            elif isinstance(item, BinaryContent) and item.is_image:
                                # Convert BinaryContent to LLM Attachment
                                llm_attachments.append(
                                    llm.Attachment(
                                        content=item.data, type=item.media_type
                                    )
                                )
                                text_parts.append("[Image attached]")
                        prompt_parts.append(f"Human: {' '.join(text_parts)}")
                elif isinstance(part, ToolReturnPart):
                    # Include tool results in conversation
                    result = part.model_response_str()
                    prompt_parts.append(f"Tool {part.tool_call_id} returned: {result}")

    return "\n".join(prompt_parts), system, llm_attachments


def _create_noop_function(tool_def: ToolDefinition) -> Any:
    """Create a NOOP fn that LLM can call but returns markers instead of executing."""
    schema = tool_def.parameters_json_schema
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Build parameter list for function signature
    params = []
    annotations: dict[str, type | Any] = {}

    for param_name, param_info in properties.items():
        param_type = param_info.get("type", "string")
        default_value: Any = inspect.Parameter.empty

        # Map JSON schema types to Python types
        if param_type == "string":
            annotations[param_name] = str
        elif param_type == "integer":
            annotations[param_name] = int
        elif param_type == "number":
            annotations[param_name] = float
        elif param_type == "boolean":
            annotations[param_name] = bool
        elif param_type == "array":
            annotations[param_name] = list
        elif param_type == "object":
            annotations[param_name] = dict
        else:
            annotations[param_name] = Any

        # Set default to None if parameter is not required
        if param_name not in required:
            default_value = None
            # Update annotation to include None
            if param_name in annotations:
                annotations[param_name] = annotations[param_name] | None

        param = inspect.Parameter(
            param_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=default_value,
            annotation=annotations[param_name],
        )
        params.append(param)

    # Create function signature
    sig = inspect.Signature(params)

    def noop_function(*args, **kwargs):
        """NOOP function that returns tool call marker."""
        # Bind arguments to parameter names
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Create unique call ID
        call_id = f"call_{uuid.uuid4().hex[:16]}"

        # Return structured marker
        marker = {
            "__TOOL_CALL_MARKER__": {
                "tool_name": tool_def.name,
                "args": dict(bound.arguments),
                "call_id": call_id,
            }
        }

        return json.dumps(marker)

    # Set function metadata
    noop_function.__name__ = tool_def.name
    noop_function.__doc__ = tool_def.description
    noop_function.__signature__ = sig  # type: ignore
    noop_function.__annotations__ = annotations

    return noop_function


def _extract_tool_calls_from_text(text: str) -> tuple[str, list[ToolCallPart]]:
    """Extract tool call markers from LLM response text."""
    tool_calls = []
    clean_text = text
    # Find potential JSON objects in the text
    json_pattern = r'\{[^{}]*"__TOOL_CALL_MARKER__"[^{}]*\}'
    matches = re.findall(json_pattern, text)

    for match in matches:
        try:
            marker_data = json.loads(match)
            if "__TOOL_CALL_MARKER__" in marker_data:
                call_info = marker_data["__TOOL_CALL_MARKER__"]

                tool_call = ToolCallPart(
                    tool_name=call_info["tool_name"],
                    args=json.dumps(call_info["args"]),
                    tool_call_id=call_info["call_id"],
                )
                tool_calls.append(tool_call)

                # Remove the marker from the text
                clean_text = clean_text.replace(match, "")

        except (json.JSONDecodeError, KeyError) as e:
            logger.debug("Failed to parse tool call marker: %s", e)

    return clean_text.strip(), tool_calls


@dataclass
class LLMAdapter(Model):
    """Adapter to use LLM library models with Pydantic-AI."""

    model: str
    needs_key: str | None = None
    key_env_var: str | None = None
    can_stream: bool = False

    def __post_init__(self):
        """Initialize models."""
        import llm

        self._async_model = None
        self._sync_model = None
        try:
            self._async_model = llm.get_async_model(self.model)
            self.needs_key = self._async_model.needs_key
            self.key_env_var = self._async_model.key_env_var
            self.can_stream = self._async_model.can_stream
        except llm.UnknownModelError:
            pass
        else:
            return

        try:
            self._sync_model = llm.get_model(self.model)
            self.needs_key = self._sync_model.needs_key
            self.key_env_var = self._sync_model.key_env_var
            self.can_stream = self._sync_model.can_stream
        except llm.UnknownModelError as e:
            msg = f"No sync or async model found for {self.model}"
            raise ValueError(msg) from e

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.model

    @property
    def system(self) -> str:
        """Return the system/provider name."""
        return "llm"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a request to the model."""
        prompt, system, attachments = _build_prompt(messages)

        # Check if tools are present
        tools = []
        if model_request_parameters.function_tools:
            tools.extend(model_request_parameters.function_tools)
        if model_request_parameters.output_tools:
            tools.extend(model_request_parameters.output_tools)

        if tools:
            # Create NOOP functions for LLM to call
            noop_functions = [_create_noop_function(tool) for tool in tools]

            if self._async_model:
                async_chain_response = self._async_model.chain(
                    prompt,
                    system=system,
                    tools=noop_functions,
                    attachments=attachments,
                )
                text = await async_chain_response.text()

                # Get usage from final response if available
                final_response: Any = None
                async for async_response in async_chain_response.responses():
                    final_response = async_response
                usage = (
                    await _map_async_usage(final_response)
                    if final_response
                    else RequestUsage()
                )

            elif self._sync_model:
                chain_response = self._sync_model.chain(
                    prompt,
                    system=system,
                    tools=noop_functions,
                    attachments=attachments,
                )
                text = chain_response.text()

                # Get usage from final response if available
                final_response = None
                for response in chain_response.responses():
                    final_response = response
                usage = (
                    _map_sync_usage(final_response) if final_response else RequestUsage()
                )

            else:
                msg = "No model available"
                raise RuntimeError(msg)

            # Extract tool calls from the response
            clean_text, tool_calls = _extract_tool_calls_from_text(text)

            parts: list[Any] = []
            if clean_text:
                parts.append(TextPart(clean_text))
            parts.extend(tool_calls)

        else:
            # No tools - use regular prompt() method
            if self._async_model:
                response = await self._async_model.prompt(
                    prompt, system=system, stream=False, attachments=attachments
                )
                text = await response.text()
                usage = await _map_async_usage(response)
            elif self._sync_model:
                response = self._sync_model.prompt(
                    prompt, system=system, stream=False, attachments=attachments
                )
                text = response.text()
                usage = _map_sync_usage(response)
            else:
                msg = "No model available"
                raise RuntimeError(msg)

            parts = [TextPart(text)]

        ts = datetime.now(UTC)
        return ModelResponse(parts=parts, timestamp=ts, usage=usage)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streaming request to the model."""
        prompt, system, attachments = _build_prompt(messages)

        # Check if tools are present
        tools = []
        if model_request_parameters.function_tools:
            tools.extend(model_request_parameters.function_tools)
        if model_request_parameters.output_tools:
            tools.extend(model_request_parameters.output_tools)

        if tools:
            # Create NOOP functions for LLM to call
            noop_functions = [_create_noop_function(tool) for tool in tools]

            if self._async_model:
                chain_response: Any = self._async_model.chain(
                    prompt,
                    system=system,
                    tools=noop_functions,
                    attachments=attachments,
                )
            elif self._sync_model:
                chain_response = self._sync_model.chain(
                    prompt,
                    system=system,
                    tools=noop_functions,
                    attachments=attachments,
                )
            else:
                msg = "No model available"
                raise RuntimeError(msg)

            yield LLMStreamedResponse(
                model_request_parameters=ModelRequestParameters(),
                response=chain_response,
                is_chain=True,
            )
        else:
            # No tools - use regular streaming

            if self._async_model:
                response = await self._async_model.prompt(
                    prompt, system=system, stream=True, attachments=attachments
                )
            elif self._sync_model and self._sync_model.can_stream:
                response = self._sync_model.prompt(
                    prompt, system=system, stream=True, attachments=attachments
                )
            else:
                msg = (
                    "No streaming capable model available. "
                    "Either async model is missing or sync model not supporting streaming"
                )
                raise RuntimeError(msg)

            yield LLMStreamedResponse(
                model_request_parameters=ModelRequestParameters(),
                response=response,
                is_chain=False,
            )


@dataclass(kw_only=True)
class LLMStreamedResponse(StreamedResponse):
    """Stream implementation for LLM responses."""

    response: Any  # llm.Response | llm.AsyncResponse | ChainResponse
    is_chain: bool = False
    _timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    _model_name: str = "llm"

    def __post_init__(self):
        """Initialize usage."""
        self._usage = RequestUsage()

    @property
    def provider_name(self) -> str | None:
        """Get the provider name."""
        return "llm"

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream response chunks as events."""
        import llm

        try:
            if self.is_chain:
                # Handle chain responses (with tools)

                try:
                    if hasattr(self.response, "__aiter__"):
                        # Async chain
                        async for chunk in self.response:
                            if chunk:  # Only yield non-empty chunks
                                event = self._parts_manager.handle_text_delta(
                                    vendor_part_id="content",
                                    content=str(chunk),
                                )
                                if event is not None:
                                    yield event
                    else:
                        # Sync chain
                        for chunk in self.response:
                            if chunk:  # Only yield non-empty chunks
                                event = self._parts_manager.handle_text_delta(
                                    vendor_part_id="content",
                                    content=chunk,
                                )
                                if event is not None:
                                    yield event
                except Exception as e:  # noqa: BLE001
                    logger.debug("Chain streaming failed, falling back to text: %s", e)
                    # Fallback: try to get final text
                    try:
                        if hasattr(self.response, "text"):
                            if callable(self.response.text):
                                if hasattr(self.response, "__aiter__"):
                                    text = await self.response.text()  # pyright: ignore
                                else:
                                    text = self.response.text()
                            else:
                                text = str(self.response.text)
                            if text:
                                event = self._parts_manager.handle_text_delta(
                                    vendor_part_id="content",
                                    content=str(text),
                                )
                                if event is not None:
                                    yield event
                    except Exception as fallback_e:  # noqa: BLE001
                        logger.warning(
                            "Could not get text from chain response: %s", fallback_e
                        )

                # Try to get usage from final response if available
                try:
                    if hasattr(self.response, "responses"):
                        final_response = None
                        if hasattr(self.response.responses, "__aiter__"):
                            async for resp in self.response.responses():
                                final_response = resp
                        else:
                            for resp in self.response.responses():
                                final_response = resp

                        if final_response and isinstance(
                            final_response, llm.AsyncResponse
                        ):
                            self._usage = await _map_async_usage(final_response)
                        elif final_response:
                            self._usage = _map_sync_usage(final_response)
                except Exception as e:  # noqa: BLE001
                    logger.debug("Could not extract usage from chain response: %s", e)
            else:
                # Handle regular streaming responses
                # LLM library's AsyncResponse with stream=True sometimes
                # doesn't support direct iteration
                # Use fallback to get complete text at once
                try:
                    if hasattr(self.response, "text"):
                        if callable(self.response.text):
                            if isinstance(self.response, llm.AsyncResponse):
                                text = await self.response.text()
                            else:
                                text = self.response.text()
                        else:
                            text = self.response.text

                        if text:
                            # Stream the text character by character to simulate streaming
                            for char in str(text):
                                event = self._parts_manager.handle_text_delta(
                                    vendor_part_id="content",
                                    content=char,
                                )
                                if event is not None:
                                    yield event

                            # Update usage after streaming
                            if isinstance(self.response, llm.AsyncResponse):
                                self._usage = await _map_async_usage(self.response)
                            else:
                                self._usage = _map_sync_usage(self.response)
                        return

                except Exception as e:  # noqa: BLE001
                    logger.debug("Text fallback failed, trying iteration: %s", e)

                # Fallback to direct iteration if text method fails
                chunk_count = 0
                while True:
                    try:
                        if isinstance(self.response, llm.AsyncResponse):
                            chunk = await self.response.__anext__()
                        else:
                            chunk = next(iter(self.response))

                        chunk_count += 1

                        if chunk:  # Only yield non-empty chunks
                            event = self._parts_manager.handle_text_delta(
                                vendor_part_id="content",
                                content=chunk,
                            )
                            if event is not None:
                                yield event

                    except (StopIteration, StopAsyncIteration):
                        break

                # Update usage after iteration
                try:
                    if isinstance(self.response, llm.AsyncResponse):
                        self._usage = await _map_async_usage(self.response)
                    else:
                        self._usage = _map_sync_usage(self.response)
                except Exception as e:  # noqa: BLE001
                    logger.debug("Could not get usage from response: %s", e)

        except Exception as e:
            msg = f"Stream error: {e}"
            logger.exception(msg)
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
        # Test with both sync and async models
        adapter = LLMAdapter(model="gpt-4o-mini")
        agent: Agent[None, str] = Agent(model=adapter)

        print("\nTesting basic request:")
        response = await agent.run("Say hello!")
        print(f"Response: {response.output}")

        print("\nTesting streaming:")
        async with agent.run_stream("Tell me a story") as stream:
            async for chunk in stream.stream_text(delta=True):
                print(chunk, end="", flush=True)

        print("\n\nTesting with tools:")

        @agent.tool_plain
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        tool_response = await agent.run("What is 42 multiplied by 56?")
        print(f"Tool response: {tool_response.output}")

    asyncio.run(test())
