"""Model that delegates to Claude Code CLI via the Claude Agent SDK."""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai import (
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelRequest,
    ModelResponse,
    ModelResponse as MsgResponse,
    PartStartEvent,
    RequestUsage,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, StreamedResponse

from llmling_models.builtin_tools import (
    ClaudeCodeBashTool,
    ClaudeCodeEditTool,
    ClaudeCodeGlobTool,
    ClaudeCodeGrepTool,
    ClaudeCodeNotebookEditTool,
    ClaudeCodeReadTool,
    ClaudeCodeTaskTool,
    ClaudeCodeWebFetchTool,
    ClaudeCodeWebSearchTool,
    ClaudeCodeWriteTool,
)
from llmling_models.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from claude_agent_sdk import ClaudeAgentOptions
    from pydantic_ai import (
        ModelMessage,
        ModelResponsePart,
        ModelResponseStreamEvent,
        ModelSettings,
        RunContext,
    )
    from pydantic_ai.builtin_tools import AbstractBuiltinTool
    from pydantic_ai.models import ModelRequestParameters

logger = get_logger(__name__)

# Permission modes supported by Claude Code
PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]


def _extract_system_prompt(messages: list[ModelMessage]) -> str | None:
    """Extract system prompt from messages.

    Collects all SystemPromptPart entries and joins them with double newlines.
    """
    system_parts: list[str] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    system_parts.append(part.content)  # noqa: PERF401
    return "\n\n".join(system_parts) if system_parts else None


def _extract_prompt(messages: list[ModelMessage]) -> str | list[dict[str, Any]]:
    """Extract prompt from pydantic-ai messages.

    Returns either:
    - str: Simple text prompt (for text-only messages)
    - list[dict]: Structured content with text and images (for multimodal)

    The SDK accepts string prompts directly, or AsyncIterable[dict] for
    streaming with multimodal content.
    """
    text_parts: list[str] = []
    content_parts: list[dict[str, Any]] = []
    has_images = False

    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                # Skip system prompts - handled separately
                if isinstance(part, SystemPromptPart):
                    continue
                if isinstance(part, UserPromptPart):
                    if isinstance(part.content, str):
                        text_parts.append(part.content)
                        content_parts.append({"type": "text", "text": part.content})
                    else:
                        # Sequence of UserContent items
                        for item in part.content:
                            if isinstance(item, str):
                                text_parts.append(item)
                                content_parts.append({"type": "text", "text": item})
                            elif isinstance(item, BinaryContent):
                                # Handle binary content (images and PDFs)
                                if item.media_type:
                                    if isinstance(item.data, bytes):
                                        b64_data = base64.b64encode(item.data).decode()
                                    else:
                                        b64_data = item.data

                                    if item.media_type.startswith("image/"):
                                        has_images = True
                                        content_parts.append({
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": item.media_type,
                                                "data": b64_data,
                                            },
                                        })
                                    elif item.media_type == "application/pdf":
                                        has_images = True  # triggers multimodal path
                                        content_parts.append({
                                            "type": "document",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "application/pdf",
                                                "data": b64_data,
                                            },
                                        })
                            # ImageUrl, AudioUrl, etc. - URLs not supported by SDK

        elif isinstance(message, MsgResponse):
            for msg_part in message.parts:
                if isinstance(msg_part, TextPart):
                    text = f"Assistant: {msg_part.content}"
                    text_parts.append(text)
                    content_parts.append({"type": "text", "text": text})

    # Return structured content if we have images, otherwise simple string
    if has_images:
        return content_parts
    return "\n\n".join(text_parts) if text_parts else ""


@dataclass(kw_only=True)
class ClaudeCodeStreamedResponse(StreamedResponse):
    """Real-time streaming response from Claude Code.

    Uses an asyncio.Queue to receive messages from the SDK as they arrive,
    enabling true streaming without waiting for all messages to complete.
    """

    _message_queue: asyncio.Queue[Any]
    _collection_done: asyncio.Event
    _collection_error: list[Exception]
    _usage: RequestUsage = field(default_factory=RequestUsage)
    _timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    _model_name: str = "claude-code"

    @property
    def provider_name(self) -> str | None:
        """Get the provider name."""
        return "claude-code"

    @property
    def provider_url(self) -> str | None:
        """Get the provider URL."""
        return None

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream events as they arrive from the queue."""
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )
        from claude_agent_sdk.types import StreamEvent

        tool_use_names: dict[str, str] = {}

        while True:
            # Check if collection is done and queue is empty
            if self._collection_done.is_set() and self._message_queue.empty():
                break
            # Check for collection errors
            if self._collection_error:
                raise self._collection_error[0]
            try:  # Try to get a message with a short timeout
                message = await asyncio.wait_for(self._message_queue.get(), timeout=0.1)
            except TimeoutError:
                continue

            # Handle StreamEvent for real-time token streaming
            if isinstance(message, StreamEvent):
                event = message.event
                event_type = event.get("type")

                if event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    delta_type = delta.get("type")

                    if delta_type == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            for ev in self._parts_manager.handle_text_delta(
                                vendor_part_id="text",
                                content=text,
                            ):
                                yield ev

                    elif delta_type == "thinking_delta":
                        thinking = delta.get("thinking", "")
                        if thinking:
                            for ev in self._parts_manager.handle_thinking_delta(
                                vendor_part_id="thinking",
                                content=thinking,
                            ):
                                yield ev

                elif event_type == "message_start":
                    # Extract model name from message_start
                    msg = event.get("message", {})
                    if model := msg.get("model"):
                        self._model_name = f"claude-code:{model}"

                continue  # StreamEvent handled, continue to next message

            # Handle full AssistantMessage (contains tool calls)
            if isinstance(message, AssistantMessage):
                self._model_name = f"claude-code:{message.model}"
                for block in message.content:
                    # Text is already streamed via StreamEvent, skip here
                    if isinstance(block, TextBlock):
                        pass
                    elif isinstance(block, ThinkingBlock):
                        pass  # Already streamed via thinking_delta
                    elif isinstance(block, ToolUseBlock):
                        # Track tool name and emit builtin tool call
                        tool_use_names[block.id] = block.name
                        part = BuiltinToolCallPart(
                            tool_name=block.name,
                            args=block.input,
                            tool_call_id=block.id,
                            provider_name="claude-code",
                        )
                        self._parts_manager._parts.append(part)
                        yield PartStartEvent(index=len(self._parts_manager._parts) - 1, part=part)
            elif isinstance(message, UserMessage):
                # UserMessage contains tool results
                if isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            tool_name = tool_use_names.get(block.tool_use_id, "unknown")
                            if isinstance(block.content, str):
                                content = block.content
                            elif block.content is None:
                                content = ""
                            else:
                                content = json.dumps(block.content)
                            return_part = BuiltinToolReturnPart(
                                tool_name=tool_name,
                                content=content,
                                tool_call_id=block.tool_use_id,
                                provider_name="claude-code",
                            )
                            self._parts_manager._parts.append(return_part)
                            yield PartStartEvent(
                                index=len(self._parts_manager._parts) - 1, part=return_part
                            )
            elif isinstance(message, ResultMessage):
                if message.usage:
                    u = message.usage
                    self._usage = RequestUsage(
                        input_tokens=u.get("input_tokens", 0),
                        output_tokens=u.get("output_tokens", 0),
                        cache_read_tokens=u.get("cache_read_input_tokens", 0),
                        cache_write_tokens=u.get("cache_creation_input_tokens", 0),
                    )

    @property
    def timestamp(self) -> datetime:
        """Get response timestamp."""
        return self._timestamp

    @property
    def model_name(self) -> str:
        """Get response model_name."""
        return self._model_name


class ClaudeCodeModel(Model):
    """Model that delegates to Claude Code CLI.

    This model uses the Claude Agent SDK to communicate with the Claude Code CLI,
    which provides access to Claude with filesystem access, code execution,
    and other agentic capabilities.

    Requires authentication via `claude login` (CLI handles auth automatically).
    The Claude Code CLI is bundled with the claude-agent-sdk package.

    Usage:
        # Direct instantiation
        model = ClaudeCodeModel(model='opus')

        # Via infer_model
        model = infer_model('claude-code:opus')
        model = infer_model('claude-code')  # defaults to sonnet
    """

    def __init__(
        self,
        model: str = "sonnet",
        *,
        cwd: str | None = None,
        permission_mode: PermissionMode = "bypassPermissions",
        system_prompt: str | None = None,
        max_turns: int | None = None,
        max_thinking_tokens: int | None = None,
        include_partial_messages: bool = True,
    ) -> None:
        """Initialize the Claude Code model.

        Args:
            model: The Claude model to use (e.g., 'opus', 'sonnet', 'haiku').
            cwd: Working directory for Claude Code operations.
            permission_mode: Permission mode for tool execution.
                - 'default': CLI prompts for dangerous tools
                - 'acceptEdits': Auto-accept file edits
                - 'plan': Plan mode (no execution)
                - 'bypassPermissions': Allow all tools (use with caution)
            system_prompt: Custom system prompt to use.
            max_turns: Maximum number of conversation turns (1-100).
            max_thinking_tokens: Maximum tokens for extended thinking.
            include_partial_messages: Enable real-time token streaming (default True).
        """
        super().__init__()
        self._model = model
        self._cwd = cwd
        self._permission_mode: PermissionMode = permission_mode
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._max_thinking_tokens = max_thinking_tokens
        self._include_partial_messages = include_partial_messages

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return f"claude-code:{self._model}"

    @property
    def system(self) -> str:
        """Return the system/provider name."""
        return "claude-code"

    @classmethod
    def supported_builtin_tools(cls) -> frozenset[type[AbstractBuiltinTool]]:
        """Return the set of builtin tool types this model class can handle.

        Subclasses should override this to reflect their actual capabilities.
        Default is empty set - subclasses must explicitly declare support.
        """
        return frozenset([
            ClaudeCodeBashTool,
            ClaudeCodeEditTool,
            ClaudeCodeGlobTool,
            ClaudeCodeGrepTool,
            ClaudeCodeNotebookEditTool,
            ClaudeCodeReadTool,
            ClaudeCodeTaskTool,
            ClaudeCodeWebFetchTool,
            ClaudeCodeWebSearchTool,
            ClaudeCodeWriteTool,
        ])

    def _build_options(
        self, model_request_parameters: ModelRequestParameters | None = None
    ) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions from settings and builtin tools.

        Tool availability follows standard pydantic-ai semantics:
        - If no builtin tools passed: no tools available
        - If builtin tools passed: only those specific tools are allowed

        Note: We use `tools=[]` (not `allowed_tools=[]`) to disable all tools.
        - `tools=[]` sends `--tools ""` to CLI → no tools
        - `tools=None` (default) → doesn't send flag → CLI uses all tools
        - `allowed_tools=[]` is falsy, doesn't send flag → no filtering

        Also handles MCPServerTool for connecting to external MCP servers.
        """
        from claude_agent_sdk import ClaudeAgentOptions
        from claude_agent_sdk.types import McpHttpServerConfig, McpSSEServerConfig
        from pydantic_ai.builtin_tools import MCPServerTool

        from llmling_models.builtin_tools import get_claude_code_tool_name

        # Collect tools and MCP servers from builtin_tools parameter
        allowed_tools: list[str] = []
        mcp_servers: dict[str, McpSSEServerConfig | McpHttpServerConfig] = {}

        if model_request_parameters and model_request_parameters.builtin_tools:
            for tool in model_request_parameters.builtin_tools:
                # Handle MCPServerTool - convert to SDK mcp_servers format
                if isinstance(tool, MCPServerTool) and tool.url:
                    url = tool.url
                    server_config = (
                        McpSSEServerConfig(type="sse", url=url)
                        if "/sse" in url or url.endswith("/sse")
                        else McpHttpServerConfig(type="http", url=url)
                    )
                    if tool.headers:  # Add headers if provided
                        server_config["headers"] = tool.headers
                    mcp_servers[tool.id] = server_config
                    # Add MCP tools to allowed list if specified
                    if tool.allowed_tools:
                        for mcp_tool in tool.allowed_tools:
                            mcp_tool_name = f"mcp__{tool.id}__{mcp_tool}"
                            if mcp_tool_name not in allowed_tools:
                                allowed_tools.append(mcp_tool_name)
                # Handle Claude Code builtin tools
                elif (builtin_name := get_claude_code_tool_name(tool)) and (
                    builtin_name not in allowed_tools
                ):
                    allowed_tools.append(builtin_name)

        # Use `tools` parameter to control base tool set:
        # - Empty list = no tools (sends --tools "" to CLI)
        # - Non-empty list = only those tools
        # This ensures standard pydantic-ai semantics: no builtin tools = no tools
        return ClaudeAgentOptions(
            model=self._model,
            cwd=self._cwd,
            permission_mode=self._permission_mode,
            system_prompt=self._system_prompt or "",
            max_turns=self._max_turns,
            max_thinking_tokens=self._max_thinking_tokens,
            tools=allowed_tools,  # Empty list = no tools, non-empty = only those tools
            mcp_servers=mcp_servers,  # type: ignore[arg-type]  # Subset of McpServerConfig
            include_partial_messages=self._include_partial_messages,
            # Disable loading external settings by default for predictable behavior
            setting_sources=[],
        )

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a request to Claude Code."""
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
            query,
        )

        prompt_content = _extract_prompt(messages)
        options = self._build_options(model_request_parameters)
        # Override system prompt if found in messages
        if system_prompt := _extract_system_prompt(messages):
            options.system_prompt = system_prompt

        parts: list[ModelResponsePart] = []
        usage = RequestUsage()
        model_name = self.model_name

        # Collect all messages first to ensure the iterator is fully consumed
        # This prevents issues with anyio task group cleanup
        all_messages: list[Any] = []

        # Handle multimodal vs text-only prompts
        if isinstance(prompt_content, list):
            # Multimodal: use streaming format with structured content
            async def multimodal_prompt() -> AsyncIterator[dict[str, Any]]:
                yield {
                    "type": "user",
                    "message": {"role": "user", "content": prompt_content},
                }

            async for message in query(prompt=multimodal_prompt(), options=options):
                all_messages.append(message)  # noqa: PERF401
        else:
            # Text-only: use simple string format
            async for message in query(prompt=prompt_content, options=options):
                all_messages.append(message)  # noqa: PERF401

        # Now process collected messages
        # Claude Code executes tools internally. We use BuiltinToolCallPart and
        # BuiltinToolReturnPart to represent these - pydantic-ai recognizes these
        # as provider-handled tools and won't try to execute them itself.
        # Track tool use IDs to match results with their calls
        tool_use_names: dict[str, str] = {}

        for message in all_messages:
            if isinstance(message, AssistantMessage):
                model_name = f"claude-code:{message.model}"
                for block in message.content:
                    if isinstance(block, TextBlock):
                        parts.append(TextPart(content=block.text))
                    elif isinstance(block, ThinkingBlock):
                        parts.append(
                            ThinkingPart(content=block.thinking, signature=block.signature)
                        )
                    elif isinstance(block, ToolUseBlock):
                        # Track tool name for matching with results
                        tool_use_names[block.id] = block.name
                        # Emit as BuiltinToolCallPart - pydantic-ai won't try to execute
                        parts.append(
                            BuiltinToolCallPart(
                                tool_name=block.name,
                                args=block.input,
                                tool_call_id=block.id,
                                provider_name="claude-code",
                            )
                        )

            elif isinstance(message, UserMessage):
                # UserMessage contains tool results from Claude Code's execution
                if isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            # Get tool name from our tracking dict
                            tool_name = tool_use_names.get(block.tool_use_id, "unknown")
                            # Convert content to string
                            if isinstance(block.content, str):
                                content = block.content
                            elif block.content is None:
                                content = ""
                            else:
                                # List of content blocks - convert to string
                                content = json.dumps(block.content)
                            parts.append(
                                BuiltinToolReturnPart(
                                    tool_name=tool_name,
                                    content=content,
                                    tool_call_id=block.tool_use_id,
                                    provider_name="claude-code",
                                )
                            )

            elif isinstance(message, ResultMessage):
                if message.usage:
                    u = message.usage
                    usage = RequestUsage(
                        input_tokens=u.get("input_tokens", 0),
                        output_tokens=u.get("output_tokens", 0),
                        cache_read_tokens=u.get("cache_read_input_tokens", 0),
                        cache_write_tokens=u.get("cache_creation_input_tokens", 0),
                    )
        now = datetime.now(UTC)
        return ModelResponse(parts=parts, usage=usage, model_name=model_name, timestamp=now)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Stream responses from Claude Code.

        Uses an asyncio.Queue to decouple the Claude Agent SDK's task group
        from pydantic-ai's streaming, enabling true real-time streaming.
        """
        from claude_agent_sdk import query

        prompt_content = _extract_prompt(messages)
        options = self._build_options(model_request_parameters)
        # Override system prompt if found in messages
        if system_prompt := _extract_system_prompt(messages):
            options.system_prompt = system_prompt
        # Use a queue to decouple SDK iteration from streaming
        message_queue: asyncio.Queue[Any] = asyncio.Queue()
        collection_done = asyncio.Event()
        collection_error: list[Exception] = []

        async def collect_messages() -> None:
            """Collect messages from SDK in background task."""
            try:
                # Handle multimodal vs text-only prompts
                if isinstance(prompt_content, list):
                    # Multimodal: use streaming format with structured content
                    async def multimodal_prompt() -> AsyncIterator[dict[str, Any]]:
                        yield {
                            "type": "user",
                            "message": {"role": "user", "content": prompt_content},
                        }

                    async for message in query(prompt=multimodal_prompt(), options=options):
                        await message_queue.put(message)
                else:
                    # Text-only: use simple string format
                    async for message in query(prompt=prompt_content, options=options):
                        await message_queue.put(message)
            except Exception as e:  # noqa: BLE001
                collection_error.append(e)
            finally:
                collection_done.set()

        # Start collection in background
        collection_task = asyncio.create_task(collect_messages())

        try:
            yield ClaudeCodeStreamedResponse(
                model_request_parameters=model_request_parameters,
                _message_queue=message_queue,
                _collection_done=collection_done,
                _collection_error=collection_error,
                _model_name=self.model_name,
            )
        finally:
            # Ensure collection task completes
            if not collection_task.done():
                # Wait for it to finish naturally
                await collection_done.wait()
            # Check for errors
            if collection_error:
                raise collection_error[0]


if __name__ == "__main__":
    from pydantic_ai import Agent

    async def test() -> None:
        model = ClaudeCodeModel(model="haiku", permission_mode="bypassPermissions")
        agent: Agent[None, str] = Agent(model=model)

        from llmling_models import ClaudeCodeGlobTool, ClaudeCodeReadTool

        print("Testing Claude Code Model (streaming, with tools):")
        async with agent.run_stream(
            "What files are in the current directory?",
            builtin_tools=[ClaudeCodeGlobTool(), ClaudeCodeReadTool()],
        ) as result:
            async for event in result.stream_text(delta=True):
                print(event, end="", flush=True)
        print("\n--- Done ---")

    asyncio.run(test())
