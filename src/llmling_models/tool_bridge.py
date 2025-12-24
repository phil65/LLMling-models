"""MCP server bridge for exposing Python functions to Claude Code.

This module provides a bridge that exposes Python functions as an MCP server
using HTTP transport. This allows Claude Code (via ClaudeCodeModel) to use custom
tools defined as simple Python functions.

The bridge runs in-process and handles the MCP protocol automatically.

Example:
    ```python
    from pydantic_ai import Agent
    from llmling_models import ClaudeCodeModel
    from llmling_models.tool_bridge import ToolBridge

    # Define custom tools as simple functions
    def greet(name: str) -> str:
        '''Greet a person by name.'''
        return f"Hello, {name}!"

    def multiply(a: int, b: int) -> int:
        '''Multiply two numbers.'''
        return a * b

    # Create bridge exposing the functions
    async with ToolBridge(tools=[greet, multiply]) as bridge:
        # Get MCP server config for Claude Code
        mcp_tool = bridge.get_mcp_server_tool()

        # Use with ClaudeCodeModel
        model = ClaudeCodeModel()
        agent = Agent(model=model)
        result = await agent.run(
            "Greet Alice and multiply 6 by 7",
            builtin_tools=[mcp_tool],
        )
    ```
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
import inspect
import socket
from typing import TYPE_CHECKING, Any, Self, get_type_hints

from pydantic_ai import MCPServerTool


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from fastmcp import Context, FastMCP
    from fastmcp.tools.tool import ToolResult
    from uvicorn import Server


def _convert_to_tool_result(result: Any) -> ToolResult:
    """Convert a tool's return value to a FastMCP ToolResult.

    Handles different result types appropriately:
    - ToolResult: Pass through unchanged
    - dict: Use as structured_content (enables programmatic access by clients)
    - Pydantic models: Serialize to dict for structured_content
    - Other types: Pass to ToolResult(content=...) which handles conversion internally
      (including str, list, ContentBlock, Image, Audio, File, primitives)
    """
    from fastmcp.tools.tool import ToolResult
    from pydantic import BaseModel

    # Already a ToolResult - pass through
    if isinstance(result, ToolResult):
        return result

    # Dict - use as structured_content (FastMCP auto-populates content as JSON)
    if isinstance(result, dict):
        return ToolResult(structured_content=result)

    # Pydantic model - serialize to dict for structured_content
    if isinstance(result, BaseModel):
        return ToolResult(structured_content=result.model_dump(mode="json"))

    # All other types (str, list, ContentBlock, Image, None, primitives, etc.)
    # ToolResult's internal _convert_to_content handles these correctly
    if result is None:
        return ToolResult(content="")
    return ToolResult(content=result)


def _get_function_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Extract JSON schema from a function's signature and docstring."""
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

    # Build properties from parameters
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        prop: dict[str, Any] = {}
        hint = hints.get(name)

        # Map Python types to JSON schema types
        if hint is str:
            prop["type"] = "string"
        elif hint is int:
            prop["type"] = "integer"
        elif hint is float:
            prop["type"] = "number"
        elif hint is bool:
            prop["type"] = "boolean"
        elif hint is list:
            prop["type"] = "array"
        elif hint is dict:
            prop["type"] = "object"
        elif hint is not None and hasattr(hint, "__origin__"):
            origin = getattr(hint, "__origin__", None)
            if origin is list:
                prop["type"] = "array"
            elif origin is dict:
                prop["type"] = "object"
            else:
                prop["type"] = "string"
        else:
            prop["type"] = "string"  # Default fallback

        properties[name] = prop

        # Check if required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _get_function_description(func: Callable[..., Any]) -> str:
    """Extract description from function docstring."""
    doc = inspect.getdoc(func)
    if doc:
        # Return first line/paragraph
        return doc.split("\n\n")[0].strip()
    name = getattr(func, "__name__", "unknown")
    return f"Call the {name} function"


@dataclass
class ToolBridgeConfig:
    """Configuration for the ToolBridge."""

    host: str = "127.0.0.1"
    """Host to bind the HTTP server to."""

    port: int = 0
    """Port to bind to (0 = auto-select available port)."""

    transport: str = "sse"
    """Transport protocol: 'sse' or 'streamable-http'."""

    server_name: str = "pydantic-ai-tools"
    """Name for the MCP server."""


@dataclass
class ToolBridge:
    """Exposes Python functions as an MCP server for Claude Code.

    This bridge allows ClaudeCodeModel to access custom Python functions
    via MCP transport.

    Example:
        ```python
        def my_tool(x: int) -> int:
            '''Double the input.'''
            return x * 2

        async with ToolBridge(tools=[my_tool]) as bridge:
            mcp_tool = bridge.get_mcp_server_tool()
            # Pass mcp_tool to agent's builtin_tools
        ```
    """

    tools: list[Callable[..., Any]]
    """The functions to expose as tools."""

    config: ToolBridgeConfig = field(default_factory=ToolBridgeConfig)
    """Bridge configuration."""

    _mcp: FastMCP | None = field(default=None, init=False, repr=False)
    """FastMCP server instance."""

    _server: Server | None = field(default=None, init=False, repr=False)
    """Uvicorn server instance."""

    _server_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    """Background task running the server."""

    _actual_port: int | None = field(default=None, init=False, repr=False)
    """Actual port the server is bound to."""

    _tool_funcs: dict[str, Callable[..., Any]] = field(default_factory=dict, init=False, repr=False)
    """Mapping of tool names to their callables."""

    async def __aenter__(self) -> Self:
        """Start the MCP server."""
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Stop the MCP server."""
        await self.stop()

    async def start(self) -> None:
        """Start the HTTP MCP server in the background."""
        from fastmcp import FastMCP

        self._mcp = FastMCP(name=self.config.server_name)
        self._register_tools()
        await self._start_server()

    async def stop(self) -> None:
        """Stop the HTTP MCP server."""
        if self._server:
            self._server.should_exit = True
            if self._server_task:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5.0)
                except TimeoutError:
                    self._server_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._server_task
            self._server = None
            self._server_task = None
        self._mcp = None
        self._actual_port = None
        self._tool_funcs.clear()

    @property
    def port(self) -> int:
        """Get the actual port the server is running on."""
        if self._actual_port is None:
            msg = "Server not started"
            raise RuntimeError(msg)
        return self._actual_port

    @property
    def url(self) -> str:
        """Get the server URL."""
        path = "/sse" if self.config.transport == "sse" else "/mcp"
        return f"http://{self.config.host}:{self.port}{path}"

    def get_mcp_server_tool(self) -> MCPServerTool:
        """Get pydantic-ai MCPServerTool for use with ClaudeCodeModel.

        Returns:
            MCPServerTool configured to connect to this bridge.
        """
        return MCPServerTool(
            id=self.config.server_name,
            url=self.url,
        )

    def _register_tools(self) -> None:
        """Register all functions with the FastMCP server."""
        if not self._mcp:
            return

        for func in self.tools:
            self._register_single_tool(func)

    def _register_single_tool(self, func: Callable[..., Any]) -> None:
        """Register a single function with the FastMCP server."""
        if not self._mcp:
            return

        from fastmcp.tools import Tool as FastMCPTool

        name = getattr(func, "__name__", "unknown")
        description = _get_function_description(func)
        parameters = _get_function_schema(func)

        # Store the function for invocation
        self._tool_funcs[name] = func

        # Create wrapper that bridges to our invocation
        bridge = self

        class BridgedTool(FastMCPTool):
            """FastMCP tool that wraps a Python function."""

            def __init__(self) -> None:
                super().__init__(
                    name=name,
                    description=description,
                    parameters=parameters,
                )

            async def run(
                self, arguments: dict[str, Any], context: Context | None = None
            ) -> ToolResult:
                """Execute the wrapped function."""
                result = await bridge._invoke_tool(name, arguments)
                return _convert_to_tool_result(result)

        self._mcp.add_tool(BridgedTool())

    async def _invoke_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a tool function."""
        if tool_name not in self._tool_funcs:
            msg = f"Tool {tool_name!r} not found"
            raise KeyError(msg)

        func = self._tool_funcs[tool_name]

        # Call the function
        result = func(**arguments)

        # Handle async functions
        if inspect.isawaitable(result):
            result = await result

        return result

    async def _start_server(self) -> None:
        """Start the uvicorn server in the background."""
        import uvicorn

        if not self._mcp:
            msg = "MCP server not initialized"
            raise RuntimeError(msg)

        # Determine actual port (auto-select if 0)
        port = self.config.port
        if port == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.config.host, 0))
                port = s.getsockname()[1]
        self._actual_port = port

        # Create the ASGI app
        app = self._mcp.http_app(transport=self.config.transport)  # type: ignore[arg-type]

        # Configure uvicorn
        cfg = uvicorn.Config(
            app=app,
            host=self.config.host,
            port=port,
            log_level="warning",
        )
        self._server = uvicorn.Server(cfg)

        # Start server in background task
        self._server_task = asyncio.create_task(
            self._server.serve(),
            name=f"tool-bridge-{self.config.server_name}",
        )

        # Wait briefly for server to start
        await asyncio.sleep(0.1)


@asynccontextmanager
async def create_tool_bridge(
    tools: list[Callable[..., Any]],
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    transport: str = "sse",
    server_name: str = "pydantic-ai-tools",
) -> AsyncIterator[ToolBridge]:
    """Create and start a ToolBridge as a context manager.

    Args:
        tools: Functions to expose as MCP tools
        host: Host to bind to
        port: Port to bind to (0 = auto-select)
        transport: Transport protocol ('sse' or 'streamable-http')
        server_name: Name for the MCP server

    Yields:
        Running ToolBridge instance

    Example:
        ```python
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        async with create_tool_bridge([greet]) as bridge:
            mcp_tool = bridge.get_mcp_server_tool()
            # Use mcp_tool with ClaudeCodeModel
        ```
    """
    config = ToolBridgeConfig(
        host=host,
        port=port,
        transport=transport,
        server_name=server_name,
    )
    bridge = ToolBridge(tools=tools, config=config)
    async with bridge:
        yield bridge
