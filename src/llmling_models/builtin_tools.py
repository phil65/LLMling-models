"""Builtin tools for Claude Code model.

These tools represent the provider-side tools available in Claude Code.
They can be passed to the agent to configure which tools are available.

Example:
    from pydantic_ai import Agent
    from llmling_models import ClaudeCodeModel
    from llmling_models.builtin_tools import (
        ClaudeCodeReadTool,
        ClaudeCodeWriteTool,
        ClaudeCodeBashTool,
    )

    model = ClaudeCodeModel()
    agent = Agent(
        model=model,
        builtin_tools=[
            ClaudeCodeReadTool(),
            ClaudeCodeWriteTool(),
            ClaudeCodeBashTool(),
        ],
    )
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.builtin_tools import AbstractBuiltinTool


@dataclass(kw_only=True)
class ClaudeCodeReadTool(AbstractBuiltinTool):
    """Read file contents via Claude Code.

    Allows Claude Code to read files from the filesystem.
    """

    kind: str = "claude_code_read"
    """The kind of tool."""


@dataclass(kw_only=True)
class ClaudeCodeWriteTool(AbstractBuiltinTool):
    """Write file contents via Claude Code.

    Allows Claude Code to write files to the filesystem.
    """

    kind: str = "claude_code_write"
    """The kind of tool."""


@dataclass(kw_only=True)
class ClaudeCodeEditTool(AbstractBuiltinTool):
    """Edit file contents via Claude Code.

    Allows Claude Code to make targeted edits to files.
    """

    kind: str = "claude_code_edit"
    """The kind of tool."""


@dataclass(kw_only=True)
class ClaudeCodeBashTool(AbstractBuiltinTool):
    """Execute bash commands via Claude Code.

    Allows Claude Code to run shell commands.
    """

    kind: str = "claude_code_bash"
    """The kind of tool."""


@dataclass(kw_only=True)
class ClaudeCodeGlobTool(AbstractBuiltinTool):
    """Find files using glob patterns via Claude Code.

    Allows Claude Code to search for files by pattern.
    """

    kind: str = "claude_code_glob"
    """The kind of tool."""


@dataclass(kw_only=True)
class ClaudeCodeGrepTool(AbstractBuiltinTool):
    """Search file contents via Claude Code.

    Allows Claude Code to search within files using regex.
    """

    kind: str = "claude_code_grep"
    """The kind of tool."""


@dataclass(kw_only=True)
class ClaudeCodeWebSearchTool(AbstractBuiltinTool):
    """Search the web via Claude Code.

    Allows Claude Code to perform web searches.
    """

    kind: str = "claude_code_web_search"
    """The kind of tool."""


@dataclass(kw_only=True)
class ClaudeCodeWebFetchTool(AbstractBuiltinTool):
    """Fetch web content via Claude Code.

    Allows Claude Code to retrieve content from URLs.
    """

    kind: str = "claude_code_web_fetch"
    """The kind of tool."""


@dataclass(kw_only=True)
class ClaudeCodeTaskTool(AbstractBuiltinTool):
    """Spawn sub-agents via Claude Code.

    Allows Claude Code to delegate work to sub-agents.
    """

    kind: str = "claude_code_task"
    """The kind of tool."""


@dataclass(kw_only=True)
class ClaudeCodeNotebookEditTool(AbstractBuiltinTool):
    """Edit Jupyter notebooks via Claude Code.

    Allows Claude Code to modify notebook cells.
    """

    kind: str = "claude_code_notebook_edit"
    """The kind of tool."""


# Mapping from builtin tool to Claude Code tool name
BUILTIN_TOOL_NAMES: dict[type[AbstractBuiltinTool], str] = {
    ClaudeCodeReadTool: "Read",
    ClaudeCodeWriteTool: "Write",
    ClaudeCodeEditTool: "Edit",
    ClaudeCodeBashTool: "Bash",
    ClaudeCodeGlobTool: "Glob",
    ClaudeCodeGrepTool: "Grep",
    ClaudeCodeWebSearchTool: "WebSearch",
    ClaudeCodeWebFetchTool: "WebFetch",
    ClaudeCodeTaskTool: "Task",
    ClaudeCodeNotebookEditTool: "NotebookEdit",
}


def get_claude_code_tool_name(tool: AbstractBuiltinTool) -> str | None:
    """Get the Claude Code tool name for a builtin tool."""
    return BUILTIN_TOOL_NAMES.get(type(tool))


def claude_code_all_tools() -> list[AbstractBuiltinTool]:
    """Return all Claude Code builtin tools.

    Convenience function to enable all Claude Code tools at once.

    Example:
        from pydantic_ai import Agent
        from llmling_models import ClaudeCodeModel
        from llmling_models.builtin_tools import claude_code_all_tools

        model = ClaudeCodeModel()
        agent = Agent(model=model, builtin_tools=claude_code_all_tools())
    """
    return [
        ClaudeCodeReadTool(),
        ClaudeCodeWriteTool(),
        ClaudeCodeEditTool(),
        ClaudeCodeBashTool(),
        ClaudeCodeGlobTool(),
        ClaudeCodeGrepTool(),
        ClaudeCodeWebSearchTool(),
        ClaudeCodeWebFetchTool(),
        ClaudeCodeTaskTool(),
        ClaudeCodeNotebookEditTool(),
    ]


def claude_code_read_only_tools() -> list[AbstractBuiltinTool]:
    """Return read-only Claude Code tools (no writes, edits, or bash).

    Convenience function for safe, read-only operations.

    Example:
        agent = Agent(model=model, builtin_tools=claude_code_read_only_tools())
    """
    return [
        ClaudeCodeReadTool(),
        ClaudeCodeGlobTool(),
        ClaudeCodeGrepTool(),
        ClaudeCodeWebSearchTool(),
        ClaudeCodeWebFetchTool(),
    ]


__all__ = [
    "BUILTIN_TOOL_NAMES",
    "ClaudeCodeBashTool",
    "ClaudeCodeEditTool",
    "ClaudeCodeGlobTool",
    "ClaudeCodeGrepTool",
    "ClaudeCodeNotebookEditTool",
    "ClaudeCodeReadTool",
    "ClaudeCodeTaskTool",
    "ClaudeCodeWebFetchTool",
    "ClaudeCodeWebSearchTool",
    "ClaudeCodeWriteTool",
    "claude_code_all_tools",
    "claude_code_read_only_tools",
    "get_claude_code_tool_name",
]
