"""Utilities for working with pydantic-ai types and objects."""

from __future__ import annotations

from pydantic_ai import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    FileUrl,
    ImageUrl,
    VideoUrl,
    messages,
)


def format_part(  # noqa: PLR0911
    response: str | messages.ModelRequestPart | messages.ModelResponsePart,
) -> str:
    """Format any kind of response part in a readable way.

    Args:
        response: Response part to format

    Returns:
        A human-readable string representation
    """
    match response:
        case str():
            return response
        case messages.ToolCallPart(args=args, tool_name=tool_name):
            return f"Tool call: {tool_name}\nArgs: {args}"
        case messages.ToolReturnPart(tool_name=tool_name, content=content):
            return f"Tool {tool_name} returned: {content}"
        case messages.RetryPromptPart(content=content) if isinstance(content, str):
            return f"Retry needed: {content}"
        case messages.RetryPromptPart(content=content):
            return f"Validation errors:\n{content}"
        case messages.UserPromptPart(content=content) if not isinstance(content, str):
            texts = []
            for item in content:
                match item:
                    case str():
                        texts.append(f"{item}")
                    case (
                        DocumentUrl(url=url)
                        | ImageUrl(url=url)
                        | AudioUrl(url=url)
                        | VideoUrl(url=url)
                        | FileUrl(url=url)
                    ):
                        texts.append(f"{url}")
                    case BinaryContent(identifier=identifier):
                        texts.append(f"Binary content: <{identifier}>")
            return "\n".join(texts)
        case (
            messages.SystemPromptPart(content=content)
            | messages.UserPromptPart(content=content)
            | messages.TextPart(content=content)
        ) if isinstance(content, str):
            return content
        case _:
            return str(response)
