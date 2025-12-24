"""Utility functions for model handling."""

from __future__ import annotations

import logging

from pydantic import ConfigDict, TypeAdapter
from pydantic_ai import ModelRequest, ModelResponse, RetryPromptPart


logger = logging.getLogger(__name__)


PydanticAIMessage = ModelRequest | ModelResponse
message_adapter: TypeAdapter[PydanticAIMessage] = TypeAdapter(
    PydanticAIMessage,
    config=ConfigDict(ser_json_bytes="base64", val_json_bytes="base64"),
)


def serialize_message(message: PydanticAIMessage) -> str:
    """Serialize pydantic-ai message.

    The `ctx` field in the `RetryPromptPart` is optionally dict[str, Any],
    which is not always serializable.
    """
    for part in message.parts:
        if isinstance(part, RetryPromptPart) and isinstance(part.content, list):
            for content in part.content:
                content["ctx"] = {k: str(v) for k, v in (content.get("ctx", None) or {}).items()}
    return message_adapter.dump_python(message, mode="json")  # type: ignore[no-any-return]
