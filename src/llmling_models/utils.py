"""Utility functions for model handling."""

from __future__ import annotations

from decimal import Decimal
import logging
from typing import TYPE_CHECKING

from pydantic import ConfigDict, TypeAdapter
from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pydantic_ai import ModelMessage
    from tokonomics import ModelCosts


def estimate_tokens(messages: list[ModelMessage]) -> int:
    """Estimate total content tokens for a list of messages.

    This function estimates the token count for message content that would be
    sent to a model. It's primarily used for pre-request estimation to help
    with model selection based on token limits and input costs.

    Note: This estimates content tokens, not usage tokens. For actual token
    usage from completed requests, use ModelResponse.usage directly.
    """
    import tokonomics

    content = ""
    for message in messages:
        for part in message.parts:
            if isinstance(
                part,
                UserPromptPart | SystemPromptPart | TextPart | ToolReturnPart,
            ):
                content += str(part.content)
    return tokonomics.count_tokens(content)


def estimate_request_cost(costs: ModelCosts, token_count: int) -> Decimal:
    """Estimate input cost for a request.

    Args:
        costs: Cost information (dict or ModelCosts object)
        token_count: Number of tokens in the request

    Returns:
        Decimal: Estimated input cost in USD
    """
    # Extract input cost per token
    input_cost = Decimal(costs["input_cost_per_token"])
    estimated_cost = input_cost * token_count
    msg = "Estimated cost: %s * %d tokens = %s"
    logger.debug(msg, input_cost, token_count, estimated_cost)
    return estimated_cost


def without_unprocessed_tool_calls(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Clean message history by removing unprocessed tool calls.

    This removes ToolCallPart from the last ModelResponse if it has unprocessed
    tool calls, but preserves all text content and reasoning.
    """
    if not messages:
        return []
    cleaned_messages = list(messages)  # Make a copy to avoid modifying the original
    last_message = cleaned_messages[-1]
    if isinstance(last_message, ModelResponse) and last_message.tool_calls:
        # Create a new ModelResponse with the same content but without tool calls
        filtered = [p for p in last_message.parts if not isinstance(p, ToolCallPart)]
        # Only replace if we actually removed some tool calls
        if len(filtered) != len(last_message.parts):
            # Create a new ModelResponse with filtered parts
            cleaned_response = ModelResponse(
                parts=filtered,
                usage=last_message.usage,
                model_name=last_message.model_name,
                timestamp=last_message.timestamp,
                provider_name=last_message.provider_name,
                provider_details=last_message.provider_details,
                provider_response_id=last_message.provider_response_id,
                finish_reason=last_message.finish_reason,
            )
            cleaned_messages[-1] = cleaned_response

    return cleaned_messages


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
