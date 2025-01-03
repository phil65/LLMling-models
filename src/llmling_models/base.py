"""Base class for YAML-configurable pydantic-ai models."""

from __future__ import annotations

from abc import ABC
import importlib.util
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict
from pydantic_ai.messages import (
    ModelMessage,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model


if TYPE_CHECKING:
    from tokonomics import ModelCosts, TokenLimits


class PydanticModel(Model, BaseModel, ABC):
    """Base for models that can be configured via YAML."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        use_attribute_docstrings=True,
    )

    @staticmethod
    async def get_model_costs(model_name: str) -> ModelCosts | None:
        """Get costs for model using tokonomics."""
        from tokonomics import get_model_costs

        return await get_model_costs(model_name)

    @staticmethod
    async def get_model_limits(model_name: str) -> TokenLimits | None:
        """Get token limits for model using tokonomics."""
        from tokonomics import get_model_limits

        return await get_model_limits(model_name)

    @staticmethod
    def estimate_tokens(messages: list[ModelMessage]) -> int:
        """Estimate token count for messages using available tokenizers.

        Will try to use tiktoken if available (best for OpenAI models),
        falling back to Mistral's tokenizer (good modern default),
        and finally using a simple character-based estimation.
        """
        # Collect all content from relevant message parts
        content = ""
        for message in messages:
            for part in message.parts:
                if isinstance(
                    part,
                    UserPromptPart | SystemPromptPart | TextPart | ToolReturnPart,
                ):
                    content += part.content

        # Try tiktoken (best for OpenAI models)
        if importlib.util.find_spec("tiktoken"):
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(content))

        # Try transformers with Mistral's tokenizer
        if importlib.util.find_spec("transformers"):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
            return len(tokenizer.encode(content))

        # Fallback to simple character-based estimation
        return len(content) // 4
