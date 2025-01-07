"""Adapter to use AISuite library models with Pydantic-AI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import aisuite
from pydantic import Field
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import AgentModel, EitherStreamedResponse
from pydantic_ai.result import Usage

from llmling_models.base import PydanticModel


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from pydantic_ai.settings import ModelSettings


class AISuiteAdapter(PydanticModel):
    """Adapter to use AISuite library models with Pydantic-AI.

    Example YAML configuration:
        ```yaml
        models:
          anthropic-claude:
            type: aisuite
            model: anthropic:claude-3-opus-20240229
            config:
              anthropic:
                api_key: your-api-key-here
        ```
    """

    type: Literal["aisuite"] = Field(default="aisuite", init=False)

    model: str
    """Model identifier in provider:model format"""

    config: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """"Provider configurations."""

    _client: aisuite.Client | None = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._client = aisuite.Client(self.config)

    def name(self) -> str:
        """Return the model name."""
        return f"aisuite:{self.model}"

    async def agent_model(
        self,
        *,
        function_tools: list[Any],  # type: ignore
        allow_text_result: bool,
        result_tools: list[Any],  # type: ignore
    ) -> AgentModel:
        """Create an agent model."""
        assert self._client
        return AISuiteAgentModel(client=self._client, model=self.model)


@dataclass
class AISuiteAgentModel(AgentModel):
    """AgentModel implementation for AISuite."""

    client: aisuite.Client
    model: str

    async def request(
        self,
        messages: Sequence[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> tuple[ModelResponse, Usage]:
        """Make a request to the model."""
        formatted_messages = []

        # Convert messages to AISuite format
        for message in messages:
            if isinstance(message, ModelResponse):
                formatted_messages.append({
                    "role": "assistant",
                    "content": str(message.parts[0].content),  # type: ignore
                })
            else:  # ModelRequest
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        formatted_messages.append({
                            "role": "system",
                            "content": part.content,
                        })
                    elif isinstance(part, UserPromptPart):
                        formatted_messages.append({
                            "role": "user",
                            "content": part.content,
                        })

        # Extract settings
        kwargs = {}
        if model_settings:
            if hasattr(model_settings, "temperature"):
                kwargs["temperature"] = model_settings.temperature  # type: ignore
            if hasattr(model_settings, "max_tokens"):
                kwargs["max_tokens"] = model_settings.max_tokens  # type: ignore

        # Make request to AISuite
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            **kwargs,
        )

        # Extract response content
        content = response.choices[0].message.content

        return ModelResponse(
            parts=[TextPart(content)],
            timestamp=datetime.now(UTC),
        ), Usage()  # AISuite doesn't provide token counts yet

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> AsyncIterator[EitherStreamedResponse]:
        """Streaming is not supported yet."""
        msg = "Streaming not supported by AISuite adapter"
        raise NotImplementedError(msg) from None
        # Need to yield even though we raise an error
        # to satisfy the async context manager protocol
        if False:  # pragma: no cover
            yield None


if __name__ == "__main__":
    import asyncio

    from pydantic_ai import Agent

    async def test():
        adapter = AISuiteAdapter(
            model="openai:gpt-4o-mini",
            config={
                "anthropic": {"api_key": "your-api-key"},
            },
        )
        agent: Agent[None, str] = Agent(model=adapter)
        response = await agent.run("Say hello!")
        print(response.data)

    asyncio.run(test())
