"""Model that imports and delegates to other models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, ImportString

from llmling_models.base import PydanticModel
from llmling_models.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models import ModelRequestParameters, StreamedResponse
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.usage import Usage

logger = get_logger(__name__)


class ImportModel(PydanticModel):
    """Model that imports and delegates to other models.

    Useful to allow using "external" models via YAML in LLMling-Agent

    Example YAML configuration:
        ```yaml
        models:
          custom-model:
            type: import
            model: my_package.models:CustomModel
            kw_args:
              param1: value1
              param2: value2
        ```
    """

    type: Literal["import"] = Field(default="import", init=False)

    model: ImportString = Field(...)
    """Model class to import and use."""

    kw_args: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments for the imported model class."""

    def model_post_init(self, __context: dict[str, Any], /) -> None:
        """Initialize model instance if needed."""
        self._instance = (
            self.model(**self.kw_args) if isinstance(self.model, type) else self.model
        )

    def name(self) -> str:
        """Get model name."""
        return f"import:{self._instance.name()}"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, Usage]:
        """Delegate request to imported model."""
        return await self._instance.request(
            messages,
            model_settings,
            model_request_parameters,
        )

    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        """Delegate streaming request to imported model."""
        async with self._instance.request_stream(
            messages,
            model_settings,
            model_request_parameters,
        ) as stream:
            yield stream


if __name__ == "__main__":
    import asyncio

    from pydantic_ai import Agent

    async def test_conversation():
        """Test the import model with an InputModel."""
        model = ImportModel(model="llmling_models.inputmodel:InputModel")

        agent: Agent[None, str] = Agent(
            model=model,
            system_prompt="You are helping test an import model.",
        )

        # Test regular request
        result = await agent.run("What's your favorite color?")
        print(f"\nFirst response: {result.data}")

        # Test streaming
        print("\nTesting streaming:")
        async with agent.run_stream("Tell me a story") as stream:
            async for chunk in stream.stream_text(delta=True):
                print(chunk, end="", flush=True)
        print("\nStreaming complete!")

    asyncio.run(test_conversation())
