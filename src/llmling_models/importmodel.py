"""Model that imports and delegates to other models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, ImportString

from llmling_models.base import PydanticModel
from llmling_models.log import get_logger


if TYPE_CHECKING:
    from pydantic_ai.models import AgentModel
    from pydantic_ai.tools import ToolDefinition

logger = get_logger(__name__)


class ImportModel(PydanticModel):
    """Model that imports and delegates to other models."""

    type: Literal["import"] = Field(default="import", init=False)

    model: ImportString = Field(...)
    """Model class to import and use."""

    # _instance: Model | None = Field(default=None, exclude=True)
    # """Cached model instance."""

    def model_post_init(self, __context: dict[str, Any], /):
        """Initialize model instance if needed."""
        self._instance = self.model() if isinstance(self.model, type) else self.model

    def name(self) -> str:
        """Get model name."""
        return f"import:{self._instance.name()}"

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create agent model implementation."""
        return await self._instance.agent_model(
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )


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

        # Ask a question through the imported input model
        result = await agent.run("What's your favorite color?")
        print(f"\nFirst response: {result.data}")

        # Follow-up to test conversation context
        result = await agent.run("Why do you like that color?")
        print(f"\nSecond response: {result.data}")

    asyncio.run(test_conversation())
