from typing import Annotated, Literal

from pydantic import Field
from pydantic_ai.models import AgentModel, KnownModelName, Model, infer_model
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

from llmling_models import RandomMultiModel
from llmling_models.base import PydanticModel


class _TestModelWrapper(PydanticModel):
    """Wrapper for TestModel."""

    type: Literal["test"] = Field(default="test", init=False)
    model: TestModel

    def name(self) -> str:
        """Get model name."""
        return "test"

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create agent model implementation."""
        return await self.model.agent_model(
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )


class StringModel(PydanticModel):
    """Wrapper for string model names."""

    type: Literal["string"] = Field(default="string", init=False)
    identifier: KnownModelName  # renamed from name

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create agent model from string name."""
        model = infer_model(self.identifier)  # type: ignore
        return await model.agent_model(
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )

    def name(self) -> str:
        """Get model name."""
        return str(self.identifier)


type ModelInput = str | KnownModelName | Model | PydanticModel
"""Type for internal model handling (after validation)."""

AnyModel = Annotated[
    StringModel | RandomMultiModel | _TestModelWrapper, Field(discriminator="type")
]
