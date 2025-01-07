import importlib.util
from typing import Annotated, Literal

from pydantic import Field
from pydantic_ai.models import AgentModel, KnownModelName, Model, infer_model
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

from llmling_models import (
    CostOptimizedMultiModel,
    DelegationMultiModel,
    FallbackMultiModel,
    ImportModel,
    InputModel,
    PydanticModel,
    TokenOptimizedMultiModel,
)


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

if importlib.util.find_spec("fastapi"):
    from llmling_models.remote_input.client import RemoteInputModel
    from llmling_models.remote_model.client import RemoteProxyModel

    AnyModel = Annotated[
        StringModel
        | DelegationMultiModel
        | CostOptimizedMultiModel
        | TokenOptimizedMultiModel
        | FallbackMultiModel
        | InputModel
        | ImportModel
        | _TestModelWrapper
        | RemoteInputModel
        | RemoteProxyModel,
        Field(discriminator="type"),
    ]
else:
    AnyModel = Annotated[  # type: ignore
        StringModel
        | DelegationMultiModel
        | CostOptimizedMultiModel
        | TokenOptimizedMultiModel
        | FallbackMultiModel
        | InputModel
        | ImportModel
        | _TestModelWrapper,
        Field(discriminator="type"),
    ]
