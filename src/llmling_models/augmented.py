"""Models with pre/post prompt processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
)
from pydantic_ai.models import AgentModel, KnownModelName, Model, infer_model
from pydantic_ai.result import Cost

from llmling_models.base import PydanticModel


if TYPE_CHECKING:
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition


class PrePostPromptConfig(BaseModel):
    """Configuration for pre/post prompts."""

    text: str
    model: KnownModelName | Model

    @property
    def model_instance(self) -> Model:
        """Get model instance."""
        if isinstance(self.model, str):
            return infer_model(self.model)
        return self.model

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AugmentedModel(PydanticModel):
    """Model with pre/post prompt processing.

    Example YAML configuration:
        ```yaml
        models:
          enhanced_gpt4:
            type: augmented
            main_model: openai:gpt-4
            pre_prompt:
              text: "Expand this question: {input}"
              model: openai:gpt-3.5-turbo
            post_prompt:
              text: "Summarize your response."
              model: openai:gpt-3.5-turbo
        ```
    """

    type: Literal["augmented"] = "augmented"
    main_model: KnownModelName | Model
    pre_prompt: PrePostPromptConfig | None = None
    post_prompt: PrePostPromptConfig | None = None

    def name(self) -> str:
        """Get descriptive model name."""
        base = str(self.main_model)
        if self.pre_prompt or self.post_prompt:
            return f"augmented({base})"
        return base

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create agent model with prompt augmentation."""
        return _AugmentedAgentModel(
            main_model=infer_model(self.main_model),
            pre_prompt=self.pre_prompt,
            post_prompt=self.post_prompt,
            function_tools=function_tools,
            allow_text_result=allow_text_result,
            result_tools=result_tools,
        )


class _AugmentedAgentModel(AgentModel):
    """AgentModel implementation for augmented models."""

    def __init__(
        self,
        main_model: Model,
        pre_prompt: PrePostPromptConfig | None,
        post_prompt: PrePostPromptConfig | None,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ):
        self.main_model = main_model
        self.pre_prompt = pre_prompt
        self.post_prompt = post_prompt
        self.function_tools = function_tools
        self.allow_text_result = allow_text_result
        self.result_tools = result_tools
        self._initialized_models: dict[str, AgentModel] | None = None

    async def _initialize_models(self) -> dict[str, AgentModel]:
        """Initialize all required models."""
        if self._initialized_models is None:
            self._initialized_models = {}

            # Initialize main model
            self._initialized_models["main"] = await self.main_model.agent_model(
                function_tools=self.function_tools,
                allow_text_result=self.allow_text_result,
                result_tools=self.result_tools,
            )

            # Initialize pre/post models if needed
            if self.pre_prompt:
                pre_result = await self.pre_prompt.model_instance.agent_model(
                    function_tools=[],  # No tools for auxiliary prompts
                    allow_text_result=True,
                    result_tools=[],
                )
                self._initialized_models["pre"] = pre_result

            if self.post_prompt:
                post_result = await self.post_prompt.model_instance.agent_model(
                    function_tools=[],
                    allow_text_result=True,
                    result_tools=[],
                )
                self._initialized_models["post"] = post_result

        return self._initialized_models

    def _get_last_content(self, messages: list[ModelMessage]) -> str:
        """Extract content from last message."""
        if not messages:
            return ""

        last_msg = messages[-1]
        if isinstance(last_msg, ModelRequest):
            for part in reversed(last_msg.parts):
                if isinstance(part, UserPromptPart):
                    return part.content
        return ""

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None = None,
    ) -> tuple[ModelResponse, Cost]:
        """Process request with optional pre/post prompting."""
        models = await self._initialize_models()
        total_cost = Cost()

        # Pre-process if configured
        if self.pre_prompt:
            last_content = self._get_last_content(messages)
            content = self.pre_prompt.text.format(input=last_content)
            part = UserPromptPart(content=content)
            pre_req = ModelRequest(parts=[part])
            pre_res, pre_cost = await models["pre"].request([pre_req], model_settings)
            total_cost += pre_cost
            # Replace last message with expanded version
            part = UserPromptPart(content=str(pre_res))
            request = ModelRequest(parts=[part])
            messages = [*messages[:-1], request]

        # Main model processing
        response, main_cost = await models["main"].request(messages, model_settings)
        total_cost += main_cost

        # Post-process if configured
        if self.post_prompt:
            content = self.post_prompt.text.format(output=str(response))
            p = UserPromptPart(content=content)
            post_req = ModelRequest(parts=[p])
            post_res, post_cost = await models["post"].request([post_req], model_settings)
            total_cost += post_cost
            return post_res, total_cost

        return response, total_cost
