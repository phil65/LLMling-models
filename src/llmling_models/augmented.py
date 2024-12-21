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
from pydantic_ai.result import Usage

from llmling_models.base import PydanticModel
from llmling_models.log import get_logger


logger = get_logger(__name__)

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
              model: openai:gpt-4o-mini
            post_prompt:
              text: "Summarize your response."
              model: openai:gpt-4o-mini
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
        self._initialized_models: dict[str, AgentModel] = {}

    async def _get_agent_model(self, key: str) -> AgentModel:
        """Get or initialize an agent model."""
        if key not in self._initialized_models:
            if key == "main":
                model = self.main_model
            elif key == "pre" and self.pre_prompt:
                model = self.pre_prompt.model_instance
            elif key == "post" and self.post_prompt:
                model = self.post_prompt.model_instance
            else:
                msg = f"Unknown model key: {key}"
                raise ValueError(msg)

            self._initialized_models[key] = await model.agent_model(
                function_tools=self.function_tools if key == "main" else [],
                allow_text_result=True,
                result_tools=self.result_tools if key == "main" else [],
            )

        return self._initialized_models[key]

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
    ) -> tuple[ModelResponse, Usage]:
        """Process request with pre/post prompting."""
        total_cost = Usage()
        all_messages = messages.copy()

        # Pre-process the question if configured
        if self.pre_prompt:
            pre_model = await self._get_agent_model("pre")
            input_question = self._get_last_content(messages)
            pre_prompt = self.pre_prompt.text.format(input=input_question)

            # Get expanded question
            pre_request = ModelRequest(parts=[UserPromptPart(content=pre_prompt)])
            pre_response, pre_cost = await pre_model.request(
                [pre_request], model_settings
            )
            total_cost += pre_cost

            # Replace original question with expanded version
            expanded_question = str(pre_response.parts[0].content)  # type: ignore
            logger.debug("Original question: %s", input_question)
            logger.debug("Expanded question: %s", expanded_question)
            expanded_part = UserPromptPart(content=expanded_question)
            all_messages[-1] = ModelRequest(parts=[expanded_part])

        # Process with main model
        main_model = await self._get_agent_model("main")
        main_response, main_cost = await main_model.request(all_messages, model_settings)
        logger.debug("Main response: %s", str(main_response.parts[0].content))  # type: ignore

        # Post-process if configured
        if self.post_prompt:
            post_model = await self._get_agent_model("post")
            post_prompt = self.post_prompt.text.format(
                output=str(main_response.parts[0].content)  # type: ignore
            )

            # Create post-processing request
            post_request = ModelRequest(parts=[UserPromptPart(content=post_prompt)])
            post_response, post_cost = await post_model.request(
                [post_request], model_settings
            )
            total_cost += post_cost
            logger.debug(
                "Post-processed response: %s",
                str(post_response.parts[0].content),  # type: ignore
            )

            # Add post-prompt messages to the chain
            all_messages.extend([main_response, post_request, post_response])
            return post_response, total_cost

        # If no post-processing, add main response to message chain
        all_messages.append(main_response)
        return main_response, total_cost


if __name__ == "__main__":
    import asyncio

    from pydantic_ai import Agent

    async def test():
        pre = PrePostPromptConfig(
            text=(
                "Your task is to rewrite '{input}' as a more detailed "
                "philosophical question. Do not answer it. Only return the expanded "
                "question."
            ),
            model="openai:gpt-4o-mini",
        )
        augmented = AugmentedModel(
            main_model="openai:gpt-4o-mini",
            pre_prompt=pre,
        )
        agent: Agent[None, str] = Agent(model=augmented)

        print("\nTesting Pre-Prompt Expansion Pipeline")
        print("=" * 60)

        question = "What is the meaning of life?"
        print(f"Original Question: {question}")

        result = await agent.run(question)

        # Get expanded question from pre-prompt response
        expanded = result._all_messages[0].parts[0].content  # type: ignore

        print("\nPipeline Steps:")
        print("\n1. Original Question:")
        print("-" * 40)
        print(question)

        print("\n2. Expanded Question:")
        print("-" * 40)
        print(expanded)

        print("\n3. Main Model Response:")
        print("-" * 40)
        print(result.data)

    asyncio.run(test())
