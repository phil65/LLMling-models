from __future__ import annotations

from typing import Any

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from llmling_models.models.augmented import AugmentedModel, PrePostPromptConfig


def test_pre_prompt_keeps_system_prompt_instructions_and_tool_returns() -> None:
    main_calls: list[list[Any]] = []
    pre_calls: list[AgentInfo] = []

    def main(messages: list[Any], info: AgentInfo) -> ModelResponse:
        main_calls.append(messages)
        if len(main_calls) == 1:
            return ModelResponse(parts=[ToolCallPart("lookup", {}, tool_call_id="c1")])
        return ModelResponse(parts=[TextPart("arr")])

    def pre(messages: list[Any], info: AgentInfo) -> ModelResponse:
        pre_calls.append(info)
        return ModelResponse(parts=[TextPart("expanded question")])

    pre_prompt = PrePostPromptConfig(text="Expand: {input}", model=FunctionModel(pre))
    agent = Agent(
        AugmentedModel(FunctionModel(main), pre_prompt=pre_prompt),
        system_prompt="You are a pirate",
        instructions="Be brief",
    )

    @agent.tool_plain
    def lookup() -> str:
        return "found"

    result = agent.run_sync("question")
    assert result.output == "arr"
    first = main_calls[0][-1]
    assert isinstance(first, ModelRequest)
    kinds = [p.part_kind for p in first.parts]
    assert kinds == ["system-prompt", "user-prompt"]
    assert first.parts[-1].content == "expanded question"
    assert first.instructions == "Be brief"
    # The step after the tool call keeps its tool return and is not "expanded" again.
    second = main_calls[1][-1]
    assert [p.part_kind for p in second.parts] == ["tool-return"]
    assert len(pre_calls) == 1
    assert pre_calls[0].function_tools == []


def test_post_prompt_skipped_for_tool_calls() -> None:
    post_calls: list[str] = []

    def main(messages: list[Any], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("lookup", {}, tool_call_id="c1")])
        return ModelResponse(parts=[TextPart("answer")])

    def post(messages: list[Any], info: AgentInfo) -> ModelResponse:
        post_calls.append(messages[-1].parts[0].content)
        return ModelResponse(parts=[TextPart("polished")])

    post_prompt = PrePostPromptConfig(text="Polish: {output}", model=FunctionModel(post))
    agent = Agent(AugmentedModel(FunctionModel(main), post_prompt=post_prompt))

    @agent.tool_plain
    def lookup() -> str:
        return "found"

    assert agent.run_sync("q").output == "polished"
    assert post_calls == ["Polish: answer"]
