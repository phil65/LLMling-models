from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelResponse,
    RunContext,
    RunUsage,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.function import FunctionToolset

from llmling_models.toolsets.codemode_toolset import CodeModeToolset


if TYPE_CHECKING:
    from pydantic_ai.toolsets import AbstractToolset, ToolsetTool


class Point(BaseModel):
    x: int
    y: int = 0


received: list[Any] = []


def norm(p: Point, scale: int = 1) -> int:
    """Scale a point."""
    received.append((p, scale))
    return (p.x + p.y) * scale


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


async def _run(code: str) -> Any:
    toolset = CodeModeToolset(toolsets=[FunctionToolset(tools=[norm])])
    ctx = _ctx()
    tools = await toolset.get_tools(ctx)
    return await toolset.call_tool(
        "execute_python", {"python_code": code}, ctx, tools["execute_python"]
    )


async def test_tool_arguments_are_validated() -> None:
    received.clear()
    result = await _run('async def main():\n    return await norm(p={"x": 1}, scale="3")')
    assert result == 3  # noqa: PLR2004
    assert isinstance(received[0][0], Point)
    assert received[0][1] == 3  # noqa: PLR2004


async def test_positional_tool_calls() -> None:
    received.clear()
    result = await _run('async def main():\n    return await norm({"x": 2, "y": 1}, 2)')
    assert result == 6  # noqa: PLR2004


async def test_invalid_arguments_are_reported() -> None:
    result = await _run('async def main():\n    return await norm(p={"x": "nope"})')
    assert isinstance(result, str)
    assert "Error executing" in result
    assert "x" in result


class CountingToolset(FunctionToolset[Any]):
    calls = 0

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        type(self).calls += 1
        return await super().get_tools(ctx)


async def test_wrapped_tools_listed_once_per_step() -> None:
    inner = CountingToolset(tools=[norm])
    toolset: AbstractToolset[Any] = CodeModeToolset(toolsets=[inner])
    code = 'async def main():\n    return await norm({"x": 1})'

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("execute_python", {"python_code": code})])
        return ModelResponse(parts=[TextPart("done")])

    CountingToolset.calls = 0
    await Agent(FunctionModel(respond), toolsets=[toolset]).run("go")
    # One listing per model step (two steps); executing the code reuses the first one.
    assert CountingToolset.calls == 2  # noqa: PLR2004
