import threading
from typing import Any, Literal

from pydantic import BaseModel

from llmling_models.tool_bridge import ToolBridge, _get_function_schema


class Point(BaseModel):
    x: int


def f(a: int | None, b: Literal[1, 2], c: Point, d: list[int], e: str = "x") -> str:
    """Do things."""
    return f"{a} {b} {c!r} {d} {e}"


def test_schema_for_optional_literal_model_and_list_params() -> None:
    schema = _get_function_schema(f)
    props = schema["properties"]
    assert props["a"]["anyOf"] == [{"type": "integer"}, {"type": "null"}]
    assert props["b"]["enum"] == [1, 2]
    c_schema = props["c"]
    if "$ref" in c_schema:
        c_schema = schema["$defs"][c_schema["$ref"].rsplit("/", 1)[-1]]
    assert c_schema["type"] == "object"
    assert props["d"] == {"items": {"type": "integer"}, "title": "D", "type": "array"}
    assert schema["required"] == ["a", "b", "c", "d"]


async def test_arguments_are_converted_and_sync_tools_run_in_a_thread() -> None:
    seen: dict[str, Any] = {}

    def g(c: Point, n: int) -> str:
        seen["c"] = c
        seen["n"] = n
        seen["thread"] = threading.current_thread()
        return "ok"

    bridge = ToolBridge(tools=[g])
    bridge._tool_funcs["g"] = g
    assert await bridge._invoke_tool("g", {"c": {"x": 1}, "n": "2"}) == "ok"
    assert seen["c"] == Point(x=1)
    assert seen["n"] == 2  # noqa: PLR2004
    assert seen["thread"] is not threading.main_thread()
