import json
from typing import TYPE_CHECKING, Any

from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from llmling_models.openai_server.model_registry import ModelRegistry
from llmling_models.openai_server.server import OpenAIServer


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _client(model: Any) -> TestClient:
    return TestClient(OpenAIServer(ModelRegistry({"t": model})).app)


def test_chat_completion_passes_temperature_and_max_tokens() -> None:
    seen: dict[str, Any] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen["settings"] = info.model_settings
        return ModelResponse(parts=[TextPart("hello")])

    resp = _client(FunctionModel(respond)).post(
        "/v1/chat/completions",
        json={
            "model": "t",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "max_tokens": 10,
        },
    )
    assert resp.is_success, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == "hello"
    assert seen["settings"] == {"temperature": 0.2, "max_tokens": 10}


def test_websocket_chat_completion_passes_temperature() -> None:
    seen: dict[str, Any] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen["settings"] = info.model_settings
        return ModelResponse(parts=[TextPart("hello")])

    server = OpenAIServer(ModelRegistry({"t": FunctionModel(respond)}), api_key="k")
    with TestClient(server.app).websocket_connect(
        "/v1/chat/completions/ws", headers={"Authorization": "Bearer k"}
    ) as ws:
        ws.send_json({
            "model": "t",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
        })
        data = ws.receive_json()
    assert "error" not in data, data
    assert data["choices"][0]["message"]["content"] == "hello"
    assert seen["settings"] == {"temperature": 0.2}


def _sse_deltas(text: str) -> tuple[list[dict[str, Any]], str | None]:
    deltas, finish = [], None
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        choice = json.loads(line.removeprefix("data: "))["choices"][0]
        deltas.append(choice["delta"])
        finish = choice["finish_reason"] or finish
    return deltas, finish


def test_stream_keeps_later_text_parts_and_tool_calls() -> None:
    from pydantic_ai.models.function import DeltaToolCall

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[Any]:
        yield "first part"
        yield {0: DeltaToolCall(name="lookup", json_args='{"q": "x"}', tool_call_id="c1")}

    resp = _client(FunctionModel(stream_function=stream)).post(
        "/v1/chat/completions",
        json={
            "model": "t",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    },
                }
            ],
        },
    )
    deltas, finish = _sse_deltas(resp.text)
    text = "".join(d.get("content", "") for d in deltas)
    assert text == "first part"
    calls = [c for d in deltas for c in d.get("tool_calls", [])]
    assert calls[0]["function"]["name"] == "lookup"
    args = "".join(c["function"].get("arguments", "") for c in calls)
    assert json.loads(args) == {"q": "x"}
    assert finish == "tool_calls"


def test_stream_second_text_part_not_truncated() -> None:
    from pydantic_ai import PartStartEvent

    from llmling_models.openai_server.helpers import StreamDeltaConverter

    conv = StreamDeltaConverter()
    assert conv.convert(PartStartEvent(index=0, part=TextPart("hello world"))) == {
        "content": "hello world"
    }
    assert conv.convert(PartStartEvent(index=1, part=TextPart("second"))) == {"content": "second"}
    assert conv.finish_reason == "stop"
