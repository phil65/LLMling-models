from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from llmling_models.openai_server.model_registry import ModelRegistry
from llmling_models.openai_server.server import OpenAIServer


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
