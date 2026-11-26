import json
from typing import TYPE_CHECKING, Any

from pydantic_ai import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from websockets.asyncio.server import serve

from llmling_models.models.remote_input.model import RemoteInputModel
from llmling_models.models.remote_model.model import RemoteProxyModel


if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from websockets.asyncio.server import ServerConnection


async def _server(seen: dict[str, Any]) -> AbstractAsyncContextManager[Any]:
    async def handler(ws: ServerConnection) -> None:
        seen["auth"] = ws.request.headers.get("Authorization") if ws.request else None
        await ws.recv()
        await ws.send(json.dumps({"chunk": "hel", "done": False}))
        await ws.send(json.dumps({"chunk": "lo", "done": False}))
        await ws.send(json.dumps({"done": True}))

    return serve(handler, "127.0.0.1", 0)


def _url(server: Any) -> str:
    port = next(iter(server.sockets)).getsockname()[1]
    return f"ws://127.0.0.1:{port}"


MESSAGES = [ModelRequest(parts=[UserPromptPart("hi")])]


async def test_remote_proxy_model_websocket_sends_auth_header() -> None:
    seen: dict[str, Any] = {}
    async with await _server(seen) as server:
        model = RemoteProxyModel(url=_url(server), api_key="k")
        response = await model.request(MESSAGES, None, ModelRequestParameters())
        assert response.parts[0].content == "hello"  # type: ignore[union-attr]
        async with model.request_stream(MESSAGES, None, ModelRequestParameters()) as stream:
            async for _ in stream:
                pass
        assert stream.get().parts[0].content == "hello"  # type: ignore[union-attr]
    assert seen["auth"] == "Bearer k"


async def test_remote_input_model_websocket_sends_auth_header() -> None:
    seen: dict[str, Any] = {}
    async with await _server(seen) as server:
        model = RemoteInputModel(url=_url(server), api_key="k")
        response = await model.request(MESSAGES, None, ModelRequestParameters())
        assert response.parts[0].content == "hello"  # type: ignore[union-attr]
        async with model.request_stream(MESSAGES, None, ModelRequestParameters()) as stream:
            async for _ in stream:
                pass
        assert stream.get().parts[0].content == "hello"  # type: ignore[union-attr]
    assert seen["auth"] == "Bearer k"
