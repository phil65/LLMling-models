from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import claude_agent_sdk
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock
from claude_agent_sdk.types import StreamEvent
from pydantic_ai import ModelRequest, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
import pytest

from llmling_models.models.claude_code_model import ClaudeCodeModel


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


MESSAGES = [ModelRequest(parts=[UserPromptPart("hi")])]
USAGE = {
    "input_tokens": 10,
    "output_tokens": 5,
    "cache_read_input_tokens": 100,
    "cache_creation_input_tokens": 20,
}


def _result() -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="s",
        usage=USAGE,
    )


def _delta(text: str, index: int = 0) -> StreamEvent:
    event = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }
    return StreamEvent(uuid="u", session_id="s", event=event)


def _start() -> StreamEvent:
    return StreamEvent(uuid="u", session_id="s", event={"type": "message_start", "message": {}})


def _fake_query(messages: list[Any], produced: list[Any], delay: float = 0.0) -> Any:
    async def query(*, prompt: Any, options: Any) -> AsyncIterator[Any]:
        for message in messages:
            if delay:
                await asyncio.sleep(delay)
            produced.append(message)
            yield message

    return query


async def test_stream_cancellation_stops_the_claude_code_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    produced: list[Any] = []
    messages = [_delta(f"t{i} ") for i in range(30)]
    monkeypatch.setattr(claude_agent_sdk, "query", _fake_query(messages, produced, 0.05))
    model = ClaudeCodeModel()

    async def consume() -> None:
        async with model.request_stream(MESSAGES, None, ModelRequestParameters()) as stream:
            async for _ in stream:
                pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.2)
    task.cancel()
    loop = asyncio.get_running_loop()
    start = loop.time()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert loop.time() - start < 0.5  # noqa: PLR2004
    count = len(produced)
    await asyncio.sleep(0.2)
    assert len(produced) == count < len(messages)


async def test_stream_early_exit_stops_the_claude_code_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    produced: list[Any] = []
    messages = [_delta(f"t{i} ") for i in range(30)]
    monkeypatch.setattr(claude_agent_sdk, "query", _fake_query(messages, produced, 0.05))
    model = ClaudeCodeModel()
    async with model.request_stream(MESSAGES, None, ModelRequestParameters()) as stream:
        async for _ in stream:
            break
    count = len(produced)
    await asyncio.sleep(0.2)
    assert len(produced) == count < len(messages)


async def test_stream_without_partial_messages_streams_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = [AssistantMessage(content=[TextBlock("hello")], model="m"), _result()]
    monkeypatch.setattr(claude_agent_sdk, "query", _fake_query(messages, []))
    model = ClaudeCodeModel(include_partial_messages=False)
    async with model.request_stream(MESSAGES, None, ModelRequestParameters()) as stream:
        async for _ in stream:
            pass
    parts = stream.get().parts
    assert [p.content for p in parts if isinstance(p, TextPart)] == ["hello"]


async def test_stream_text_of_later_turns_gets_its_own_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = ToolUseBlock(id="c1", name="Read", input={})
    messages = [
        _start(),
        _delta("first"),
        AssistantMessage(content=[TextBlock("first"), tool], model="m"),
        _start(),
        _delta("second"),
        AssistantMessage(content=[TextBlock("second")], model="m"),
        _result(),
    ]
    monkeypatch.setattr(claude_agent_sdk, "query", _fake_query(messages, []))
    async with ClaudeCodeModel().request_stream(MESSAGES, None, ModelRequestParameters()) as stream:
        async for _ in stream:
            pass
    parts = stream.get().parts
    assert [type(p).__name__ for p in parts] == ["TextPart", "NativeToolCallPart", "TextPart"]
    assert [p.content for p in parts if isinstance(p, TextPart)] == ["first", "second"]


async def test_usage_input_tokens_include_cache_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = [AssistantMessage(content=[TextBlock("hello")], model="m"), _result()]
    monkeypatch.setattr(claude_agent_sdk, "query", _fake_query(messages, []))
    model = ClaudeCodeModel(include_partial_messages=False)
    response = await model.request(MESSAGES, None, ModelRequestParameters())
    assert response.usage.input_tokens == 130  # noqa: PLR2004
    assert response.usage.cache_read_tokens == 100  # noqa: PLR2004
    async with model.request_stream(MESSAGES, None, ModelRequestParameters()) as stream:
        async for _ in stream:
            pass
    assert stream.usage.input_tokens == 130  # noqa: PLR2004
