"""Utility functions for model handling."""

from __future__ import annotations

import functools
import inspect
import logging
from typing import TYPE_CHECKING, Any

import anyenv
from pydantic import BaseModel
from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from llmling_models.formatting import format_part


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from pydantic_ai import ModelMessage, ModelResponsePart
    from pydantic_ai.models.function import (
        AgentInfo,
        BuiltinToolCallsReturns,
        DeltaThinkingCalls,
        DeltaToolCalls,
    )


def function_to_model(
    callback: Callable[..., Any],
    streamable: bool = True,
) -> FunctionModel:
    """Factory to get a text model for Callables with "simpler" signatures.

    This function serves as a helper to allow creating FunctionModels which take either
    no arguments or a single argument in form of a prompt.
    """
    sig = inspect.signature(callback)
    # Count required parameters (those without defaults)
    required_params = sum(
        1 for param in sig.parameters.values() if param.default is inspect.Parameter.empty
    )
    takes_prompt = required_params > 0

    @functools.wraps(callback)
    async def callback_wrapper(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> ModelResponse:
        try:
            if takes_prompt:
                prompt = format_part(messages[-1].parts[-1])
                if inspect.iscoroutinefunction(callback):
                    result = await callback(prompt)
                else:
                    result = callback(prompt)
            elif inspect.iscoroutinefunction(callback):
                result = await callback()
            else:
                result = callback()

            if isinstance(result, str):
                part: ModelResponsePart = TextPart(result)
            # For structured responses, check if agent expects structured output
            elif agent_info.allow_text_output:
                # Agent expects text - serialize the structured result

                serialized = (
                    anyenv.dump_json(result.model_dump())
                    if isinstance(result, BaseModel)
                    else str(result)
                )
                part = TextPart(serialized)
            else:
                # Agent expects structured output - return as ToolCallPart
                part = ToolCallPart(tool_name="final_result", args=result.model_dump())
            return ModelResponse(parts=[part])
        except Exception as e:
            logger.exception("Processor callback failed")
            name = getattr(callback, "__name__", str(callback))
            msg = f"Processor error in {name!r}: {e}"
            raise RuntimeError(msg) from e

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls | DeltaThinkingCalls | BuiltinToolCallsReturns]:
        result = await callback_wrapper(messages, agent_info)
        part = result.parts[0]
        match part:
            case TextPart():
                yield part.content
            case ToolCallPart():
                args_json = anyenv.dump_json(part.args) if part.args else "{}"
                yield {0: DeltaToolCall(name=part.tool_name, json_args=args_json)}
            case _:
                msg = f"Unexpected part type: {type(part)}"
                raise ValueError(msg)

    kwargs: dict[str, Any] = {"stream_function": stream_function} if streamable else {}
    return FunctionModel(function=callback_wrapper, **kwargs)


if __name__ == "__main__":
    import asyncio

    from pydantic_ai import Agent

    class Response(BaseModel):
        text: str

    def structured_response(text: str) -> Response:
        return Response(text=text)

    agent = Agent(model=function_to_model(structured_response))

    async def main() -> None:
        async for event in agent.run_stream_events(str(dict(a="test")), output_type=Response):
            print(event)

    asyncio.run(main())
