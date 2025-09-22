"""LM Studio provider implementation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, overload

from openai import AsyncOpenAI
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.providers import Provider

from llmling_models.log import get_logger


if TYPE_CHECKING:
    from httpx import AsyncClient as AsyncHTTPClient


logger = get_logger(__name__)


class LMStudioProvider(Provider[AsyncOpenAI]):
    """Provider for LM Studio local API."""

    @property
    def name(self) -> str:
        """The provider name."""
        return "lm-studio"

    @property
    def base_url(self) -> str:
        """The base URL for the provider API."""
        return os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")

    @property
    def client(self) -> AsyncOpenAI:
        """Get a client configured for LM Studio."""
        return self._client

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, base_url: str) -> None: ...

    @overload
    def __init__(self, *, base_url: str, http_client: AsyncHTTPClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        base_url: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        """Initialize provider for LM Studio.

        Args:
            base_url: The base URL for the API, defaults to http://localhost:1234/v1
            openai_client: An existing AsyncOpenAI client to use.
                           If provided, other parameters must be None.
            http_client: An existing AsyncHTTPClient to use for making HTTP requests.
        """
        self._base_url = base_url or self.base_url

        if openai_client is not None:
            assert http_client is None, (
                "Cannot provide both `openai_client` and `http_client`"
            )
            assert base_url is None, "Cannot provide both `openai_client` and `base_url`"
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(
                base_url=self._base_url,
                api_key="lm-studio",  # Not actually used
                http_client=http_client,
            )
        else:
            self._client = AsyncOpenAI(
                base_url=self._base_url,
                api_key="lm-studio",  # Not actually used
                http_client=cached_async_http_client(),
            )


if __name__ == "__main__":
    import asyncio

    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIResponsesModel

    async def main():
        provider = LMStudioProvider(base_url="http://100.69.216.7:11434/v1")
        model = OpenAIResponsesModel(
            "lmstudio-community/Qwen2.5-7B-Instruct-1M-GGUF/Qwen2.5-7B-Instruct-1M-Q4_K_M",
            provider=provider,
        )
        agent = Agent(model=model)
        result = await agent.run("Hello, world!")
        print(result)

    asyncio.run(main())
