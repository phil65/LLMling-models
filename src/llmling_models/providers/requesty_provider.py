"""Requesty provider implementation."""

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


class RequestyProvider(Provider[AsyncOpenAI]):
    """Provider for Requesty API."""

    @property
    def name(self) -> str:
        """The provider name."""
        return "requesty"

    @property
    def base_url(self) -> str:
        """The base URL for the provider API."""
        return os.environ.get("REQUESTY_BASE_URL", "https://router.requesty.ai/v1")

    @property
    def client(self) -> AsyncOpenAI:
        """Get a client configured for Requesty."""
        return self._client

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: AsyncHTTPClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        """Initialize provider for Requesty.

        Args:
            api_key: The API key to use for authentication.
                     If not provided, the `REQUESTY_API_KEY`
                     environment variable will be used if available.
            openai_client: An existing AsyncOpenAI client to use. If provided, other parameters must be None.
            http_client: An existing AsyncHTTPClient to use for making HTTP requests.
        """  # noqa: E501
        api_key = api_key or os.environ.get("REQUESTY_API_KEY")

        if api_key is None and openai_client is None:
            msg = (
                "Set the `REQUESTY_API_KEY` environment variable or pass it via "
                "`RequestyProvider(api_key=...)` to use the Requesty provider."
            )
            raise ValueError(msg)

        if openai_client is not None:
            assert http_client is None, (
                "Cannot provide both `openai_client` and `http_client`"
            )
            assert api_key is None, "Cannot provide both `openai_client` and `api_key`"
            self._client = openai_client
        else:
            if http_client is None:
                # Create HTTP client with Requesty headers
                http_client = cached_async_http_client()

                # Add Requesty-specific headers
                http_client.headers["HTTP-Referer"] = "https://llmling.dev"
                http_client.headers["X-Title"] = "LLMling"

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=api_key,
                http_client=http_client,
            )


if __name__ == "__main__":
    import asyncio

    from pydantic_ai import Agent, BinaryContent
    from pydantic_ai.models.openai import OpenAIResponsesModel

    async def main():
        provider = RequestyProvider()
        model = OpenAIResponsesModel("moondream-2B", provider=provider)
        agent = Agent(model=model)
        result = await agent.run([
            "Hello, world!",
            BinaryContent(b"https://example.com", media_type="image/png"),
        ])
        print(result)

    asyncio.run(main())
