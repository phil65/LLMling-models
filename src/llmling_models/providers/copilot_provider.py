"""GitHub Copilot provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from httpx import AsyncClient as AsyncHTTPClient
from openai import AsyncOpenAI
from pydantic_ai.providers import Provider

from llmling_models.log import get_logger


if TYPE_CHECKING:
    from httpx import Request, Response
    from tokonomics import CopilotTokenManager


logger = get_logger(__name__)


class CopilotHTTPClient(AsyncHTTPClient):
    """Custom client that adds fresh token headers before each request."""

    def __init__(self, token_manager: CopilotTokenManager, **kwargs):
        super().__init__(**kwargs)
        self.token_manager = token_manager

    async def send(self, request: Request, *args, **kwargs) -> Response:
        header = await self.token_manager.generate_headers()
        request.headers.update(header)
        return await super().send(request, *args, **kwargs)


def _create_client(token_manager: CopilotTokenManager) -> AsyncOpenAI:
    """Create OpenAI client with Copilot-specific configuration."""
    client = CopilotHTTPClient(token_manager, timeout=60.0)
    return AsyncOpenAI(
        api_key="not-used-but-required",
        base_url=token_manager._api_endpoint,
        http_client=client,
    )


class CopilotProvider(Provider[AsyncOpenAI]):
    """Provider for GitHub Copilot API.

    Uses tokonomics.CopilotTokenManager for token management.
    """

    def __init__(self) -> None:
        """Initialize the provider with tokonomics token manager."""
        from tokonomics import CopilotTokenManager

        self._token_manager = CopilotTokenManager()
        self._client = _create_client(self._token_manager)

    @property
    def name(self) -> str:
        """The provider name."""
        return "copilot"

    @property
    def base_url(self) -> str:
        """The base URL for the provider API."""
        return self._token_manager._api_endpoint

    @property
    def client(self) -> AsyncOpenAI:
        """Get a client with the current token."""
        return self._client


if __name__ == "__main__":
    import asyncio

    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIResponsesModel

    async def main():
        provider = CopilotProvider()
        model = OpenAIResponsesModel("gpt-5-nano", provider=provider)
        agent = Agent(model=model)
        result = await agent.run("Hello, world!")
        print(result)

    asyncio.run(main())
