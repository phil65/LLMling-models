"""GitHub Copilot provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from httpx import AsyncClient as AsyncHTTPClient
from openai import AsyncOpenAI
from pydantic_ai.providers import Provider

from llmling_models.log import get_logger


if TYPE_CHECKING:
    from tokonomics import CopilotTokenManager


# Initialize the token manager from tokonomics


logger = get_logger(__name__)


class CopilotProvider(Provider[AsyncOpenAI]):
    """Provider for GitHub Copilot API.

    Uses tokonomics.CopilotTokenManager for token management.
    """

    def __init__(self) -> None:
        """Initialize the provider with tokonomics token manager."""
        from tokonomics import CopilotTokenManager

        self._token_manager = CopilotTokenManager()
        self._client = self._create_client()

    def _create_client(self) -> AsyncOpenAI:
        """Create OpenAI client with Copilot-specific configuration."""

        # Custom client that adds fresh token headers before each request
        class CopilotHTTPClient(AsyncHTTPClient):
            def __init__(self, token_manager: CopilotTokenManager, **kwargs):
                super().__init__(**kwargs)
                self.token_manager = token_manager

            async def send(self, request, *args, **kwargs):
                # Get fresh token for each request
                token = self.token_manager.get_token()

                # Set required headers
                request.headers["Authorization"] = f"Bearer {token}"
                request.headers["editor-version"] = "Neovim/0.9.0"
                request.headers["Copilot-Integration-Id"] = "vscode-chat"

                # Send with updated headers
                return await super().send(request, *args, **kwargs)

        # Create the HTTP client with token manager
        http_client = CopilotHTTPClient(
            token_manager=self._token_manager,
            timeout=60.0,
        )

        # Create OpenAI client with our custom HTTP client
        return AsyncOpenAI(
            api_key="not-used-but-required",
            base_url=self._token_manager._api_endpoint,
            http_client=http_client,
        )

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
    from pydantic_ai.models.openai import OpenAIModel

    async def main():
        provider = CopilotProvider()
        model = OpenAIModel("gpt-4o-mini", provider=provider)
        agent = Agent(model=model)
        result = await agent.run("Hello, world!")
        print(result)

    asyncio.run(main())
