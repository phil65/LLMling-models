"""Anthropic Claude Max/Pro OAuth provider implementation.

This provider uses OAuth authentication instead of API keys, allowing Claude Max/Pro
subscribers to use their subscription through the Anthropic API.

IMPORTANT: When using OAuth tokens, the system prompt MUST include the text
"You are Claude Code" to pass Anthropic's validation. This is enforced by
the AnthropicMaxHTTPClient which injects this into the request body.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anthropic import AsyncAnthropic
from httpx import AsyncClient as AsyncHTTPClient
from pydantic_ai.providers import Provider

from llmling_models.auth.anthropic_auth import (
    OAUTH_BETA_HEADERS,
    AnthropicTokenStore,
    get_or_refresh_token_async,
)
from llmling_models.log import get_logger


if TYPE_CHECKING:
    from httpx import Request, Response

    from llmling_models.auth.anthropic_auth import AnthropicOAuthToken


logger = get_logger(__name__)

# Required system prompt prefix for OAuth validation
# Anthropic checks for this to validate the token is being used by "Claude Code"
CLAUDE_CODE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."


class AnthropicMaxHTTPClient(AsyncHTTPClient):
    """Custom HTTP client that injects OAuth Bearer token and beta headers.

    This client:
    - Adds Authorization: Bearer <access_token> header
    - Adds required anthropic-beta headers for OAuth
    - Injects "You are Claude Code" system prompt (required for OAuth validation)
    - Automatically refreshes expired tokens
    """

    def __init__(
        self,
        token_store: AnthropicTokenStore,
        **kwargs: Any,
    ) -> None:
        """Initialize the client.

        Args:
            token_store: Token store for retrieving/refreshing tokens
            **kwargs: Additional arguments passed to AsyncClient
        """
        super().__init__(**kwargs)
        self.token_store = token_store
        self._cached_token: AnthropicOAuthToken | None = None

    async def _get_token(self) -> AnthropicOAuthToken:
        """Get a valid token, using cache when possible."""
        # Check if cached token is still valid
        if self._cached_token is not None and not self._cached_token.is_expired():
            return self._cached_token

        # Get or refresh token
        self._cached_token = await get_or_refresh_token_async(self.token_store)
        return self._cached_token

    def _inject_claude_code_system(self, body: bytes) -> bytes:
        """Inject 'Claude Code' system prompt if not present.

        Anthropic's OAuth validation requires the system prompt to contain
        "You are Claude Code" as a SEPARATE text block (not concatenated).

        Args:
            body: Original request body

        Returns:
            Modified request body with Claude Code system prompt
        """
        import json

        try:
            data = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return body

        # Only modify messages API requests
        if "messages" not in data:
            return body

        system = data.get("system", "")

        # Check if Claude Code is already mentioned
        if "Claude Code" in str(system):
            return body

        # Inject the required system prompt as a SEPARATE text block
        # This is critical - concatenating strings doesn't work!
        claude_code_block = {"type": "text", "text": CLAUDE_CODE_SYSTEM_PREFIX}

        if isinstance(system, str):
            if system:
                # Convert string to array with Claude Code as first block
                data["system"] = [claude_code_block, {"type": "text", "text": system}]
            else:
                data["system"] = CLAUDE_CODE_SYSTEM_PREFIX
        elif isinstance(system, list):
            # Prepend Claude Code block to existing list
            data["system"] = [claude_code_block, *system]
        else:
            data["system"] = CLAUDE_CODE_SYSTEM_PREFIX

        logger.debug("Injected Claude Code system prompt for OAuth validation")
        return json.dumps(data).encode()

    async def send(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        """Send request with OAuth headers and system prompt injected.

        Args:
            request: The HTTP request to send
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            The HTTP response
        """
        import httpx

        token = await self._get_token()

        # Set Authorization header (Bearer token, not API key)
        request.headers["authorization"] = f"Bearer {token.access_token}"

        # Remove x-api-key if present (SDK might add it)
        if "x-api-key" in request.headers:
            del request.headers["x-api-key"]

        # Merge beta headers with any existing ones
        existing_beta = request.headers.get("anthropic-beta", "")
        existing_list = [b.strip() for b in existing_beta.split(",") if b.strip()]

        # Combine and deduplicate
        all_betas = list(dict.fromkeys(OAUTH_BETA_HEADERS + existing_list))
        request.headers["anthropic-beta"] = ",".join(all_betas)

        # Inject Claude Code system prompt into request body
        # This is required because Anthropic validates OAuth tokens by checking
        # if the system prompt contains "You are Claude Code" - yes, really! 🤣
        # See: anthropic_spoof.txt in OpenCode
        if request.content:
            modified_body = self._inject_claude_code_system(request.content)
            # Rebuild request with modified body and updated headers
            new_request = httpx.Request(
                method=request.method,
                url=request.url,
                headers=dict(request.headers),
                content=modified_body,
            )
            new_request.headers["content-length"] = str(len(modified_body))
            logger.debug("Sending request with OAuth authentication and Claude Code spoof")
            return await super().send(new_request, *args, **kwargs)

        logger.debug("Sending request with OAuth authentication")
        return await super().send(request, *args, **kwargs)


def _create_client(token_store: AnthropicTokenStore) -> AsyncAnthropic:
    """Create Anthropic client with OAuth-enabled HTTP client.

    Args:
        token_store: Token store for authentication

    Returns:
        Configured AsyncAnthropic client
    """
    http_client = AnthropicMaxHTTPClient(token_store, timeout=600.0)
    return AsyncAnthropic(
        api_key="oauth-placeholder",  # Required by SDK but not used
        http_client=http_client,
    )


class AnthropicMaxProvider(Provider[AsyncAnthropic]):
    """Provider for Anthropic API using Claude Max/Pro OAuth authentication.

    This provider allows Claude Max/Pro subscribers to use their subscription
    through the Anthropic API instead of requiring a separate API key.

    Usage:
        1. Run `llmling-models anthropic-auth` to authenticate
        2. Use this provider with Anthropic models

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.models.anthropic import AnthropicModel
        from llmling_models.providers import AnthropicMaxProvider

        provider = AnthropicMaxProvider()
        model = AnthropicModel("claude-sonnet-4-20250514", provider=provider)
        agent = Agent(model=model)
        result = await agent.run("Hello!")
        ```
    """

    def __init__(self, token_store: AnthropicTokenStore | None = None) -> None:
        """Initialize the provider.

        Args:
            token_store: Custom token store (defaults to standard location)
        """
        self._token_store = token_store or AnthropicTokenStore()
        self._client: AsyncAnthropic | None = None  # type: ignore[assignment]

    @property
    def name(self) -> str:
        """The provider name."""
        return "anthropic-max"

    @property
    def base_url(self) -> str:
        """The base URL for the Anthropic API."""
        return "https://api.anthropic.com"

    @property
    def client(self) -> AsyncAnthropic:
        """Get the Anthropic client with OAuth authentication."""
        if self._client is None:
            self._client = _create_client(self._token_store)
        return self._client


if __name__ == "__main__":
    import asyncio
    from typing import Any, cast

    from pydantic_ai import Agent
    from pydantic_ai.models.anthropic import AnthropicModel

    async def main() -> None:
        provider = AnthropicMaxProvider()
        # Cast needed due to complex union type in AnthropicModel
        model = AnthropicModel(
            "claude-sonnet-4-20250514",
            provider=cast(Any, provider),
        )
        agent: Agent[None, str] = Agent(model=model)
        result = await agent.run("Hello! Can you confirm you're working via OAuth?")
        print(result.output)

    asyncio.run(main())
