"""Anthropic Claude Max/Pro OAuth provider implementation.

This provider uses OAuth authentication instead of API keys, allowing Claude Max/Pro
subscribers to use their subscription through the Anthropic API.

IMPORTANT: When using OAuth tokens, the system prompt MUST include the text
"You are Claude Code" to pass Anthropic's validation. This is enforced by
the AnthropicMaxHTTPClient which injects this into the request body.

Requires the `anthropic-max` extra: pip install llmling-models[anthropic-max]
"""

from __future__ import annotations

import hashlib
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

# Version string to match Claude Code CLI
CC_VERSION = "2.1.87"

# CCH (checksum hash) constants
_CCH_SEED_B64 = b"blJzasgGgx4="
CCH_MASK = 0xFFFFF
CCH_PLACEHOLDER = "cch=00000"

# Fingerprint salt for billing header
FINGERPRINT_SALT = "59cf53e54c78"


def _compute_fingerprint(first_user_message: str) -> str:
    """Compute fingerprint from first user message for billing header.

    Takes chars at indices 4, 7, 20 from the message, combines with
    a salt and version string, then SHA-256 hashes it.

    Args:
        first_user_message: The first user message content

    Returns:
        3-char hex fingerprint string
    """
    indices = [4, 7, 20]
    chars = "".join(first_user_message[i] if i < len(first_user_message) else "0" for i in indices)
    input_str = f"{FINGERPRINT_SALT}{chars}{CC_VERSION}"
    return hashlib.sha256(input_str.encode()).hexdigest()[:3]


def _compute_cch(body: bytes) -> str:
    """Compute CCH checksum over request body using xxhash64.

    Args:
        body: Request body bytes

    Returns:
        5-char hex checksum string

    Raises:
        ImportError: If xxhash package is not installed
    """
    try:
        import xxhash
    except ImportError:
        msg = (
            "xxhash package required for Anthropic Max OAuth. "
            "Install with: pip install llmling-models[anthropic-max]"
        )
        raise ImportError(msg) from None

    seed = int.from_bytes(base64.b64decode(_CCH_SEED_B64), "big")
    hash_value = xxhash.xxh64(body, seed=seed).intdigest()
    return f"{hash_value & CCH_MASK:05x}"


class AnthropicMaxHTTPClient(AsyncHTTPClient):
    """Custom HTTP client that injects OAuth Bearer token and beta headers.

    This client:
    - Adds Authorization: Bearer <access_token> header
    - Adds required anthropic-beta headers for OAuth
    - Injects "You are Claude Code" system prompt (required for OAuth validation)
    - Adds ?beta=true query parameter to match Claude Code
    - Sets user-agent to identify as Claude Code CLI
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
        """Inject Claude Code system prompt and billing header.

        Anthropic's OAuth validation requires:
        1. System prompt containing "You are Claude Code" as a SEPARATE text block
        2. A billing header block with version, fingerprint, and CCH placeholder

        The CCH placeholder is later replaced with the actual checksum.

        Args:
            body: Original request body

        Returns:
            Modified request body with Claude Code system prompt and billing header
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
        needs_claude_code = "Claude Code" not in str(system)

        # Build system blocks list
        blocks: list[dict[str, Any]] = []

        # 1. Billing header block (with CCH placeholder to be replaced later)
        first_user_msg = self._extract_first_user_text(data.get("messages", []))
        fingerprint = _compute_fingerprint(first_user_msg)
        billing_text = (
            f"x-anthropic-billing-header: cc_version={CC_VERSION}.{fingerprint}; "
            f"cc_entrypoint=cli; {CCH_PLACEHOLDER};"
        )
        blocks.append({"type": "text", "text": billing_text})

        # 2. Claude Code system prompt
        if needs_claude_code:
            blocks.append({"type": "text", "text": CLAUDE_CODE_SYSTEM_PREFIX})

        # 3. Existing system content
        if isinstance(system, str) and system:
            blocks.append({"type": "text", "text": system})
        elif isinstance(system, list):
            blocks.extend(system)

        data["system"] = blocks

        logger.debug("Injected Claude Code system prompt and billing header")
        return json.dumps(data).encode()

    @staticmethod
    def _extract_first_user_text(messages: list[Any]) -> str:
        """Extract text from the first user message."""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")
        return ""

    @staticmethod
    def _apply_cch(body: bytes) -> bytes:
        """Compute CCH checksum and replace placeholder in body.

        Args:
            body: Request body containing CCH_PLACEHOLDER

        Returns:
            Body with placeholder replaced by actual checksum
        """
        body_str = body.decode()
        if CCH_PLACEHOLDER not in body_str:
            return body
        cch = _compute_cch(body)
        return body_str.replace(CCH_PLACEHOLDER, f"cch={cch}").encode()

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

        # Set user-agent to identify as Claude Code CLI (required for OAuth validation)
        # Anthropic checks this to ensure the token is being used by Claude Code
        request.headers["user-agent"] = f"claude-cli/{CC_VERSION} (external, cli)"
        request.headers["x-app"] = "cli"

        # Add ?beta=true query parameter to match Claude Code endpoint
        # This is critical - without it, Anthropic rejects OAuth tokens
        url = str(request.url)
        if "?" not in url:
            url = f"{url}?beta=true"
        elif "beta=true" not in url:
            url = f"{url}&beta=true"

        # Merge beta headers with any existing ones
        existing_beta = request.headers.get("anthropic-beta", "")
        existing_list = [b.strip() for b in existing_beta.split(",") if b.strip()]

        # Combine and deduplicate
        all_betas = list(dict.fromkeys(OAUTH_BETA_HEADERS + existing_list))
        request.headers["anthropic-beta"] = ",".join(all_betas)

        # Inject Claude Code system prompt and billing header into request body,
        # then compute CCH checksum and replace placeholder.
        if request.content:
            modified_body = self._inject_claude_code_system(request.content)
            # Compute CCH over the body (with placeholder) and replace it
            modified_body = self._apply_cch(modified_body)
            # Rebuild request with modified URL, body and updated headers
            new_request = httpx.Request(
                method=request.method,
                url=url,
                headers=dict(request.headers),
                content=modified_body,
            )
            new_request.headers["content-length"] = str(len(modified_body))
            logger.debug(
                "Sending request with OAuth authentication and Claude Code spoof to %s",
                url,
            )
            return await super().send(new_request, *args, **kwargs)

        # Rebuild request with modified URL even if no body
        new_request = httpx.Request(
            method=request.method,
            url=url,
            headers=dict(request.headers),
        )
        logger.debug("Sending request with OAuth authentication to %s", url)
        return await super().send(new_request, *args, **kwargs)


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
