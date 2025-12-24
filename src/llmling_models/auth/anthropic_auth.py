"""Anthropic Claude Max/Pro OAuth authentication.

This module implements OAuth 2.0 PKCE authentication for Claude Max/Pro subscriptions,
allowing users to use their subscription directly through the Anthropic API.

Based on the opencode-anthropic-auth plugin by SST.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import secrets
import sys
import time
from typing import TYPE_CHECKING
import webbrowser

import httpx

from llmling_models.log import get_logger


if TYPE_CHECKING:
    from typing import Self

logger = get_logger(__name__)

# OAuth client ID registered with Anthropic (from opencode-anthropic-auth)
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

# OAuth endpoints
OAUTH_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
OAUTH_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"

# Required scopes for API access
OAUTH_SCOPES = "org:create_api_key user:profile user:inference"

# Beta headers required for OAuth authentication
OAUTH_BETA_HEADERS = [
    "oauth-2025-04-20",
    "claude-code-20250219",
    "interleaved-thinking-2025-05-14",
    "fine-grained-tool-streaming-2025-05-14",
]

# Default token storage location
DEFAULT_TOKEN_PATH = Path.home() / ".config" / "llmling-models" / "anthropic_oauth.json"


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge.

    Returns:
        Tuple of (verifier, challenge)
    """
    # Generate random verifier (43-128 characters)
    verifier = secrets.token_urlsafe(32)

    # Create SHA256 hash and base64url encode it
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

    return verifier, challenge


@dataclass
class AnthropicOAuthToken:
    """Stored OAuth token data."""

    access_token: str
    refresh_token: str
    expires_at: float  # Unix timestamp

    def is_expired(self, buffer_seconds: int = 60) -> bool:
        """Check if the token is expired or about to expire."""
        return time.time() >= (self.expires_at - buffer_seconds)

    def to_dict(self) -> dict[str, str | float]:
        """Convert to dictionary for JSON serialization."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | float]) -> Self:
        """Create from dictionary."""
        return cls(
            access_token=str(data["access_token"]),
            refresh_token=str(data["refresh_token"]),
            expires_at=float(data["expires_at"]),
        )


@dataclass
class AnthropicTokenStore:
    """File-based token storage for Anthropic OAuth."""

    path: Path = field(default_factory=lambda: DEFAULT_TOKEN_PATH)
    _token: AnthropicOAuthToken | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Ensure storage directory exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> AnthropicOAuthToken | None:
        """Load token from file."""
        if self._token is not None:
            return self._token

        if not self.path.exists():
            return None

        try:
            data = json.loads(self.path.read_text())
            self._token = AnthropicOAuthToken.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to load token from %s: %s", self.path, e)
            return None
        else:
            return self._token

    def save(self, token: AnthropicOAuthToken) -> None:
        """Save token to file."""
        self._token = token
        self.path.write_text(json.dumps(token.to_dict(), indent=2))
        # Set restrictive permissions (owner read/write only)
        self.path.chmod(0o600)
        logger.debug("Saved token to %s", self.path)

    def clear(self) -> None:
        """Remove stored token."""
        self._token = None
        if self.path.exists():
            self.path.unlink()
            logger.debug("Removed token from %s", self.path)

    def get_valid_token(self) -> AnthropicOAuthToken | None:
        """Get token if it exists and is not expired."""
        token = self.load()
        if token is None:
            return None
        if token.is_expired():
            logger.debug("Token is expired, needs refresh")
            return None
        return token


def build_authorization_url(verifier: str, challenge: str) -> str:
    """Build the OAuth authorization URL.

    Args:
        verifier: PKCE code verifier (used as state)
        challenge: PKCE code challenge

    Returns:
        The authorization URL to open in browser
    """
    params = {
        "code": "true",
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": OAUTH_REDIRECT_URI,
        "scope": OAUTH_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{OAUTH_AUTHORIZE_URL}?{query}"


def exchange_code_for_token(code: str, verifier: str) -> AnthropicOAuthToken:
    """Exchange authorization code for access token.

    Args:
        code: The authorization code from callback (may include state after #)
        verifier: The PKCE code verifier

    Returns:
        The OAuth token

    Raises:
        RuntimeError: If token exchange fails
    """
    # Code may be in format "code#state"
    auth_code = code.split("#")[0]

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            OAUTH_TOKEN_URL,
            json={
                "code": auth_code,
                "state": code.split("#")[1] if "#" in code else "",
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "redirect_uri": OAUTH_REDIRECT_URI,
                "code_verifier": verifier,
            },
        )

        if not response.is_success:
            msg = f"Token exchange failed: {response.status_code} - {response.text}"
            raise RuntimeError(msg)

        data = response.json()
        expires_at = time.time() + data["expires_in"]

        return AnthropicOAuthToken(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=expires_at,
        )


def refresh_access_token(refresh_token: str) -> AnthropicOAuthToken:
    """Refresh an expired access token.

    Args:
        refresh_token: The refresh token

    Returns:
        New OAuth token

    Raises:
        RuntimeError: If refresh fails
    """
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            OAUTH_TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLIENT_ID,
            },
        )

        if not response.is_success:
            msg = f"Token refresh failed: {response.status_code} - {response.text}"
            raise RuntimeError(msg)

        data = response.json()
        expires_at = time.time() + data["expires_in"]

        return AnthropicOAuthToken(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=expires_at,
        )


def authenticate_anthropic_max(
    verbose: bool = True,
    open_browser: bool = True,
) -> AnthropicOAuthToken:
    """Authenticate with Anthropic using OAuth for Claude Max/Pro.

    This initiates the OAuth PKCE flow:
    1. Opens browser to Anthropic authorization page
    2. User authenticates and authorizes the application
    3. User copies the authorization code and pastes it back
    4. Code is exchanged for access/refresh tokens

    Args:
        verbose: Whether to print status messages
        open_browser: Whether to automatically open the browser

    Returns:
        The OAuth token

    Raises:
        RuntimeError: If authentication fails
    """
    # Generate PKCE challenge
    verifier, challenge = generate_pkce()

    # Build authorization URL
    auth_url = build_authorization_url(verifier, challenge)

    if verbose:
        print("\nTo authenticate with Claude Max/Pro:")
        print(f"\n1. Visit: {auth_url}")
        print("\n2. Sign in with your Anthropic account")
        print("3. Copy the authorization code shown")
        print()

    if open_browser:
        if verbose:
            print("Opening browser...")
        webbrowser.open(auth_url)

    # Get code from user
    try:
        code = input("Paste the authorization code here: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nAuthentication cancelled.")
        msg = "Authentication cancelled by user"
        raise RuntimeError(msg) from None

    if not code:
        msg = "No authorization code provided"
        raise RuntimeError(msg)

    # Exchange code for token
    if verbose:
        print("\nExchanging code for token...")

    token = exchange_code_for_token(code, verifier)

    if verbose:
        print("Authentication successful!")

    return token


def get_or_refresh_token(store: AnthropicTokenStore | None = None) -> AnthropicOAuthToken:
    """Get a valid token, refreshing if necessary.

    Args:
        store: Token store to use (defaults to standard location)

    Returns:
        Valid OAuth token

    Raises:
        RuntimeError: If no token exists and authentication is needed
    """
    if store is None:
        store = AnthropicTokenStore()

    token = store.load()
    if token is None:
        msg = "No Anthropic OAuth token found. Run 'llmling-models anthropic-auth' to authenticate."
        raise RuntimeError(msg)

    if token.is_expired():
        logger.info("Token expired, refreshing...")
        token = refresh_access_token(token.refresh_token)
        store.save(token)

    return token


async def get_or_refresh_token_async(
    store: AnthropicTokenStore | None = None,
) -> AnthropicOAuthToken:
    """Async version of get_or_refresh_token.

    Args:
        store: Token store to use (defaults to standard location)

    Returns:
        Valid OAuth token

    Raises:
        RuntimeError: If no token exists and authentication is needed
    """
    if store is None:
        store = AnthropicTokenStore()

    token = store.load()
    if token is None:
        msg = "No Anthropic OAuth token found. Run 'llmling-models anthropic-auth' to authenticate."
        raise RuntimeError(msg)

    if token.is_expired():
        logger.info("Token expired, refreshing...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                OAUTH_TOKEN_URL,
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": token.refresh_token,
                    "client_id": CLIENT_ID,
                },
            )

            if not response.is_success:
                msg = f"Token refresh failed: {response.status_code} - {response.text}"
                raise RuntimeError(msg)

            data = response.json()
            expires_at = time.time() + data["expires_in"]

            token = AnthropicOAuthToken(
                access_token=data["access_token"],
                refresh_token=data["refresh_token"],
                expires_at=expires_at,
            )
            store.save(token)

    return token


def anthropic_auth_main() -> None:
    """Command-line entry point for Anthropic OAuth authentication."""
    parser = argparse.ArgumentParser(
        description="Authenticate with Anthropic Claude Max/Pro using OAuth."
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open the browser",
    )
    parser.add_argument(
        "--token-path",
        type=Path,
        default=DEFAULT_TOKEN_PATH,
        help=f"Path to store token (default: {DEFAULT_TOKEN_PATH})",
    )
    parser.add_argument(
        "--logout",
        action="store_true",
        help="Remove stored token and log out",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current authentication status",
    )

    args = parser.parse_args()
    store = AnthropicTokenStore(path=args.token_path)

    if args.logout:
        store.clear()
        print("Logged out. Token removed.")
        return

    if args.status:
        token = store.load()
        if token is None:
            print("Not authenticated.")
            print(f"Token path: {args.token_path}")
            sys.exit(1)
        elif token.is_expired():
            print("Token expired. Run without --status to refresh.")
            sys.exit(1)
        else:
            remaining = token.expires_at - time.time()
            hours = int(remaining // 3600)
            minutes = int((remaining % 3600) // 60)
            print(f"Authenticated. Token expires in {hours}h {minutes}m.")
            print(f"Token path: {args.token_path}")
        return

    try:
        token = authenticate_anthropic_max(
            verbose=True,
            open_browser=not args.no_browser,
        )
        store.save(token)
        print(f"\nToken saved to: {args.token_path}")
        print("You can now use Claude Max/Pro models with auth_method='oauth'")
    except Exception as e:
        logger.exception("Authentication failed")
        print(f"\nAuthentication failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    anthropic_auth_main()
