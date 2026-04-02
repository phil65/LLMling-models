"""Anthropic Claude Max/Pro OAuth authentication.

This module implements OAuth 2.0 PKCE authentication for Claude Max/Pro subscriptions,
allowing users to use their subscription directly through the Anthropic API.

Based on the pi-mono implementation by badlogic.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, field
import hashlib
import http.server
import json
from pathlib import Path
import secrets
import socketserver
import sys
import threading
import time
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse
import webbrowser

import anyenv
import httpx

from llmling_models.auth.base import ProviderAuthBackend
from llmling_models.auth.models import (
    OAuthAuthInfo,
    ProviderAuthAuthorization,
    ProviderAuthMethod,
)
from llmling_models.log import get_logger


if TYPE_CHECKING:
    from typing import Self

    from llmling_models.auth.models import AuthInfo

logger = get_logger(__name__)

# OAuth client ID registered with Anthropic
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

# OAuth endpoints
OAUTH_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
OAUTH_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
OAUTH_MANUAL_REDIRECT_URI = "https://platform.claude.com/oauth/code/callback"

# Required scopes for API access
OAUTH_SCOPES = (
    "org:create_api_key user:profile user:inference "
    "user:sessions:claude_code user:mcp_servers user:file_upload"
)

# Beta headers required for OAuth authentication
OAUTH_BETA_HEADERS = [
    "claude-code-20250219",
    "oauth-2025-04-20",
    "interleaved-thinking-2025-05-14",
]

# Default token storage location
DEFAULT_TOKEN_PATH = Path.home() / ".config" / "llmling-models" / "anthropic_oauth.json"


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge.

    Returns:
        Tuple of (verifier, challenge)
    """
    verifier = secrets.token_urlsafe(32)
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
            data = anyenv.load_json(self.path.read_text(), return_type=dict)
            self._token = AnthropicOAuthToken.from_dict(data)
        except (anyenv.JsonLoadError, KeyError, TypeError) as e:
            logger.warning("Failed to load token from %s: %s", self.path, e)
            return None
        else:
            return self._token

    def save(self, token: AnthropicOAuthToken) -> None:
        """Save token to file."""
        self._token = token
        self.path.write_text(json.dumps(token.to_dict(), indent=2))
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


class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    code: str | None = None
    state: str | None = None
    error: str | None = None

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Suppress default logging."""

    def do_GET(self) -> None:
        """Handle GET request for OAuth callback."""
        parsed = urlparse(self.path)

        if parsed.path != "/callback":
            self.send_response(404)
            self.end_headers()
            return

        params = parse_qs(parsed.query)

        if "error" in params:
            _OAuthCallbackHandler.error = params["error"][0]
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                f"<h1>Authentication Failed</h1><p>Error: {params['error'][0]}</p>"
                "<p>You can close this window.</p>".encode()
            )
            return

        if "code" in params and "state" in params:
            _OAuthCallbackHandler.code = params["code"][0]
            _OAuthCallbackHandler.state = params["state"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<h1>Authentication Successful</h1>"
                b"<p>You can close this window and return to the terminal.</p>"
            )
        else:
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<h1>Authentication Failed</h1><p>Missing code or state parameter.</p>"
            )


def _start_callback_server() -> tuple[socketserver.TCPServer, threading.Thread, int]:
    """Start local HTTP server for OAuth callback on a dynamic port.

    Returns:
        Tuple of (server, thread, port)
    """
    _OAuthCallbackHandler.code = None
    _OAuthCallbackHandler.state = None
    _OAuthCallbackHandler.error = None

    server = socketserver.TCPServer(("127.0.0.1", 0), _OAuthCallbackHandler)
    port = server.server_address[1]
    server.timeout = 300

    thread = threading.Thread(target=server.handle_request)
    thread.daemon = True
    thread.start()

    return server, thread, port


def build_authorization_url(verifier: str, challenge: str, redirect_uri: str) -> str:
    """Build the OAuth authorization URL."""
    params = {
        "code": "true",
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": OAUTH_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{OAUTH_AUTHORIZE_URL}?{query}"


def exchange_code_for_token(
    code: str, state: str, verifier: str, redirect_uri: str
) -> AnthropicOAuthToken:
    """Exchange authorization code for access token.

    Args:
        code: The authorization code from callback
        state: The OAuth state from callback
        verifier: The PKCE code verifier
        redirect_uri: The redirect URI used in the authorization request

    Returns:
        The OAuth token
    """
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            OAUTH_TOKEN_URL,
            json={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
            },
        )

        if not response.is_success:
            msg = f"Token exchange failed: {response.status_code} - {response.text}"
            raise RuntimeError(msg)

        data = response.json()
        expires_at = time.time() + data["expires_in"] - 300

        return AnthropicOAuthToken(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=expires_at,
        )


def refresh_access_token(refresh_token: str) -> AnthropicOAuthToken:
    """Refresh an expired access token."""
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
        expires_at = time.time() + data["expires_in"] - 300

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

    Uses a local callback server to automatically capture the authorization code.
    Falls back to manual paste if the callback fails.
    """
    verifier, challenge = generate_pkce()

    if verbose:
        print("Starting local server for OAuth callback...")
    server, thread, port = _start_callback_server()
    redirect_uri = f"http://localhost:{port}/callback"

    try:
        auth_url = build_authorization_url(verifier, challenge, redirect_uri)

        if verbose:
            print("\nTo authenticate with Claude Max/Pro:")
            print(f"\n1. Visit: {auth_url}")
            print("\n2. Sign in with your Anthropic account")
            print("3. The callback will be captured automatically")
            print()

        if open_browser:
            if verbose:
                print("Opening browser...")
            webbrowser.open(auth_url)

        if verbose:
            print("Waiting for OAuth callback...")
        thread.join(timeout=300)

        if _OAuthCallbackHandler.error:
            msg = f"OAuth error: {_OAuthCallbackHandler.error}"
            raise RuntimeError(msg)

        code = _OAuthCallbackHandler.code
        state = _OAuthCallbackHandler.state

        # Fall back to manual paste if callback wasn't received
        if not code or not state:
            print("\nCallback not received. Paste the authorization code or full redirect URL:")
            try:
                user_input = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nAuthentication cancelled.")
                msg = "Authentication cancelled by user"
                raise RuntimeError(msg) from None

            if not user_input:
                msg = "No authorization code provided"
                raise RuntimeError(msg)

            # Parse input - could be code, code#state, or full URL
            parsed = _parse_authorization_input(user_input)
            code = parsed.get("code")
            state = parsed.get("state", verifier)

            if not code:
                msg = "Could not extract authorization code from input"
                raise RuntimeError(msg)

        # Verify state
        if state != verifier:
            msg = "OAuth state mismatch - possible CSRF attack"
            raise RuntimeError(msg)

        if verbose:
            print("\nExchanging code for token...")

        token = exchange_code_for_token(code, state, verifier, redirect_uri)

        if verbose:
            print("Authentication successful!")

        return token

    finally:
        server.server_close()


def _parse_authorization_input(user_input: str) -> dict[str, str]:
    """Parse authorization input which may be a code, code#state, or full URL."""
    value = user_input.strip()
    if not value:
        return {}

    # Try as URL
    try:
        parsed = urlparse(value)
        params = parse_qs(parsed.query)
        result: dict[str, str] = {}
        if "code" in params:
            result["code"] = params["code"][0]
        if "state" in params:
            result["state"] = params["state"][0]
        if result:
            return result
    except Exception:  # noqa: BLE001
        pass

    # Try code#state format
    if "#" in value:
        parts = value.split("#", maxsplit=1)
        return {"code": parts[0], "state": parts[1]}

    # Try query string format
    if "code=" in value:
        params = parse_qs(value)
        result = {}
        if "code" in params:
            result["code"] = params["code"][0]
        if "state" in params:
            result["state"] = params["state"][0]
        if result:
            return result

    # Plain code
    return {"code": value}


def get_or_refresh_token(
    store: AnthropicTokenStore | None = None,
) -> AnthropicOAuthToken:
    """Get a valid token, refreshing if necessary."""
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
    """Async version of get_or_refresh_token."""
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
            expires_at = time.time() + data["expires_in"] - 300

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


class AnthropicAuthBackend(ProviderAuthBackend):
    """Anthropic OAuth (PKCE) auth backend."""

    def __init__(self) -> None:
        self._pending_verifiers: dict[str, str] = {}

    @property
    def provider_id(self) -> str:
        return "anthropic"

    def methods(self) -> list[ProviderAuthMethod]:
        return [ProviderAuthMethod(type="oauth", label="Connect Claude Max/Pro")]

    async def authorize(self, method: int = 0) -> ProviderAuthAuthorization:
        verifier, challenge = generate_pkce()
        auth_url = build_authorization_url(verifier, challenge, OAUTH_MANUAL_REDIRECT_URI)
        self._pending_verifiers[verifier] = verifier
        return ProviderAuthAuthorization(
            url=auth_url,
            instructions="Sign in with your Anthropic account and copy the authorization code",
            method="code",
        )

    async def callback(
        self,
        *,
        code: str | None = None,
        device_code: str | None = None,
        verifier: str | None = None,
    ) -> bool:
        if not code or not verifier:
            raise ValueError("Missing code or verifier for Anthropic OAuth")
        token = exchange_code_for_token(code, verifier, verifier, OAUTH_MANUAL_REDIRECT_URI)
        store = AnthropicTokenStore()
        store.save(token)
        self._pending_verifiers.pop(verifier, None)
        return True

    async def set_credentials(self, info: AuthInfo) -> bool:
        if not isinstance(info, OAuthAuthInfo):
            return False
        store = AnthropicTokenStore()
        token = AnthropicOAuthToken(
            access_token=info.access,
            refresh_token=info.refresh,
            expires_at=info.expires,
        )
        store.save(token)
        return True

    async def remove_credentials(self) -> bool:
        store = AnthropicTokenStore()
        store.clear()
        return True


if __name__ == "__main__":
    anthropic_auth_main()
