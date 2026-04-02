"""OpenAI Codex (ChatGPT Plus/Pro) OAuth authentication.

This module implements OAuth 2.0 PKCE authentication for OpenAI Codex,
allowing users with ChatGPT Plus/Pro subscriptions to use their
subscription through the OpenAI API.

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
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlencode, urlparse
import webbrowser

import anyenv
import httpx

from llmling_models.auth.base import ProviderAuthBackend
from llmling_models.auth.models import ProviderAuthAuthorization, ProviderAuthMethod
from llmling_models.log import get_logger


if TYPE_CHECKING:
    from typing import Self

logger = get_logger(__name__)

# OAuth client credentials
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

# OAuth endpoints
OAUTH_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
OAUTH_REDIRECT_URI = "http://localhost:1455/auth/callback"
OAUTH_REDIRECT_PORT = 1455
OAUTH_SCOPE = "openid profile email offline_access"

# JWT claim path for extracting account ID
JWT_CLAIM_PATH = "https://api.openai.com/auth"

# Default token storage location
DEFAULT_TOKEN_PATH = Path.home() / ".config" / "llmling-models" / "openai_codex_oauth.json"


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge."""
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def _decode_jwt(token: str) -> dict[str, Any] | None:
    """Decode a JWT token payload without verification."""
    try:
        parts = token.split(".")
        if len(parts) != 3:  # noqa: PLR2004
            return None
        payload = parts[1]
        # Add padding
        padding = 4 - len(payload) % 4
        if padding != 4:  # noqa: PLR2004
            payload += "=" * padding
        decoded = base64.urlsafe_b64decode(payload)
        return anyenv.load_json(decoded, return_type=dict)
    except Exception:  # noqa: BLE001
        return None


def _get_account_id(access_token: str) -> str | None:
    """Extract ChatGPT account ID from JWT access token."""
    payload = _decode_jwt(access_token)
    if not payload:
        return None
    auth = payload.get(JWT_CLAIM_PATH, {})
    account_id = auth.get("chatgpt_account_id")
    return account_id if isinstance(account_id, str) and account_id else None


@dataclass
class OpenAICodexToken:
    """Stored OAuth token data for OpenAI Codex."""

    access_token: str
    refresh_token: str
    expires_at: float
    account_id: str

    def is_expired(self, buffer_seconds: int = 300) -> bool:
        """Check if the token is expired or about to expire."""
        return time.time() >= (self.expires_at - buffer_seconds)

    def to_dict(self) -> dict[str, str | float]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "account_id": self.account_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | float]) -> Self:
        return cls(
            access_token=str(data["access_token"]),
            refresh_token=str(data["refresh_token"]),
            expires_at=float(data["expires_at"]),
            account_id=str(data["account_id"]),
        )


@dataclass
class OpenAICodexTokenStore:
    """File-based token storage for OpenAI Codex OAuth."""

    path: Path = field(default_factory=lambda: DEFAULT_TOKEN_PATH)
    _token: OpenAICodexToken | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> OpenAICodexToken | None:
        if self._token is not None:
            return self._token

        if not self.path.exists():
            return None

        try:
            data = anyenv.load_json(self.path.read_text(), return_type=dict)
            self._token = OpenAICodexToken.from_dict(data)
        except (anyenv.JsonLoadError, KeyError, TypeError) as e:
            logger.warning("Failed to load token from %s: %s", self.path, e)
            return None
        else:
            return self._token

    def save(self, token: OpenAICodexToken) -> None:
        self._token = token
        self.path.write_text(json.dumps(token.to_dict(), indent=2))
        self.path.chmod(0o600)
        logger.debug("Saved token to %s", self.path)

    def clear(self) -> None:
        self._token = None
        if self.path.exists():
            self.path.unlink()
            logger.debug("Removed token from %s", self.path)

    def get_valid_token(self) -> OpenAICodexToken | None:
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
        parsed = urlparse(self.path)

        if parsed.path != "/auth/callback":
            self.send_response(404)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Not Found</h1>")
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

        state = params.get("state", [None])[0]
        code = params.get("code", [None])[0]

        if not code:
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Authentication Failed</h1><p>Missing authorization code.</p>")
            return

        _OAuthCallbackHandler.code = code
        _OAuthCallbackHandler.state = state
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(
            b"<h1>Authentication Successful</h1>"
            b"<p>You can close this window and return to the terminal.</p>"
        )


def _start_callback_server() -> tuple[socketserver.TCPServer, threading.Thread]:
    _OAuthCallbackHandler.code = None
    _OAuthCallbackHandler.state = None
    _OAuthCallbackHandler.error = None

    server = socketserver.TCPServer(("127.0.0.1", OAUTH_REDIRECT_PORT), _OAuthCallbackHandler)
    server.timeout = 300

    thread = threading.Thread(target=server.handle_request)
    thread.daemon = True
    thread.start()

    return server, thread


def _parse_authorization_input(user_input: str) -> dict[str, str]:
    """Parse authorization input (code, code#state, or full URL)."""
    value = user_input.strip()
    if not value:
        return {}

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

    if "#" in value:
        parts = value.split("#", maxsplit=1)
        return {"code": parts[0], "state": parts[1]}

    return {"code": value}


def build_authorization_url(verifier: str, challenge: str, state: str) -> str:
    """Build the OAuth authorization URL."""
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": OAUTH_REDIRECT_URI,
        "scope": OAUTH_SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "llmling",
    }
    return f"{OAUTH_AUTHORIZE_URL}?{urlencode(params)}"


def exchange_code_for_token(code: str, verifier: str) -> OpenAICodexToken:
    """Exchange authorization code for access token."""
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            OAUTH_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": OAUTH_REDIRECT_URI,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if not response.is_success:
            msg = f"Token exchange failed: {response.status_code} - {response.text}"
            raise RuntimeError(msg)

        data = response.json()

        if not data.get("access_token") or not data.get("refresh_token"):
            msg = f"Token response missing fields: {data}"
            raise RuntimeError(msg)

        expires_in = data.get("expires_in", 3600)
        expires_at = time.time() + expires_in

        account_id = _get_account_id(data["access_token"])
        if not account_id:
            msg = "Failed to extract account ID from access token"
            raise RuntimeError(msg)

        return OpenAICodexToken(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=expires_at,
            account_id=account_id,
        )


def refresh_access_token(refresh_token: str) -> OpenAICodexToken:
    """Refresh an expired access token."""
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            OAUTH_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if not response.is_success:
            msg = f"Token refresh failed: {response.status_code} - {response.text}"
            raise RuntimeError(msg)

        data = response.json()

        if not data.get("access_token") or not data.get("refresh_token"):
            msg = f"Refresh response missing fields: {data}"
            raise RuntimeError(msg)

        expires_in = data.get("expires_in", 3600)
        expires_at = time.time() + expires_in

        account_id = _get_account_id(data["access_token"])
        if not account_id:
            msg = "Failed to extract account ID from refreshed token"
            raise RuntimeError(msg)

        return OpenAICodexToken(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=expires_at,
            account_id=account_id,
        )


def authenticate_openai_codex(
    verbose: bool = True,
    open_browser: bool = True,
) -> OpenAICodexToken:
    """Authenticate with OpenAI Codex using OAuth.

    Uses a local callback server to capture the authorization code,
    with fallback to manual paste.
    """
    verifier, challenge = generate_pkce()
    state = secrets.token_hex(16)

    if verbose:
        print("Starting local server for OAuth callback...")
    server, thread = _start_callback_server()

    try:
        auth_url = build_authorization_url(verifier, challenge, state)

        if verbose:
            print("\nTo authenticate with ChatGPT Plus/Pro:")
            print(f"\n1. Visit: {auth_url}")
            print("\n2. Sign in with your OpenAI account")
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
        cb_state = _OAuthCallbackHandler.state

        # Fall back to manual paste
        if not code:
            print("\nCallback not received. Paste the authorization code or full redirect URL:")
            try:
                user_input = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nAuthentication cancelled.")
                msg = "Authentication cancelled by user"
                raise RuntimeError(msg) from None

            parsed = _parse_authorization_input(user_input)
            code = parsed.get("code")
            cb_state = parsed.get("state", state)

        if not code:
            msg = "No authorization code received"
            raise RuntimeError(msg)

        # Verify state if we got one back
        if cb_state and cb_state != state:
            msg = "OAuth state mismatch"
            raise RuntimeError(msg)

        if verbose:
            print("\nExchanging code for token...")

        token = exchange_code_for_token(code, verifier)

        if verbose:
            print("Authentication successful!")
            print(f"Account ID: {token.account_id}")

        return token

    finally:
        server.server_close()


def get_or_refresh_token(
    store: OpenAICodexTokenStore | None = None,
) -> OpenAICodexToken:
    """Get a valid token, refreshing if necessary."""
    if store is None:
        store = OpenAICodexTokenStore()

    token = store.load()
    if token is None:
        msg = (
            "No OpenAI Codex OAuth token found. "
            "Run 'llmling-models openai-codex-auth' to authenticate."
        )
        raise RuntimeError(msg)

    if token.is_expired():
        logger.info("Token expired, refreshing...")
        token = refresh_access_token(token.refresh_token)
        store.save(token)

    return token


async def get_or_refresh_token_async(
    store: OpenAICodexTokenStore | None = None,
) -> OpenAICodexToken:
    """Async version of get_or_refresh_token."""
    if store is None:
        store = OpenAICodexTokenStore()

    token = store.load()
    if token is None:
        msg = (
            "No OpenAI Codex OAuth token found. "
            "Run 'llmling-models openai-codex-auth' to authenticate."
        )
        raise RuntimeError(msg)

    if token.is_expired():
        logger.info("Token expired, refreshing...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                OAUTH_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": token.refresh_token,
                    "client_id": CLIENT_ID,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            if not response.is_success:
                msg = f"Token refresh failed: {response.status_code} - {response.text}"
                raise RuntimeError(msg)

            data = response.json()

            if not data.get("access_token") or not data.get("refresh_token"):
                msg = "Refresh response missing fields"
                raise RuntimeError(msg)

            expires_at = time.time() + data.get("expires_in", 3600)
            account_id = _get_account_id(data["access_token"])
            if not account_id:
                msg = "Failed to extract account ID from refreshed token"
                raise RuntimeError(msg)

            token = OpenAICodexToken(
                access_token=data["access_token"],
                refresh_token=data["refresh_token"],
                expires_at=expires_at,
                account_id=account_id,
            )
            store.save(token)

    return token


def openai_codex_auth_main() -> None:
    """Command-line entry point for OpenAI Codex OAuth authentication."""
    parser = argparse.ArgumentParser(
        description="Authenticate with OpenAI Codex (ChatGPT Plus/Pro) using OAuth."
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
    store = OpenAICodexTokenStore(path=args.token_path)

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
            print(f"Account ID: {token.account_id}")
            print(f"Token path: {args.token_path}")
        return

    try:
        token = authenticate_openai_codex(
            verbose=True,
            open_browser=not args.no_browser,
        )
        store.save(token)
        print(f"\nToken saved to: {args.token_path}")
        print("You can now use OpenAI Codex models with auth_method='oauth'")
    except Exception as e:
        logger.exception("Authentication failed")
        print(f"\nAuthentication failed: {e}", file=sys.stderr)
        sys.exit(1)


class OpenAICodexAuthBackend(ProviderAuthBackend):
    """OpenAI Codex (ChatGPT Plus/Pro) OAuth backend."""

    def __init__(self) -> None:
        self._pending: dict[str, tuple[str, str]] = {}  # state -> (verifier, state)

    @property
    def provider_id(self) -> str:
        return "openai-codex"

    def methods(self) -> list[ProviderAuthMethod]:
        return [ProviderAuthMethod(type="oauth", label="Connect ChatGPT Plus/Pro")]

    async def authorize(self, method: int = 0) -> ProviderAuthAuthorization:
        import secrets as _secrets

        verifier, challenge = generate_pkce()
        state = _secrets.token_hex(16)
        auth_url = build_authorization_url(verifier, challenge, state)
        self._pending[state] = (verifier, state)
        return ProviderAuthAuthorization(
            url=auth_url,
            instructions="Sign in with your OpenAI account",
            method="auto",
        )

    async def callback(
        self,
        *,
        code: str | None = None,
        device_code: str | None = None,
        verifier: str | None = None,
    ) -> bool:
        if not code or not verifier:
            msg = "Missing code or verifier for OpenAI Codex OAuth"
            raise ValueError(msg)
        # verifier here is the PKCE verifier stored during authorize
        token = exchange_code_for_token(code, verifier)
        store = OpenAICodexTokenStore()
        store.save(token)
        return True

    async def remove_credentials(self) -> bool:
        OpenAICodexTokenStore().clear()
        return True


if __name__ == "__main__":
    openai_codex_auth_main()
