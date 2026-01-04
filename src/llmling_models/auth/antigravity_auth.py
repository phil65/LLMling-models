"""Antigravity OAuth authentication (Gemini 3, Claude, GPT-OSS via Google Cloud).

This module implements OAuth 2.0 PKCE authentication for Antigravity,
which provides access to additional models beyond standard Gemini CLI
including Gemini 3, Claude, and GPT-OSS via Google Cloud.

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
import webbrowser

import httpx

from llmling_models.log import get_logger


if TYPE_CHECKING:
    from typing import Self

logger = get_logger(__name__)

# Antigravity OAuth client credentials (different from Gemini CLI)
_CLIENT_ID = base64.b64decode(
    "MTA3MTAwNjA2MDU5MS10bWhzc2luMmgyMWxjcmUyMzV2dG9sb2poNGc0MDNlcC5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbQ=="
).decode()
_CLIENT_SECRET = base64.b64decode("R09DU1BYLUs1OEZXUjQ4NkxkTEoxbUxCOHNYQzR6NnFEQWY=").decode()

# OAuth endpoints
OAUTH_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
OAUTH_REDIRECT_URI = "http://localhost:51121/oauth-callback"
OAUTH_REDIRECT_PORT = 51121

# Antigravity requires additional scopes beyond Gemini CLI
OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]

# Cloud Code Assist endpoints (prod first, then sandbox)
CODE_ASSIST_ENDPOINTS = [
    "https://cloudcode-pa.googleapis.com",
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
]

# Fallback project ID when discovery fails
DEFAULT_PROJECT_ID = "rising-fact-p41fc"

# Default token storage location
DEFAULT_TOKEN_PATH = Path.home() / ".config" / "llmling-models" / "antigravity_oauth.json"


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
class AntigravityOAuthToken:
    """Stored OAuth token data for Antigravity."""

    access_token: str
    refresh_token: str
    expires_at: float  # Unix timestamp
    project_id: str
    email: str | None = None

    def is_expired(self, buffer_seconds: int = 300) -> bool:
        """Check if the token is expired or about to expire.

        Uses a 5-minute buffer by default to ensure we refresh before expiry.
        """
        return time.time() >= (self.expires_at - buffer_seconds)

    def to_dict(self) -> dict[str, str | float | None]:
        """Convert to dictionary for JSON serialization."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "project_id": self.project_id,
            "email": self.email,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | float | None]) -> Self:
        """Create from dictionary."""
        return cls(
            access_token=str(data["access_token"]),
            refresh_token=str(data["refresh_token"]),
            expires_at=float(data["expires_at"]),  # type: ignore[arg-type]
            project_id=str(data["project_id"]),
            email=str(data["email"]) if data.get("email") else None,
        )


@dataclass
class AntigravityTokenStore:
    """File-based token storage for Antigravity OAuth."""

    path: Path = field(default_factory=lambda: DEFAULT_TOKEN_PATH)
    _token: AntigravityOAuthToken | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Ensure storage directory exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> AntigravityOAuthToken | None:
        """Load token from file."""
        if self._token is not None:
            return self._token

        if not self.path.exists():
            return None

        try:
            data = json.loads(self.path.read_text())
            self._token = AntigravityOAuthToken.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to load token from %s: %s", self.path, e)
            return None
        else:
            return self._token

    def save(self, token: AntigravityOAuthToken) -> None:
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

    def get_valid_token(self) -> AntigravityOAuthToken | None:
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
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(self.path)

        if parsed.path == "/oauth-callback":
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
        else:
            self.send_response(404)
            self.end_headers()


def _start_callback_server() -> tuple[socketserver.TCPServer, threading.Thread]:
    """Start local HTTP server for OAuth callback.

    Returns:
        Tuple of (server, thread)
    """
    # Reset handler state
    _OAuthCallbackHandler.code = None
    _OAuthCallbackHandler.state = None
    _OAuthCallbackHandler.error = None

    server = socketserver.TCPServer(("127.0.0.1", OAUTH_REDIRECT_PORT), _OAuthCallbackHandler)
    server.timeout = 300  # 5 minute timeout

    thread = threading.Thread(target=server.handle_request)
    thread.daemon = True
    thread.start()

    return server, thread


def _get_user_email(access_token: str) -> str | None:
    """Get user email from access token."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                "https://www.googleapis.com/oauth2/v1/userinfo",
                params={"alt": "json"},
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if response.is_success:
                data = response.json()
                return data.get("email")  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001
        pass
    return None


def _discover_project(
    access_token: str,
    verbose: bool = True,
) -> str:
    """Discover a project for the user.

    Tries multiple Cloud Code Assist endpoints and falls back to default project.

    Args:
        access_token: Valid OAuth access token
        verbose: Whether to print progress messages

    Returns:
        Project ID
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "google-api-nodejs-client/9.15.1",
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
        "Client-Metadata": json.dumps({
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }),
    }

    if verbose:
        print("Checking for existing project...")

    with httpx.Client(timeout=60.0) as client:
        # Try endpoints in order: prod first, then sandbox
        for endpoint in CODE_ASSIST_ENDPOINTS:
            try:
                response = client.post(
                    f"{endpoint}/v1internal:loadCodeAssist",
                    headers=headers,
                    json={
                        "metadata": {
                            "ideType": "IDE_UNSPECIFIED",
                            "platform": "PLATFORM_UNSPECIFIED",
                            "pluginType": "GEMINI",
                        },
                    },
                )

                if response.is_success:
                    data = response.json()
                    project = data.get("cloudaicompanionProject")

                    # Handle both string and object formats
                    if isinstance(project, str) and project:
                        return project
                    if isinstance(project, dict) and project.get("id"):
                        return project["id"]  # type: ignore[no-any-return]
            except Exception:  # noqa: BLE001
                # Try next endpoint
                continue

    # Use fallback project ID
    if verbose:
        print("Using default project...")
    return DEFAULT_PROJECT_ID


def build_authorization_url(verifier: str, challenge: str) -> str:
    """Build the OAuth authorization URL.

    Args:
        verifier: PKCE code verifier (used as state)
        challenge: PKCE code challenge

    Returns:
        The authorization URL to open in browser
    """
    params = {
        "client_id": _CLIENT_ID,
        "response_type": "code",
        "redirect_uri": OAUTH_REDIRECT_URI,
        "scope": " ".join(OAUTH_SCOPES),
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
        "access_type": "offline",
        "prompt": "consent",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{OAUTH_AUTHORIZE_URL}?{query}"


def exchange_code_for_token(code: str, verifier: str) -> dict[str, str | int]:
    """Exchange authorization code for access token.

    Args:
        code: The authorization code from callback
        verifier: The PKCE code verifier

    Returns:
        Token response data

    Raises:
        RuntimeError: If token exchange fails
    """
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            OAUTH_TOKEN_URL,
            data={
                "client_id": _CLIENT_ID,
                "client_secret": _CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": OAUTH_REDIRECT_URI,
                "code_verifier": verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if not response.is_success:
            msg = f"Token exchange failed: {response.status_code} - {response.text}"
            raise RuntimeError(msg)

        return response.json()  # type: ignore[no-any-return]


def refresh_access_token(
    refresh_token: str,
    project_id: str,
    email: str | None = None,
) -> AntigravityOAuthToken:
    """Refresh an expired access token.

    Args:
        refresh_token: The refresh token
        project_id: The project ID to preserve
        email: The email to preserve

    Returns:
        New OAuth token

    Raises:
        RuntimeError: If refresh fails
    """
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            OAUTH_TOKEN_URL,
            data={
                "client_id": _CLIENT_ID,
                "client_secret": _CLIENT_SECRET,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if not response.is_success:
            msg = f"Token refresh failed: {response.status_code} - {response.text}"
            raise RuntimeError(msg)

        data = response.json()
        # Calculate expiry (expires_in seconds - 5 min buffer)
        expires_at = time.time() + data["expires_in"] - 300

        return AntigravityOAuthToken(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", refresh_token),
            expires_at=expires_at,
            project_id=project_id,
            email=email,
        )


def authenticate_antigravity(
    verbose: bool = True,
    open_browser: bool = True,
) -> AntigravityOAuthToken:
    """Authenticate with Antigravity using OAuth.

    This initiates the OAuth PKCE flow:
    1. Starts a local HTTP server for the callback
    2. Opens browser to Google authorization page
    3. User authenticates and authorizes the application
    4. Callback is captured automatically
    5. Code is exchanged for access/refresh tokens
    6. Project is discovered

    Args:
        verbose: Whether to print status messages
        open_browser: Whether to automatically open the browser

    Returns:
        The OAuth token with project info

    Raises:
        RuntimeError: If authentication fails
    """
    # Generate PKCE challenge
    verifier, challenge = generate_pkce()

    # Start callback server
    if verbose:
        print("Starting local server for OAuth callback...")
    server, thread = _start_callback_server()

    try:
        # Build authorization URL
        auth_url = build_authorization_url(verifier, challenge)

        if verbose:
            print("\nTo authenticate with Antigravity:")
            print(f"\n1. Visit: {auth_url}")
            print("\n2. Sign in with your Google account")
            print("3. The callback will be captured automatically")
            print()

        if open_browser:
            if verbose:
                print("Opening browser...")
            webbrowser.open(auth_url)

        # Wait for callback
        if verbose:
            print("Waiting for OAuth callback...")
        thread.join(timeout=300)  # 5 minute timeout

        # Check for errors
        if _OAuthCallbackHandler.error:
            msg = f"OAuth error: {_OAuthCallbackHandler.error}"
            raise RuntimeError(msg)

        if not _OAuthCallbackHandler.code or not _OAuthCallbackHandler.state:
            msg = "OAuth callback timed out or was not received"
            raise RuntimeError(msg)

        # Verify state
        if _OAuthCallbackHandler.state != verifier:
            msg = "OAuth state mismatch - possible CSRF attack"
            raise RuntimeError(msg)

        # Exchange code for token
        if verbose:
            print("\nExchanging authorization code for tokens...")

        token_data = exchange_code_for_token(_OAuthCallbackHandler.code, verifier)

        if not token_data.get("refresh_token"):
            msg = "No refresh token received. Please try again."
            raise RuntimeError(msg)

        access_token = token_data["access_token"]

        # Get user email
        if verbose:
            print("Getting user info...")
        email = _get_user_email(str(access_token))

        # Discover project
        project_id = _discover_project(str(access_token), verbose=verbose)

        # Calculate expiry (expires_in seconds - 5 min buffer)
        expires_at = time.time() + int(token_data["expires_in"]) - 300

        if verbose:
            print("\nAuthentication successful!")
            if email:
                print(f"Logged in as: {email}")
            print(f"Project ID: {project_id}")

        return AntigravityOAuthToken(
            access_token=str(access_token),
            refresh_token=str(token_data["refresh_token"]),
            expires_at=expires_at,
            project_id=project_id,
            email=email,
        )

    finally:
        server.server_close()


def get_or_refresh_token(
    store: AntigravityTokenStore | None = None,
) -> AntigravityOAuthToken:
    """Get a valid token, refreshing if necessary.

    Args:
        store: Token store to use (defaults to standard location)

    Returns:
        Valid OAuth token

    Raises:
        RuntimeError: If no token exists and authentication is needed
    """
    if store is None:
        store = AntigravityTokenStore()

    token = store.load()
    if token is None:
        msg = (
            "No Antigravity OAuth token found. "
            "Run 'llmling-models antigravity-auth' to authenticate."
        )
        raise RuntimeError(msg)

    if token.is_expired():
        logger.info("Token expired, refreshing...")
        token = refresh_access_token(token.refresh_token, token.project_id, token.email)
        store.save(token)

    return token


async def get_or_refresh_token_async(
    store: AntigravityTokenStore | None = None,
) -> AntigravityOAuthToken:
    """Async version of get_or_refresh_token.

    Args:
        store: Token store to use (defaults to standard location)

    Returns:
        Valid OAuth token

    Raises:
        RuntimeError: If no token exists and authentication is needed
    """
    if store is None:
        store = AntigravityTokenStore()

    token = store.load()
    if token is None:
        msg = (
            "No Antigravity OAuth token found. "
            "Run 'llmling-models antigravity-auth' to authenticate."
        )
        raise RuntimeError(msg)

    if token.is_expired():
        logger.info("Token expired, refreshing...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                OAUTH_TOKEN_URL,
                data={
                    "client_id": _CLIENT_ID,
                    "client_secret": _CLIENT_SECRET,
                    "refresh_token": token.refresh_token,
                    "grant_type": "refresh_token",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            if not response.is_success:
                msg = f"Token refresh failed: {response.status_code} - {response.text}"
                raise RuntimeError(msg)

            data = response.json()
            expires_at = time.time() + data["expires_in"] - 300

            token = AntigravityOAuthToken(
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token", token.refresh_token),
                expires_at=expires_at,
                project_id=token.project_id,
                email=token.email,
            )
            store.save(token)

    return token


def antigravity_auth_main() -> None:
    """Command-line entry point for Antigravity OAuth authentication."""
    parser = argparse.ArgumentParser(
        description="Authenticate with Antigravity (Gemini 3, Claude, GPT-OSS) using OAuth."
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
    store = AntigravityTokenStore(path=args.token_path)

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
            if token.email:
                print(f"Email: {token.email}")
            print(f"Project ID: {token.project_id}")
            print(f"Token path: {args.token_path}")
        return

    try:
        token = authenticate_antigravity(
            verbose=True,
            open_browser=not args.no_browser,
        )
        store.save(token)
        print(f"\nToken saved to: {args.token_path}")
        print("You can now use Antigravity models (Gemini 3, Claude, GPT-OSS)")
    except Exception as e:
        logger.exception("Authentication failed")
        print(f"\nAuthentication failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    antigravity_auth_main()
