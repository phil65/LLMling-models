"""Gemini CLI OAuth authentication (Google Cloud Code Assist).

This module implements OAuth 2.0 PKCE authentication for Gemini CLI,
allowing users to use standard Gemini models (gemini-2.0-flash, gemini-2.5-*)
through Google Cloud Code Assist.

Based on the pi-mono implementation by badlogic.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, field
import hashlib
import http.server
import json
import os
from pathlib import Path
import secrets
import socketserver
import sys
import threading
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse
import webbrowser

import anyenv
import httpx

from llmling_models.log import get_logger


if TYPE_CHECKING:
    from typing import Self

logger = get_logger(__name__)

# OAuth client credentials (from Gemini CLI / Google Cloud Code Assist)
_CLIENT_ID = base64.b64decode(
    "NjgxMjU1ODA5Mzk1LW9vOGZ0Mm9wcmRybnA5ZTNhcWY2YXYzaG1kaWIxMzVqLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29t"
).decode()
_CLIENT_SECRET = base64.b64decode("R09DU1BYLTR1SGdNUG0tMW83U2stZ2VWNkN1NWNsWEZzeGw=").decode()

# OAuth endpoints
OAUTH_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
OAUTH_REDIRECT_URI = "http://localhost:8085/oauth2callback"
OAUTH_REDIRECT_PORT = 8085

# Required scopes
OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

# Google Cloud Code Assist endpoint
CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"

# Tier IDs as used by the Cloud Code API
TIER_FREE = "free-tier"
TIER_LEGACY = "legacy-tier"

# Default token storage location
DEFAULT_TOKEN_PATH = Path.home() / ".config" / "llmling-models" / "gemini_oauth.json"


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge."""
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


@dataclass
class GeminiOAuthToken:
    """Stored OAuth token data for Gemini CLI."""

    access_token: str
    refresh_token: str
    expires_at: float
    project_id: str
    email: str | None = None

    def is_expired(self, buffer_seconds: int = 300) -> bool:
        """Check if the token is expired or about to expire."""
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
class GeminiTokenStore:
    """File-based token storage for Gemini OAuth."""

    path: Path = field(default_factory=lambda: DEFAULT_TOKEN_PATH)
    _token: GeminiOAuthToken | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> GeminiOAuthToken | None:
        if self._token is not None:
            return self._token

        if not self.path.exists():
            return None

        try:
            data = anyenv.load_json(self.path.read_text(), return_type=dict)
            self._token = GeminiOAuthToken.from_dict(data)
        except (anyenv.JsonLoadError, KeyError, TypeError) as e:
            logger.warning("Failed to load token from %s: %s", self.path, e)
            return None
        else:
            return self._token

    def save(self, token: GeminiOAuthToken) -> None:
        self._token = token
        self.path.write_text(json.dumps(token.to_dict(), indent=2))
        self.path.chmod(0o600)
        logger.debug("Saved token to %s", self.path)

    def clear(self) -> None:
        self._token = None
        if self.path.exists():
            self.path.unlink()
            logger.debug("Removed token from %s", self.path)

    def get_valid_token(self) -> GeminiOAuthToken | None:
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

        if parsed.path != "/oauth2callback":
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


def _get_user_email(access_token: str) -> str | None:
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                "https://www.googleapis.com/oauth2/v1/userinfo",
                params={"alt": "json"},
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if response.is_success:
                data = anyenv.load_json(response.text, return_type=dict)
                email: str | None = data.get("email")
                return email
    except Exception:  # noqa: BLE001
        pass
    return None


def _is_vpc_sc_affected(error_payload: dict[str, Any]) -> bool:
    """Check if error indicates a VPC Service Controls affected user."""
    details = error_payload.get("error", {}).get("details", [])
    return any(d.get("reason") == "SECURITY_POLICY_VIOLATED" for d in details)


def _get_default_tier(
    allowed_tiers: list[dict[str, str | bool]] | None,
) -> str:
    """Get default tier ID from allowed tiers list."""
    if not allowed_tiers:
        return TIER_LEGACY
    default = next((t for t in allowed_tiers if t.get("isDefault")), None)
    return str(default.get("id", TIER_LEGACY)) if default else TIER_LEGACY


def _poll_operation(
    client: httpx.Client,
    operation_name: str,
    headers: dict[str, str],
    verbose: bool = True,
) -> dict[str, Any]:
    """Poll a long-running operation until completion."""
    attempt = 0
    while True:
        if attempt > 0:
            if verbose:
                print(f"Waiting for project provisioning (attempt {attempt + 1})...")
            time.sleep(5)

        response = client.get(
            f"{CODE_ASSIST_ENDPOINT}/v1internal/{operation_name}",
            headers=headers,
        )
        if not response.is_success:
            msg = f"Failed to poll operation: {response.status_code}"
            raise RuntimeError(msg)

        data = anyenv.load_json(response.text, return_type=dict)
        if data.get("done"):
            return data

        attempt += 1


def _discover_project(
    access_token: str,
    verbose: bool = True,
) -> str:
    """Discover or provision a Google Cloud project for the user.

    Handles:
    - Existing projects (returns immediately)
    - VPC-SC affected users (requires GOOGLE_CLOUD_PROJECT env var)
    - Free tier onboarding with LRO polling
    - Non-free tiers requiring explicit project ID
    """
    env_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get(
        "GOOGLE_CLOUD_PROJECT_ID"
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "google-api-nodejs-client/9.15.1",
        "X-Goog-Api-Client": "gl-python/3.12",
    }

    with httpx.Client(timeout=60.0) as client:
        if verbose:
            print("Checking for existing Cloud Code Assist project...")

        response = client.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            headers=headers,
            json={
                "cloudaicompanionProject": env_project_id,
                "metadata": {
                    "ideType": "IDE_UNSPECIFIED",
                    "platform": "PLATFORM_UNSPECIFIED",
                    "pluginType": "GEMINI",
                    "duetProject": env_project_id,
                },
            },
        )

        if not response.is_success:
            # Check for VPC-SC
            try:
                error_payload = response.json()
            except Exception:  # noqa: BLE001
                error_payload = {}

            if _is_vpc_sc_affected(error_payload):
                data = {"currentTier": {"id": "standard-tier"}}
            else:
                msg = f"loadCodeAssist failed: {response.status_code} - {response.text}"
                raise RuntimeError(msg)
        else:
            data = anyenv.load_json(response.text, return_type=dict)

        # User has an existing tier and project
        if data.get("currentTier"):
            if data.get("cloudaicompanionProject"):
                return str(data["cloudaicompanionProject"])
            # Has tier but no managed project - need env var
            if env_project_id:
                return env_project_id
            msg = (
                "This account requires setting the GOOGLE_CLOUD_PROJECT or "
                "GOOGLE_CLOUD_PROJECT_ID environment variable. "
                "See https://goo.gle/gemini-cli-auth-docs#workspace-gca"
            )
            raise RuntimeError(msg)

        # Need onboarding - get default tier
        tier_id = _get_default_tier(data.get("allowedTiers"))  # type: ignore[arg-type]

        if tier_id != TIER_FREE and not env_project_id:
            msg = (
                "This account requires setting the GOOGLE_CLOUD_PROJECT or "
                "GOOGLE_CLOUD_PROJECT_ID environment variable. "
                "See https://goo.gle/gemini-cli-auth-docs#workspace-gca"
            )
            raise RuntimeError(msg)

        if verbose:
            print("Provisioning Cloud Code Assist project (this may take a moment)...")

        onboard_body: dict[str, Any] = {
            "tierId": tier_id,
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            },
        }

        if tier_id != TIER_FREE and env_project_id:
            onboard_body["cloudaicompanionProject"] = env_project_id
            onboard_body["metadata"]["duetProject"] = env_project_id

        onboard_response = client.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
            headers=headers,
            json=onboard_body,
        )

        if not onboard_response.is_success:
            msg = f"onboardUser failed: {onboard_response.status_code} - {onboard_response.text}"
            raise RuntimeError(msg)

        lro_data = onboard_response.json()

        # Poll if not done yet
        if not lro_data.get("done") and lro_data.get("name"):
            lro_data = _poll_operation(client, lro_data["name"], headers, verbose)

        project_id = lro_data.get("response", {}).get("cloudaicompanionProject", {}).get("id")
        if project_id:
            return str(project_id)

        if env_project_id:
            return env_project_id

    msg = (
        "Could not discover or provision a Google Cloud project. "
        "Try setting the GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_PROJECT_ID "
        "environment variable. "
        "See https://goo.gle/gemini-cli-auth-docs#workspace-gca"
    )
    raise RuntimeError(msg)


def build_authorization_url(verifier: str, challenge: str) -> str:
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

        data = response.json()
        assert isinstance(data, dict)
        return data


def refresh_access_token(
    refresh_token: str,
    project_id: str,
    email: str | None = None,
) -> GeminiOAuthToken:
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
        expires_at = time.time() + data["expires_in"] - 300

        return GeminiOAuthToken(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", refresh_token),
            expires_at=expires_at,
            project_id=project_id,
            email=email,
        )


def authenticate_gemini_cli(
    verbose: bool = True,
    open_browser: bool = True,
) -> GeminiOAuthToken:
    """Authenticate with Gemini CLI using OAuth."""
    verifier, challenge = generate_pkce()

    if verbose:
        print("Starting local server for OAuth callback...")
    server, thread = _start_callback_server()

    try:
        auth_url = build_authorization_url(verifier, challenge)

        if verbose:
            print("\nTo authenticate with Gemini CLI:")
            print(f"\n1. Visit: {auth_url}")
            print("\n2. Sign in with your Google account")
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

        if not _OAuthCallbackHandler.code or not _OAuthCallbackHandler.state:
            msg = "OAuth callback timed out or was not received"
            raise RuntimeError(msg)

        if _OAuthCallbackHandler.state != verifier:
            msg = "OAuth state mismatch - possible CSRF attack"
            raise RuntimeError(msg)

        if verbose:
            print("\nExchanging authorization code for tokens...")

        token_data = exchange_code_for_token(_OAuthCallbackHandler.code, verifier)

        if not token_data.get("refresh_token"):
            msg = "No refresh token received. Please try again."
            raise RuntimeError(msg)

        access_token = token_data["access_token"]

        if verbose:
            print("Getting user info...")
        email = _get_user_email(str(access_token))

        project_id = _discover_project(str(access_token), verbose=verbose)

        expires_at = time.time() + int(token_data["expires_in"]) - 300

        if verbose:
            print("\nAuthentication successful!")
            if email:
                print(f"Logged in as: {email}")
            print(f"Project ID: {project_id}")

        return GeminiOAuthToken(
            access_token=str(access_token),
            refresh_token=str(token_data["refresh_token"]),
            expires_at=expires_at,
            project_id=project_id,
            email=email,
        )

    finally:
        server.server_close()


def get_or_refresh_token(store: GeminiTokenStore | None = None) -> GeminiOAuthToken:
    if store is None:
        store = GeminiTokenStore()

    token = store.load()
    if token is None:
        msg = "No Gemini OAuth token found. Run 'llmling-models gemini-auth' to authenticate."
        raise RuntimeError(msg)

    if token.is_expired():
        logger.info("Token expired, refreshing...")
        token = refresh_access_token(token.refresh_token, token.project_id, token.email)
        store.save(token)

    return token


async def get_or_refresh_token_async(
    store: GeminiTokenStore | None = None,
) -> GeminiOAuthToken:
    if store is None:
        store = GeminiTokenStore()

    token = store.load()
    if token is None:
        msg = "No Gemini OAuth token found. Run 'llmling-models gemini-auth' to authenticate."
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

            token = GeminiOAuthToken(
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token", token.refresh_token),
                expires_at=expires_at,
                project_id=token.project_id,
                email=token.email,
            )
            store.save(token)

    return token


def gemini_auth_main() -> None:
    """Command-line entry point for Gemini OAuth authentication."""
    parser = argparse.ArgumentParser(
        description="Authenticate with Gemini CLI (Google Cloud Code Assist) using OAuth."
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
    store = GeminiTokenStore(path=args.token_path)

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
        token = authenticate_gemini_cli(
            verbose=True,
            open_browser=not args.no_browser,
        )
        store.save(token)
        print(f"\nToken saved to: {args.token_path}")
        print("You can now use Gemini models with auth_method='oauth'")
    except Exception as e:
        logger.exception("Authentication failed")
        print(f"\nAuthentication failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    gemini_auth_main()
