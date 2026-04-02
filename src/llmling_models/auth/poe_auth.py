"""Poe OAuth authentication.

OAuth PKCE flow for Poe (poe.com) which returns an API key.
The API key can also be entered manually as a fallback.

Manage your API keys at: https://poe.com/api_key
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import sys
import threading
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
import webbrowser

import anyenv
import httpx

from llmling_models.auth.base import ProviderAuthBackend
from llmling_models.auth.models import ProviderAuthAuthorization, ProviderAuthMethod
from llmling_models.log import get_logger


logger = get_logger(__name__)

POE_CLIENT_ID = "client_728290227fc048cc9262091a1ea197ea"
POE_AUTHORIZE_URL = "https://poe.com/oauth/authorize"
POE_TOKEN_URL = "https://poe.com/api/oauth/token"
POE_API_URL = "https://api.poe.com/v1"
POE_CALLBACK_PORT = 0  # Dynamic port
POE_REDIRECT_URI_BASE = "http://localhost"
DEFAULT_TOKEN_PATH = Path.home() / ".config" / "llmling-models" / "poe_auth.json"


class PoeTokenStore:
    """File-based storage for Poe API key."""

    def __init__(self, path: Path = DEFAULT_TOKEN_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> str | None:
        """Load stored API key."""
        if not self.path.exists():
            return None
        try:
            data = anyenv.load_json(self.path.read_text(), return_type=dict)
            api_key = data.get("api_key")
            return str(api_key) if api_key else None
        except (anyenv.JsonLoadError, KeyError) as e:
            logger.warning("Failed to load Poe API key from %s: %s", self.path, e)
            return None

    def save(self, api_key: str, expires_in: int | None = None) -> None:
        """Save API key."""
        data: dict[str, Any] = {"api_key": api_key}
        if expires_in is not None:
            data["expires_in"] = expires_in
        self.path.write_text(json.dumps(data, indent=2))
        self.path.chmod(0o600)

    def clear(self) -> None:
        """Remove stored API key."""
        if self.path.exists():
            self.path.unlink()


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE verifier and challenge."""
    verifier = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def build_authorization_url(
    verifier: str,
    challenge: str,
    redirect_uri: str,
    state: str | None = None,
) -> str:
    """Build the Poe OAuth authorization URL."""
    params = {
        "client_id": POE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    if state:
        params["state"] = state
    return f"{POE_AUTHORIZE_URL}?{urlencode(params)}"


def exchange_code_for_token(
    code: str,
    verifier: str,
    redirect_uri: str,
) -> dict[str, Any]:
    """Exchange authorization code for API key.

    Returns:
        Dict with 'api_key' and optionally 'expires_in'.
    """
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            POE_TOKEN_URL,
            json={
                "client_id": POE_CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        resp.raise_for_status()
        data = anyenv.load_json(resp.text, return_type=dict)

    api_key = data.get("apiKey") or data.get("api_key") or data.get("access_token")
    if not api_key:
        msg = f"No API key in Poe response: {list(data.keys())}"
        raise ValueError(msg)
    return {
        "api_key": str(api_key),
        "expires_in": data.get("expiresIn") or data.get("expires_in"),
    }


def validate_poe_key(api_key: str) -> bool:
    """Validate a Poe API key by fetching models."""
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(
            f"{POE_API_URL}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        return resp.is_success


def authenticate_poe(*, manual_key: str | None = None) -> str:
    """Run the full Poe authentication flow.

    Args:
        manual_key: If provided, skip OAuth and use this API key directly.

    Returns:
        The API key.
    """
    if manual_key:
        if not validate_poe_key(manual_key):
            msg = "Invalid Poe API key"
            raise ValueError(msg)
        return manual_key

    verifier, challenge = generate_pkce()
    state = secrets.token_hex(16)
    result: dict[str, Any] = {}
    error: str | None = None

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            nonlocal result, error
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)

            if "error" in params:
                error = params["error"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h2>Authentication failed.</h2>"
                    b"<p>You can close this window.</p></body></html>"
                )
                return

            code = params.get("code", [None])[0]
            returned_state = params.get("state", [None])[0]

            if not code:
                error = "No authorization code received"
                self.send_response(400)
                self.end_headers()
                return

            if returned_state and returned_state != state:
                error = "State mismatch"
                self.send_response(400)
                self.end_headers()
                return

            try:
                redirect_uri = f"{POE_REDIRECT_URI_BASE}:{server.server_port}"
                result = exchange_code_for_token(code, verifier, redirect_uri)
            except (httpx.HTTPError, ValueError, anyenv.JsonLoadError) as exc:
                error = str(exc)

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Authentication successful!</h2>"
                b"<p>You can close this window.</p></body></html>"
            )

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass  # Suppress HTTP logs

    server = HTTPServer(("127.0.0.1", POE_CALLBACK_PORT), CallbackHandler)
    port = server.server_port
    redirect_uri = f"{POE_REDIRECT_URI_BASE}:{port}"

    auth_url = build_authorization_url(verifier, challenge, redirect_uri, state)

    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    webbrowser.open(auth_url)
    print(f"Opened browser for Poe authentication (port {port})...")
    print("Waiting for callback...")

    thread.join(timeout=120)
    server.server_close()

    if error:
        msg = f"Poe authentication failed: {error}"
        raise ValueError(msg)

    if "api_key" not in result:
        msg = "No API key received from Poe"
        raise ValueError(msg)

    return str(result["api_key"])


def poe_auth_main() -> None:
    """Command-line entry point for Poe authentication."""
    parser = argparse.ArgumentParser(description="Authenticate with Poe.")
    parser.add_argument(
        "--token-path",
        type=Path,
        default=DEFAULT_TOKEN_PATH,
        help=f"Path to store credentials (default: {DEFAULT_TOKEN_PATH})",
    )
    parser.add_argument(
        "--logout",
        action="store_true",
        help="Remove stored credentials",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current authentication status",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="API key (skip OAuth, use directly)",
    )

    args = parser.parse_args()
    store = PoeTokenStore(path=args.token_path)

    if args.logout:
        store.clear()
        print("Logged out. Poe credentials removed.")
        return

    if args.status:
        api_key = store.load()
        if api_key is None:
            print("Not authenticated.")
            sys.exit(1)
        print(f"API key stored at: {args.token_path}")
        print(f"Key: {api_key[:8]}...{api_key[-4:]}")
        return

    try:
        api_key = authenticate_poe(manual_key=args.key)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)

    store.save(api_key)
    print(f"Poe API key saved to: {args.token_path}")
    print("You can now use Poe models.")


class PoeAuthBackend(ProviderAuthBackend):
    """Poe OAuth / API key auth backend."""

    def __init__(self) -> None:
        self._pending: dict[str, str] = {}  # state -> verifier

    @property
    def provider_id(self) -> str:
        return "poe"

    def methods(self) -> list[ProviderAuthMethod]:
        return [
            ProviderAuthMethod(type="oauth", label="Connect Poe (OAuth)"),
            ProviderAuthMethod(type="api", label="Connect Poe (API key)"),
        ]

    async def authorize(self, method: int = 0) -> ProviderAuthAuthorization:
        if method == 1:
            # API key method
            return ProviderAuthAuthorization(
                url="https://poe.com/api_key",
                instructions="Get your API key and enter it below",
                method="code",
            )
        import secrets as _secrets

        verifier, challenge = generate_pkce()
        state = _secrets.token_hex(16)
        # Use a fixed port for server-side flows
        redirect_uri = f"{POE_REDIRECT_URI_BASE}:3000/callback"
        auth_url = build_authorization_url(verifier, challenge, redirect_uri, state)
        self._pending[state] = verifier
        return ProviderAuthAuthorization(
            url=auth_url,
            instructions="Sign in with your Poe account",
            method="auto",
        )

    async def callback(
        self,
        *,
        code: str | None = None,
        device_code: str | None = None,
        verifier: str | None = None,
    ) -> bool:
        if not code:
            msg = "Missing code/key for Poe"
            raise ValueError(msg)
        # If verifier is provided, it's an OAuth flow
        if verifier:
            redirect_uri = f"{POE_REDIRECT_URI_BASE}:3000/callback"
            result = exchange_code_for_token(code, verifier, redirect_uri)
            PoeTokenStore().save(result["api_key"], result.get("expires_in"))
        else:
            # Direct API key
            if not validate_poe_key(code):
                msg = "Invalid Poe API key"
                raise ValueError(msg)
            PoeTokenStore().save(code)
        return True

    async def remove_credentials(self) -> bool:
        PoeTokenStore().clear()
        return True


if __name__ == "__main__":
    poe_auth_main()
