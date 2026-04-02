"""GitHub Copilot authentication helper.

Supports both github.com and GitHub Enterprise via device code flow.
After login, exchanges the GitHub access token for a Copilot API token.

Based on the pi-mono implementation by badlogic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import NamedTuple

import anyenv
import httpx

from llmling_models.log import get_logger


logger = get_logger(__name__)

CLIENT_ID = "Iv1.b507a08c87ecfe98"

COPILOT_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}

# Polling interval multipliers for device code flow
_INITIAL_POLL_MULTIPLIER = 1.2
_SLOW_DOWN_POLL_MULTIPLIER = 1.4

# Default token storage location
DEFAULT_TOKEN_PATH = Path.home() / ".config" / "llmling-models" / "copilot_oauth.json"


class CopilotAuthResult(NamedTuple):
    """Result of Copilot authentication."""

    token: str
    token_type: str
    scope: str
    refresh_token: str | None = None


def _normalize_domain(input_str: str) -> str | None:
    """Normalize a domain input to just the hostname."""
    trimmed = input_str.strip()
    if not trimmed:
        return None
    try:
        from urllib.parse import urlparse

        url = trimmed if "://" in trimmed else f"https://{trimmed}"
        return urlparse(url).hostname
    except Exception:  # noqa: BLE001
        return None


def _get_urls(domain: str) -> dict[str, str]:
    """Get OAuth URLs for a GitHub domain."""
    return {
        "device_code": f"https://{domain}/login/device/code",
        "access_token": f"https://{domain}/login/oauth/access_token",
        "copilot_token": f"https://api.{domain}/copilot_internal/v2/token",
    }


def _get_base_url_from_token(token: str) -> str | None:
    """Extract API base URL from Copilot token's proxy-ep field.

    Token format: tid=...;exp=...;proxy-ep=proxy.individual.githubcopilot.com;...
    Returns URL like https://api.individual.githubcopilot.com
    """
    import re

    match = re.search(r"proxy-ep=([^;]+)", token)
    if not match:
        return None
    proxy_host = match.group(1)
    api_host = proxy_host.replace("proxy.", "api.", 1)
    return f"https://{api_host}"


def get_copilot_base_url(
    token: str | None = None,
    enterprise_domain: str | None = None,
) -> str:
    """Get the Copilot API base URL.

    Prefers extracting from the token's proxy-ep field, falls back to
    constructing from enterprise domain or default.
    """
    if token:
        url = _get_base_url_from_token(token)
        if url:
            return url
    if enterprise_domain:
        return f"https://copilot-api.{enterprise_domain}"
    return "https://api.individual.githubcopilot.com"


def _start_device_flow(client: httpx.Client, domain: str) -> dict[str, str | int]:
    """Start the device code flow and return device code info."""
    urls = _get_urls(domain)
    resp = client.post(
        urls["device_code"],
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "GitHubCopilotChat/0.35.0",
        },
        data={"client_id": CLIENT_ID, "scope": "read:user"},
    )
    resp.raise_for_status()
    return anyenv.load_json(resp.text, return_type=dict)


def _poll_for_github_token(
    client: httpx.Client,
    domain: str,
    device_code: str,
    interval_seconds: float,
    expires_in: float,
) -> str:
    """Poll for GitHub access token after device code flow."""
    urls = _get_urls(domain)
    deadline = time.time() + expires_in
    interval_ms = max(1000, interval_seconds * 1000)
    multiplier = _INITIAL_POLL_MULTIPLIER

    while time.time() < deadline:
        remaining_ms = (deadline - time.time()) * 1000
        wait_ms = min(interval_ms * multiplier, remaining_ms)
        time.sleep(wait_ms / 1000)

        resp = client.post(
            urls["access_token"],
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "GitHubCopilotChat/0.35.0",
            },
            data={
                "client_id": CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
        )
        resp.raise_for_status()
        data = anyenv.load_json(resp.text, return_type=dict)

        if "access_token" in data:
            return str(data["access_token"])

        error = data.get("error", "")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            new_interval = data.get("interval")
            if isinstance(new_interval, int) and new_interval > 0:
                interval_ms = new_interval * 1000
            else:
                interval_ms = max(1000, interval_ms + 5000)
            multiplier = _SLOW_DOWN_POLL_MULTIPLIER
            continue

        description = data.get("error_description", error)
        msg = f"Device flow failed: {description}"
        raise RuntimeError(msg)

    msg = "Device flow timed out"
    raise RuntimeError(msg)


def _get_copilot_token(
    client: httpx.Client,
    github_token: str,
    domain: str,
) -> dict[str, str | int]:
    """Exchange GitHub access token for a Copilot API token."""
    urls = _get_urls(domain)
    resp = client.get(
        urls["copilot_token"],
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {github_token}",
            **COPILOT_HEADERS,
        },
    )
    resp.raise_for_status()
    return anyenv.load_json(resp.text, return_type=dict)


def authenticate_copilot(
    verbose: bool = True,
    enterprise_domain: str | None = None,
) -> dict[str, str | float | None]:
    """Authenticate with GitHub Copilot via device code flow.

    Args:
        verbose: Whether to print authentication status messages
        enterprise_domain: GitHub Enterprise domain (None for github.com)

    Returns:
        Dict with keys: github_token, copilot_token, base_url, expires_at,
        enterprise_domain
    """
    domain = enterprise_domain or "github.com"

    with httpx.Client(timeout=30.0) as client:
        # Step 1: Start device flow
        if verbose:
            print("Requesting GitHub device code...")
        device_data = _start_device_flow(client, domain)

        user_code = device_data["user_code"]
        verification_uri = device_data["verification_uri"]

        if verbose:
            print()
            print("To authenticate with GitHub Copilot, please:")
            print(f"1. Visit:  {verification_uri}")
            print(f"2. Enter code:  {user_code}")
            print("\nWaiting for authentication...")

        # Step 2: Poll for GitHub access token
        github_token = _poll_for_github_token(
            client,
            domain,
            str(device_data["device_code"]),
            int(device_data.get("interval", 5)),
            int(device_data.get("expires_in", 900)),
        )

        if verbose:
            print("\nGitHub authentication successful!")
            print("Exchanging for Copilot API token...")

        # Step 3: Exchange for Copilot token
        copilot_data = _get_copilot_token(client, github_token, domain)
        copilot_token = str(copilot_data["token"])
        expires_at = copilot_data["expires_at"]

        base_url = get_copilot_base_url(copilot_token, enterprise_domain)

        if verbose:
            print(f"Copilot API base URL: {base_url}")
            print("Authentication complete!")

    return {
        "github_token": github_token,
        "copilot_token": copilot_token,
        "base_url": base_url,
        "expires_at": float(expires_at),
        "enterprise_domain": enterprise_domain,
    }


def refresh_copilot_token(
    github_token: str,
    enterprise_domain: str | None = None,
) -> dict[str, str | float | None]:
    """Refresh the Copilot API token using the stored GitHub token.

    Args:
        github_token: The GitHub OAuth access token
        enterprise_domain: GitHub Enterprise domain (None for github.com)

    Returns:
        Dict with copilot_token, base_url, expires_at
    """
    domain = enterprise_domain or "github.com"

    with httpx.Client(timeout=30.0) as client:
        copilot_data = _get_copilot_token(client, github_token, domain)

    copilot_token = str(copilot_data["token"])
    expires_at = copilot_data["expires_at"]
    base_url = get_copilot_base_url(copilot_token, enterprise_domain)

    return {
        "copilot_token": copilot_token,
        "base_url": base_url,
        "expires_at": float(expires_at),
    }


class CopilotTokenStore:
    """File-based token storage for Copilot OAuth."""

    def __init__(self, path: Path = DEFAULT_TOKEN_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, str | float | None] | None:
        """Load stored credentials."""
        if not self.path.exists():
            return None
        try:
            return anyenv.load_json(self.path.read_text(), return_type=dict)
        except (anyenv.JsonLoadError, KeyError) as e:
            logger.warning("Failed to load copilot token from %s: %s", self.path, e)
            return None

    def save(self, data: dict[str, str | float | None]) -> None:
        """Save credentials."""
        self.path.write_text(json.dumps(data, indent=2))
        self.path.chmod(0o600)

    def clear(self) -> None:
        """Remove stored credentials."""
        if self.path.exists():
            self.path.unlink()


def get_or_refresh_copilot_token(
    store: CopilotTokenStore | None = None,
) -> dict[str, str | float | None]:
    """Get a valid Copilot token, refreshing if necessary.

    The GitHub OAuth token doesn't expire, but the Copilot API token does.
    This refreshes the Copilot token using the stored GitHub token.
    """
    if store is None:
        store = CopilotTokenStore()

    data = store.load()
    if data is None:
        msg = "No Copilot credentials found. Run 'llmling-models copilot-auth' to authenticate."
        raise RuntimeError(msg)

    # Check if Copilot token is expired
    expires_at = data.get("expires_at", 0)
    if isinstance(expires_at, (int, float)) and time.time() < expires_at - 300:
        return data

    # Refresh using stored GitHub token
    github_token = data.get("github_token")
    if not github_token:
        msg = "Stored credentials missing GitHub token. Re-authenticate."
        raise RuntimeError(msg)

    logger.info("Copilot token expired, refreshing...")
    enterprise = data.get("enterprise_domain")
    refreshed = refresh_copilot_token(
        str(github_token),
        str(enterprise) if enterprise else None,
    )

    # Merge refreshed data back
    data["copilot_token"] = refreshed["copilot_token"]
    data["base_url"] = refreshed["base_url"]
    data["expires_at"] = refreshed["expires_at"]
    store.save(data)

    return data


def copilot_auth_main() -> None:
    """Command-line entry point for Copilot authentication."""
    parser = argparse.ArgumentParser(
        description="Authenticate with GitHub Copilot and get API token."
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Suppress status messages",
    )
    parser.add_argument(
        "--enterprise",
        default=None,
        help="GitHub Enterprise domain (e.g. company.ghe.com)",
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
    store = CopilotTokenStore(path=args.token_path)

    if args.logout:
        store.clear()
        print("Logged out. Token removed.")
        return

    if args.status:
        data = store.load()
        if data is None:
            print("Not authenticated.")
            sys.exit(1)
        expires_at = data.get("expires_at", 0)
        if isinstance(expires_at, (int, float)) and time.time() < expires_at:
            remaining = expires_at - time.time()
            hours = int(remaining // 3600)
            minutes = int((remaining % 3600) // 60)
            print(f"Authenticated. Copilot token expires in {hours}h {minutes}m.")
        else:
            print("Copilot token expired (will refresh automatically on use).")
        print(f"Base URL: {data.get('base_url', 'unknown')}")
        enterprise = data.get("enterprise_domain")
        if enterprise:
            print(f"Enterprise: {enterprise}")
        print(f"Token path: {args.token_path}")
        return

    # Normalize enterprise domain
    enterprise_domain = None
    if args.enterprise:
        enterprise_domain = _normalize_domain(args.enterprise)
        if not enterprise_domain:
            print(f"Invalid enterprise domain: {args.enterprise}", file=sys.stderr)
            sys.exit(1)

    try:
        result = authenticate_copilot(
            verbose=not args.silent,
            enterprise_domain=enterprise_domain,
        )
        store.save(result)
        print(f"\nToken saved to: {args.token_path}")
        print("You can now use GitHub Copilot models.")

    except Exception:
        logger.exception("Authentication failed")
        sys.exit(1)


if __name__ == "__main__":
    copilot_auth_main()
