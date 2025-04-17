"""GitHub Copilot authentication helper."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import NamedTuple

import httpx

from llmling_models.log import get_logger


logger = get_logger(__name__)


class CopilotAuthResult(NamedTuple):
    """Result of Copilot authentication."""

    token: str
    token_type: str
    scope: str
    refresh_token: str | None = None


def authenticate_copilot(verbose: bool = True) -> CopilotAuthResult:
    """Authenticate with GitHub Copilot and return the access token.

    Args:
        verbose: Whether to print authentication status messages

    Returns:
        The authentication result containing the access token
    """
    # Step 1: Get device code
    if verbose:
        print("Requesting GitHub device code...")

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            "https://github.com/login/device/code",
            headers={
                "accept": "application/json",
                "editor-version": "Neovim/0.6.1",
                "editor-plugin-version": "copilot.vim/1.16.0",
                "content-type": "application/json",
                "user-agent": "GithubCopilot/1.155.0",
                "accept-encoding": "gzip,deflate,br",
            },
            content='{"client_id":"Iv1.b507a08c87ecfe98","scope":"read:user"}',
        )
        resp.raise_for_status()

        auth_data = resp.json()
        device_code = auth_data["device_code"]
        user_code = auth_data["user_code"]
        verification_uri = auth_data["verification_uri"]

        # Step 2: Prompt user to authenticate
        if verbose:
            print()
            print("To authenticate with GitHub Copilot, please:")
            print(f"1. Visit:  {verification_uri}")
            print(f"2. Enter code:  {user_code}")
            print("\nWaiting for authentication...")

        # Step 3: Poll for token
        interval = auth_data.get("interval", 5)
        while True:
            time.sleep(interval)

            try:
                resp = client.post(
                    "https://github.com/login/oauth/access_token",
                    headers={
                        "accept": "application/json",
                        "editor-version": "Neovim/0.6.1",
                        "editor-plugin-version": "copilot.vim/1.16.0",
                        "content-type": "application/json",
                        "user-agent": "GithubCopilot/1.155.0",
                        "accept-encoding": "gzip,deflate,br",
                    },
                    content=(
                        f'{{"client_id":"Iv1.b507a08c87ecfe98","device_code":"{device_code}",'
                        f'"grant_type":"urn:ietf:params:oauth:grant-type:device_code"}}'
                    ),
                )

                resp.raise_for_status()
                resp_data = resp.json()

                # Check for errors
                if "error" in resp_data:
                    if resp_data["error"] == "authorization_pending":
                        if verbose:
                            print(".", end="", flush=True)
                        continue

                    error_msg = resp_data.get("error_description", resp_data["error"])
                    if verbose:
                        print(f"\nAuthentication failed: {error_msg}")
                    msg = f"Authentication failed: {error_msg}"
                    raise RuntimeError(msg)

                # Extract token
                access_token = resp_data.get("access_token")
                if access_token:
                    if verbose:
                        print("\nAuthentication successful!")
                    return CopilotAuthResult(
                        token=access_token,
                        token_type=resp_data.get("token_type", "bearer"),
                        scope=resp_data.get("scope", ""),
                        refresh_token=resp_data.get("refresh_token"),
                    )

            except httpx.HTTPError as e:
                if verbose:
                    print(f"\nError during authentication: {e}")
                msg = f"Error during authentication: {e}"
                raise RuntimeError(msg) from e


def save_token(token: str, env_var: str = "GITHUB_COPILOT_TOKEN") -> None:
    """Save the token to a file and print instructions.

    Args:
        token: The token to save
        env_var: The environment variable name to suggest
    """
    env_file = ".env"
    if (path := Path(env_file)).exists():
        with path.open("a") as f:
            f.write(f"\n# GitHub Copilot API Token\n{env_var}={token}\n")
        print(f"Token saved to {env_file}")

    # Print instructions
    print(f"\nTo use this token, set the environment variable {env_var}:")
    print(f"  export {env_var}={token}")
    print("\nOr add it to your .env file:")
    print(f"  {env_var}={token}")


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
        "--save",
        action="store_true",
        help="Save token to .env file if it exists",
    )
    parser.add_argument(
        "--env-var",
        default="GITHUB_COPILOT_TOKEN",
        help="Environment variable name to use (default: GITHUB_COPILOT_TOKEN)",
    )

    args = parser.parse_args()

    try:
        result = authenticate_copilot(verbose=not args.silent)
        print(f"\nToken: {result.token}")

        if args.save:
            save_token(result.token, args.env_var)

    except Exception:
        logger.exception("Authentication failed")
        sys.exit(1)


if __name__ == "__main__":
    copilot_auth_main()
