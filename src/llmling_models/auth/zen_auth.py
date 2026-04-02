"""OpenCode Zen API key authentication.

Simple API key auth for OpenCode Zen, which provides access to
multiple coding models through a single API key.

Get your API key at: https://opencode.ai/zen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import anyenv
import httpx

from llmling_models.log import get_logger


logger = get_logger(__name__)

ZEN_API_URL = "https://opencode.ai/zen/v1"
DEFAULT_TOKEN_PATH = Path.home() / ".config" / "llmling-models" / "zen_auth.json"


class ZenTokenStore:
    """File-based storage for Zen API key."""

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
            logger.warning("Failed to load Zen API key from %s: %s", self.path, e)
            return None

    def save(self, api_key: str) -> None:
        """Save API key."""
        self.path.write_text(json.dumps({"api_key": api_key}, indent=2))
        self.path.chmod(0o600)

    def clear(self) -> None:
        """Remove stored API key."""
        if self.path.exists():
            self.path.unlink()


def validate_zen_key(api_key: str) -> bool:
    """Validate a Zen API key by fetching models."""
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(
            f"{ZEN_API_URL}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        return resp.is_success


def zen_auth_main() -> None:
    """Command-line entry point for Zen API key setup."""
    parser = argparse.ArgumentParser(description="Set up OpenCode Zen API key.")
    parser.add_argument(
        "--token-path",
        type=Path,
        default=DEFAULT_TOKEN_PATH,
        help=f"Path to store key (default: {DEFAULT_TOKEN_PATH})",
    )
    parser.add_argument(
        "--logout",
        action="store_true",
        help="Remove stored API key",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current authentication status",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="API key (if not provided, prompts interactively)",
    )

    args = parser.parse_args()
    store = ZenTokenStore(path=args.token_path)

    if args.logout:
        store.clear()
        print("Logged out. API key removed.")
        return

    if args.status:
        api_key = store.load()
        if api_key is None:
            print("Not authenticated.")
            sys.exit(1)
        print(f"API key stored at: {args.token_path}")
        print(f"Key: {api_key[:8]}...{api_key[-4:]}")
        return

    api_key = args.key
    if not api_key:
        print("Get your API key at: https://opencode.ai/zen")
        try:
            api_key = input("Enter your OpenCode Zen API key: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            sys.exit(1)

    if not api_key:
        print("No API key provided.", file=sys.stderr)
        sys.exit(1)

    print("Validating API key...")
    if not validate_zen_key(api_key):
        print("Invalid API key.", file=sys.stderr)
        sys.exit(1)

    store.save(api_key)
    print(f"API key saved to: {args.token_path}")
    print("You can now use OpenCode Zen models.")


if __name__ == "__main__":
    zen_auth_main()
