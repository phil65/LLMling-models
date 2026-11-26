import os

import pytest


# Provider keys needed to *construct* models in tests (they never call the API).
_PLACEHOLDER_KEYS = ("OPENAI_API_KEY", "OPENROUTER_API_KEY")


@pytest.fixture(autouse=True)
def _placeholder_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide placeholder provider keys when none are set.

    The tests only construct models and never call the API, but the provider SDKs
    and pydantic-ai refuse to build a client without a key. CI runs without secrets
    (e.g. Dependabot PRs) would otherwise fail.
    """
    for name in _PLACEHOLDER_KEYS:
        if not os.environ.get(name):
            monkeypatch.setenv(name, "test-key-not-used")
