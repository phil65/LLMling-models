from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _dummy_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a placeholder OpenAI key when none is set.

    The tests only construct models and never call the API, but the OpenAI SDK and
    pydantic-ai refuse to build a client without a key. CI runs without secrets
    (e.g. Dependabot PRs) would otherwise fail.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
