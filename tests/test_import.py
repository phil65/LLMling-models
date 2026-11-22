"""Smoke test: the package imports and reports its version."""

from __future__ import annotations

import llmling_models


def test_package_imports() -> None:
    """The package can be imported and exposes a version string."""
    assert isinstance(llmling_models.__version__, str)
