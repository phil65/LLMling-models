from __future__ import annotations

from typing import Literal

from pydantic_ai.models import KnownModelName, Model


AllModels = Literal[
    "delegation",
    "cost_optimized",
    "token_optimized",
    "fallback",
    "input",
    "import",
    "remote_model",
    "remote_input",
    "llm",
    "aisuite",
    "augmented",
    "user_select",
]


type ModelInput = str | KnownModelName | Model
"""Type for internal model handling (after validation)."""
