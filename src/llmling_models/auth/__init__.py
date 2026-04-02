"""Authentication helpers for various LLM providers.

Submodules:
- anthropic_auth: Anthropic Claude Max/Pro OAuth authentication
- github_auth: GitHub Copilot authentication (with enterprise support)
- gemini_auth: Gemini CLI (Google Cloud Code Assist) OAuth authentication
- antigravity_auth: Antigravity (Gemini 3, Claude, GPT-OSS via Google Cloud) OAuth
- openai_codex_auth: OpenAI Codex (ChatGPT Plus/Pro) OAuth authentication
- zen_auth: OpenCode Zen API key authentication
- poe_auth: Poe OAuth / API key authentication
"""

from __future__ import annotations

from .models import (
    OAuthAuthInfo,
    ApiAuthInfo,
    WellKnownAuthInfo,
    AuthInfo,
    ProviderAuthAuthorization,
)

from .auth_service import ProviderAuthService


__all__ = [
    "ApiAuthInfo",
    "AuthInfo",
    "OAuthAuthInfo",
    "ProviderAuthAuthorization",
    "ProviderAuthService",
    "WellKnownAuthInfo",
]
