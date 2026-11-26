"""Authentication helpers for various LLM providers.

Submodules:
- github_auth: GitHub Copilot authentication (with enterprise support)
- openai_codex_auth: OpenAI Codex (ChatGPT Plus/Pro) OAuth authentication
- zen_auth: OpenCode Zen API key authentication
- poe_auth: Poe OAuth / API key authentication
"""

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
