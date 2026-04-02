"""Agent and command models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ProviderAuthMethod(BaseModel):
    """Authentication method for a provider."""

    type: Literal["oauth", "api"]
    """Auth type."""

    label: str
    """Human-readable label for the auth method."""


class ProviderAuthAuthorization(BaseModel):
    """Response from starting a provider OAuth flow."""

    url: str
    """URL to open in browser for authorization."""

    method: Literal["auto", "code"]
    """Authorization method."""

    instructions: str
    """Instructions to display to the user."""


class OAuthAuthInfo(BaseModel):
    """OAuth authentication credentials."""

    type: Literal["oauth"]
    """Auth type discriminator."""

    refresh: str
    """Refresh token."""

    access: str
    """Access token."""

    expires: int
    """Token expiry timestamp."""

    account_id: str | None = None
    """Optional account identifier."""

    enterprise_url: str | None = None
    """Optional enterprise URL."""


class ApiAuthInfo(BaseModel):
    """API key authentication credentials."""

    type: Literal["api"]
    """Auth type discriminator."""

    key: str
    """API key."""


class WellKnownAuthInfo(BaseModel):
    """Well-known authentication credentials."""

    type: Literal["wellknown"]
    """Auth type discriminator."""

    key: str
    """Key identifier."""

    token: str
    """Authentication token."""


AuthInfo = OAuthAuthInfo | ApiAuthInfo | WellKnownAuthInfo
"""Authentication credentials (discriminated union on 'type')."""
