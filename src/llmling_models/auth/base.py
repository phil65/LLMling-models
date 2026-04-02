"""Provider authentication service.

Composable auth backend system matching the opencode plugin auth pattern.
Each provider registers a backend that handles its specific auth flow
(OAuth PKCE, device code, API key, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from llmling_models.auth.models import AuthInfo, ProviderAuthAuthorization, ProviderAuthMethod


class ProviderAuthBackend(ABC):
    """Protocol for a provider-specific auth backend."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique provider identifier."""
        ...

    @abstractmethod
    def methods(self) -> list[ProviderAuthMethod]:
        """Return available auth methods for this provider."""
        ...

    @abstractmethod
    async def authorize(self, method: int = 0) -> ProviderAuthAuthorization:
        """Start an authorization flow.

        Args:
            method: Index into the methods list.

        Returns:
            Authorization info with URL and instructions.
        """
        ...

    @abstractmethod
    async def callback(
        self,
        *,
        code: str | None = None,
        device_code: str | None = None,
        verifier: str | None = None,
    ) -> bool:
        """Handle the auth callback / code exchange.

        Returns:
            True if auth succeeded.

        Raises:
            ValueError: If required parameters are missing or exchange fails.
        """
        ...

    async def set_credentials(self, info: AuthInfo) -> bool:
        """Store credentials for this provider.

        Default implementation is a no-op. Override for providers that
        support direct credential setting (e.g. API key or token import).
        """
        return False

    async def remove_credentials(self) -> bool:
        """Remove stored credentials for this provider.

        Default implementation is a no-op.
        """
        return False
