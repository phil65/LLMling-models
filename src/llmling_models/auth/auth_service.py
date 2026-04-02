from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self


if TYPE_CHECKING:
    from llmling_models.auth.base import ProviderAuthBackend
    from llmling_models.auth.models import AuthInfo, ProviderAuthAuthorization, ProviderAuthMethod


@dataclass
class ProviderAuthService:
    """Registry of provider auth backends.

    Mirrors opencode's ProviderAuth namespace — routes call service methods
    instead of containing provider-specific logic.
    """

    _backends: dict[str, ProviderAuthBackend] = field(default_factory=dict)

    def register(self, backend: ProviderAuthBackend) -> None:
        """Register an auth backend."""
        self._backends[backend.provider_id] = backend

    def get_backend(self, provider_id: str) -> ProviderAuthBackend:
        """Get backend by provider ID.

        Raises:
            KeyError: If provider_id is not registered.
        """
        try:
            return self._backends[provider_id]
        except KeyError:
            raise KeyError(f"Unknown provider: {provider_id}") from None

    def methods(self) -> dict[str, list[ProviderAuthMethod]]:
        """Return auth methods for all registered providers."""
        return {pid: backend.methods() for pid, backend in self._backends.items()}

    async def authorize(self, provider_id: str, method: int = 0) -> ProviderAuthAuthorization:
        """Start auth flow for a provider."""
        return await self.get_backend(provider_id).authorize(method)

    async def callback(
        self,
        provider_id: str,
        *,
        code: str | None = None,
        device_code: str | None = None,
        verifier: str | None = None,
    ) -> bool:
        """Handle auth callback for a provider."""
        return await self.get_backend(provider_id).callback(
            code=code, device_code=device_code, verifier=verifier
        )

    async def set_credentials(self, provider_id: str, info: AuthInfo) -> bool:
        """Set credentials for a provider."""
        return await self.get_backend(provider_id).set_credentials(info)

    async def remove_credentials(self, provider_id: str) -> bool:
        """Remove credentials for a provider."""
        return await self.get_backend(provider_id).remove_credentials()

    @classmethod
    def create_default(cls) -> Self:
        """Create auth service with built-in providers."""
        from llmling_models.auth.anthropic_auth import AnthropicAuthBackend
        from llmling_models.auth.antigravity_auth import AntigravityAuthBackend
        from llmling_models.auth.gemini_auth import GeminiAuthBackend
        from llmling_models.auth.github_auth import CopilotAuthBackend
        from llmling_models.auth.openai_codex_auth import OpenAICodexAuthBackend
        from llmling_models.auth.poe_auth import PoeAuthBackend
        from llmling_models.auth.zen_auth import ZenAuthBackend

        service = cls()
        service.register(AnthropicAuthBackend())
        service.register(CopilotAuthBackend())
        service.register(GeminiAuthBackend())
        service.register(AntigravityAuthBackend())
        service.register(OpenAICodexAuthBackend())
        service.register(PoeAuthBackend())
        service.register(ZenAuthBackend())
        return service
