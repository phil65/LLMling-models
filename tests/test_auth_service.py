"""Tests for the provider auth backend registry."""

from llmling_models.auth import ProviderAuthService


def test_default_backends():
    service = ProviderAuthService.create_default()
    assert set(service.methods()) == {"copilot", "openai-codex", "poe", "zen"}
