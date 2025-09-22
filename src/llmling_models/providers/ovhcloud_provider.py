"""OVHCloud provider implementation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from openai import AsyncOpenAI
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.providers import Provider

from llmling_models.log import get_logger


if TYPE_CHECKING:
    from httpx import AsyncClient as AsyncHTTPClient


logger = get_logger(__name__)


class OVHCloudProvider(Provider[AsyncOpenAI]):
    """Provider for OVHCloud AI endpoints."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        """Initialize provider for OVHCloud AI.

        Args:
            model_name: The OVHCloud model name to use. If provided, the base_url will be
                       constructed automatically unless explicitly provided.
            base_url: The base URL for the specific OVHCloud AI endpoint.
                     If not provided and model_name is given,
                     it will be constructed from the model name.
                     Otherwise, the OVH_CLOUD_BASE_URL env variable will be used.
            api_key: The API key to use for authentication.
                     If not provided,
                     the OVH_AI_ENDPOINTS_ACCESS_TOKEN env variable will be used.
            openai_client: An existing AsyncOpenAI client to use. If provided,
                           other parameters must be None.
            http_client: An existing AsyncHTTPClient to use for making HTTP requests.
        """
        api_key = api_key or os.environ.get("OVH_AI_ENDPOINTS_ACCESS_TOKEN")

        # Calculate base URL from model name if not provided
        if base_url is None and model_name is not None:
            endpoint_name = model_name.lower().replace(" ", "-")
            self._base_url = f"https://{endpoint_name}.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1"
        else:
            self._base_url = (
                base_url
                or os.environ.get(
                    "OVH_CLOUD_BASE_URL",
                    "https://mistral-7b-instruct-v0-3.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1",
                )
                or ""
            )
            assert self._base_url, "Base URL must be provided"

        if api_key is None and openai_client is None:
            msg = (
                "Set the OVH_AI_ENDPOINTS_ACCESS_TOKEN env variable or pass it via "
                "`OVHCloudProvider(api_key=...)` to use the OVHCloud provider."
            )
            raise ValueError(msg)

        if openai_client is not None:
            assert http_client is None, (
                "Cannot provide both `openai_client` and `http_client`"
            )
            assert api_key is None, "Cannot provide both `openai_client` and `api_key`"
            assert base_url is None, "Cannot provide both `openai_client` and `base_url`"
            assert model_name is None, (
                "Cannot provide both `openai_client` and `model_name`"
            )
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(
                base_url=self._base_url,
                api_key=api_key,
                http_client=http_client,
            )
        else:
            self._client = AsyncOpenAI(
                base_url=self._base_url,
                api_key=api_key,
                http_client=cached_async_http_client(),
            )

        # Store model name for reference
        self._model_name = model_name

    @property
    def name(self) -> str:
        """The provider name."""
        return "ovhcloud"

    @property
    def base_url(self) -> str:
        """The base URL for the provider API."""
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        """Get a client configured for OVHCloud."""
        return self._client


if __name__ == "__main__":
    import asyncio

    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIResponsesModel

    async def main():
        # Example using model name for automatic endpoint construction
        provider = OVHCloudProvider(model_name="mistral-7b-instruct-v0-3")
        model = OpenAIResponsesModel("Mistral-7B-Instruct-v0.3", provider=provider)
        agent = Agent(model=model)
        result = await agent.run("Explain gravity for a 6 years old")
        print(result.output)

    asyncio.run(main())
