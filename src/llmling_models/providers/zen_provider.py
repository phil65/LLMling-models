"""OpenCode Zen provider implementation."""

from __future__ import annotations

import os
import secrets
from typing import TYPE_CHECKING, Any

from httpx import AsyncClient as AsyncHTTPClient
from openai import AsyncOpenAI
from pydantic_ai.providers import Provider

from llmling_models.log import get_logger


if TYPE_CHECKING:
    from httpx import Request, Response


logger = get_logger(__name__)

ZEN_BASE_URL = "https://opencode.ai/zen/v1"


class ZenHTTPClient(AsyncHTTPClient):
    """Custom client that adds opencode headers before each request."""

    def __init__(self, session_id: str, project_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._session_id = session_id
        self._project_id = project_id

    async def send(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        request.headers["User-Agent"] = "opencode/latest/1.3.15/cli"
        request.headers["x-opencode-client"] = "cli"
        request.headers["x-opencode-session"] = self._session_id
        request.headers["x-opencode-project"] = self._project_id
        request.headers["x-opencode-request"] = secrets.token_hex(14)
        return await super().send(request, *args, **kwargs)


class ZenProvider(Provider[AsyncOpenAI]):
    """Provider for OpenCode Zen API."""

    def __init__(self, api_key: str | None = None) -> None:
        api_key = api_key or os.environ.get("ZEN_API_KEY") or os.environ.get("ZENMUX_API_KEY")
        if not api_key:
            msg = (
                "Set the `ZEN_API_KEY` environment variable or pass it via "
                "`ZenProvider(api_key=...)` to use the Zen provider."
            )
            raise ValueError(msg)
        http_client = ZenHTTPClient(
            session_id=secrets.token_hex(14),
            project_id=secrets.token_hex(14),
            timeout=60.0,
        )
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=ZEN_BASE_URL,
            http_client=http_client,
        )

    @property
    def name(self) -> str:
        return "zen"

    @property
    def base_url(self) -> str:
        return ZEN_BASE_URL

    @property
    def client(self) -> AsyncOpenAI:
        return self._client
