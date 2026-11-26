from __future__ import annotations

import asyncio
import stat
import sys
import time
from typing import TYPE_CHECKING, Any

import pytest

from llmling_models.auth import openai_codex_auth as codex
from llmling_models.auth.github_auth import CopilotTokenStore
from llmling_models.auth.openai_codex_auth import OpenAICodexToken, OpenAICodexTokenStore
from llmling_models.auth.poe_auth import PoeTokenStore
from llmling_models.auth.zen_auth import ZenTokenStore


if TYPE_CHECKING:
    from pathlib import Path


PRIVATE = 0o600


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _token(n: int, expired: bool = False) -> OpenAICodexToken:
    expires = time.time() - 10 if expired else time.time() + 3600
    return OpenAICodexToken(f"access{n}", f"refresh{n}", expires, "acct")


posix_modes = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")


@posix_modes
def test_token_files_are_private_from_the_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The secret goes into a private temp file that replaces the target atomically."""
    from llmling_models.auth import storage

    modes: list[int] = []
    real_replace = storage.Path.replace

    def spy_replace(src: Path, dst: Path) -> Path:
        modes.append(_mode(src))
        return real_replace(src, dst)

    monkeypatch.setattr(storage.Path, "replace", spy_replace)
    path = tmp_path / "codex.json"
    OpenAICodexTokenStore(path=path).save(_token(1))
    assert modes == [PRIVATE]
    assert _mode(path) == PRIVATE
    assert sorted(p.name for p in tmp_path.iterdir() if not p.name.endswith(".lock")) == [
        "codex.json"
    ]


@posix_modes
@pytest.mark.parametrize(
    ("store_cls", "args"),
    [
        (PoeTokenStore, ("key",)),
        (ZenTokenStore, ("key",)),
        (CopilotTokenStore, ({"github_token": "x"},)),
    ],
)
def test_other_token_stores_write_private_files(
    tmp_path: Path, store_cls: type[Any], args: tuple[Any, ...]
) -> None:
    path = tmp_path / "t.json"
    store_cls(path=path).save(*args)
    assert _mode(path) == PRIVATE


async def test_concurrent_refreshes_run_once_and_pick_up_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def fake_refresh(refresh_token: str) -> OpenAICodexToken:
        calls.append(refresh_token)
        await asyncio.sleep(0.05)
        return _token(2)

    monkeypatch.setattr(codex, "refresh_access_token_async", fake_refresh)
    path = tmp_path / "codex.json"
    OpenAICodexTokenStore(path=path).save(_token(1, expired=True))
    store = OpenAICodexTokenStore(path=path)
    other = OpenAICodexTokenStore(path=path)  # another process, same file
    other.load()  # caches the expired token
    tokens = await asyncio.gather(*(codex.get_or_refresh_token_async(store) for _ in range(3)))
    assert calls == ["refresh1"]
    assert {t.access_token for t in tokens} == {"access2"}
    # The other "process" reads the refreshed file instead of refreshing again.
    assert (await codex.get_or_refresh_token_async(other)).access_token == "access2"
    assert calls == ["refresh1"]


def test_sync_refresh_picks_up_a_token_refreshed_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_refresh(refresh_token: str) -> OpenAICodexToken:
        calls.append(refresh_token)
        return _token(2)

    monkeypatch.setattr(codex, "refresh_access_token", fake_refresh)
    path = tmp_path / "codex.json"
    store = OpenAICodexTokenStore(path=path)
    store.save(_token(1, expired=True))
    OpenAICodexTokenStore(path=path).save(_token(3))  # refreshed by another process
    assert codex.get_or_refresh_token(store).access_token == "access3"
    assert calls == []
