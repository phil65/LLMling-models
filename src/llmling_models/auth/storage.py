"""Private, atomic token files and a cross-process lock for token refreshes."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Generator


def write_private_json(path: Path, data: Any) -> None:
    """Write JSON to `path`, readable by the owner only, replacing it atomically.

    The temp file is created with mode 0600 (mkstemp), so the secret is never in a
    world-readable file, and a concurrent reader sees the old or the new file, never a
    truncated one.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@contextmanager
def file_lock(path: Path) -> Generator[None]:
    """Hold an exclusive lock on `<path>.lock` (blocking; no-op where flock is missing)."""
    try:
        import fcntl
    except ImportError:  # Windows
        yield
        return
    lock_path = path.with_name(path.name + ".lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # releases the lock
