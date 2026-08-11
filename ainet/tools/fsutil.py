"""Cross-platform filesystem helpers (Windows-safe atomic writes)."""

from __future__ import annotations

import os
import stat
import tempfile
import time
from pathlib import Path

_WIN_RETRY_ERRNOS = {13, 11}
_WIN_RETRY_WINERRORS = {5, 32, 33}


def _is_lock_error(exc: OSError) -> bool:
    winerr = getattr(exc, "winerror", None)
    if winerr in _WIN_RETRY_WINERRORS:
        return True
    if exc.errno in _WIN_RETRY_ERRNOS:
        return True
    return isinstance(exc, PermissionError)


def _replace_with_retry(tmp_path: Path, dest: Path, *, attempts: int = 16) -> None:
    last: OSError | None = None
    for i in range(attempts):
        try:
            os.replace(tmp_path, dest)
            return
        except OSError as exc:
            last = exc
            if not _is_lock_error(exc):
                raise
            try:
                dest.chmod(stat.S_IWRITE | stat.S_IREAD)
            except OSError:
                pass
            time.sleep(0.04 * (i + 1))
    try:
        payload = tmp_path.read_bytes()
        with dest.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.unlink(missing_ok=True)
        return
    except OSError:
        if last is not None:
            raise last
        raise


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes to path via temp file + os.replace (works when target exists on Windows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".ainet-",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))
