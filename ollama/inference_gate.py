"""Serialize Ollama inference so OAC and SOI do not pile onto one GPU request."""

from __future__ import annotations

import threading
import time


class InferenceGate:
    """Non-reentrant lock with a snapshot so the UI can show who is blocking chat."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ticket = 0
        self.holder: str | None = None
        self.held_since: float = 0.0

    def acquire(self, timeout: float | None = None, *, holder: str = "unknown") -> int:
        """Return a ticket > 0 on success, or 0 on timeout."""
        ok = self._lock.acquire(timeout=timeout) if timeout is not None else self._lock.acquire()
        if not ok:
            return 0
        self._ticket += 1
        self.holder = holder
        self.held_since = time.monotonic()
        return self._ticket

    def release(self, ticket: int | None = None) -> None:
        if ticket is not None and ticket != self._ticket:
            return
        self.holder = None
        self.held_since = 0.0
        try:
            self._lock.release()
        except RuntimeError:
            pass

    def force_reset(self) -> None:
        """Drop a stuck hold so Reset AI can start a new turn (any thread)."""
        self._ticket += 1
        self.holder = None
        self.held_since = 0.0
        try:
            if self._lock.locked():
                self._lock.release()
        except RuntimeError:
            pass

    def locked(self) -> bool:
        return self._lock.locked()

    def snapshot(self) -> dict[str, object]:
        locked = self._lock.locked()
        held_s = 0.0
        if locked and self.held_since:
            held_s = max(0.0, time.monotonic() - self.held_since)
        return {
            "locked": locked,
            "holder": self.holder if locked else None,
            "held_s": round(held_s, 1),
        }

    def __enter__(self) -> "InferenceGate":
        self.acquire(holder="unknown")
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.release()
        return False


INFERENCE_GATE = InferenceGate()
