"""Background idle watcher — SOI Phase 1 (filing) then Phase 2 (Read refresh)."""

from __future__ import annotations

import threading
import time
from typing import Callable

from ollama.config import OllamaConfig
from ollama.session import ChatSession
from ollama.soi_log import SOILogger
from ollama.soi_worker import SOIWorker


class IdleSOIWatcher:
    """Daemon thread driven by OAC idle time (SOI runs do not reset OAC idle)."""

    def __init__(
        self,
        session: ChatSession,
        config: OllamaConfig,
        *,
        on_status: Callable[[str], None] | None = None,
        error_backoff_s: float = 90.0,
    ) -> None:
        self.session = session
        self.config = config
        self.on_status = on_status or (lambda _msg: None)
        self.error_backoff_s = error_backoff_s
        self._stop = threading.Event()
        self._kick = threading.Event()
        self._busy = threading.Lock()
        self._next_ok_at = 0.0
        self._thread = threading.Thread(target=self._loop, name="soi-idle", daemon=True)

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    @property
    def running(self) -> bool:
        return self.busy or self._kick.is_set()

    def start(self, *, force: bool = False) -> bool:
        if not force and not self.config.soi_enabled:
            return False
        if self._thread.is_alive():
            return True
        if self._thread.ident is not None:
            self._thread = threading.Thread(target=self._loop, name="soi-idle", daemon=True)
        self._thread.start()
        return True

    def request_run(self) -> dict[str, bool | str]:
        """Kick filing now (does not wait). Used by the :1111 Start SOI button."""
        if not self.start(force=True):
            return {"started": False, "reason": "watcher failed to start"}
        if self.busy:
            return {"started": False, "reason": "already running"}
        self._kick.set()
        return {"started": True}

    def stop(self) -> None:
        self._stop.set()
        self._kick.set()

    def _loop(self) -> None:
        logger = SOILogger(self.config.db_root, on_status=self.on_status)
        worker = SOIWorker(self.config, on_status=self.on_status, logger=logger)
        logger.log(
            "watcher_start",
            soi_idle_seconds=self.config.soi_idle_seconds,
            soi_read_refresh_idle_seconds=self.config.soi_read_refresh_idle_seconds,
            soi_timeout_s=self.config.soi_timeout_s,
        )
        while not self._stop.is_set():
            self._kick.wait(3.0)
            if self._stop.is_set():
                break
            kicked = self._kick.is_set()
            if kicked:
                self._kick.clear()
            if time.monotonic() < self._next_ok_at and not kicked:
                continue
            idle = self.session.idle_seconds()
            if not self._busy.acquire(blocking=False):
                if kicked:
                    self._kick.set()
                continue
            try:
                if kicked or (
                    worker.has_filing_work() and idle >= self.config.soi_idle_seconds
                ):
                    logger.log(
                        "idle_wake",
                        phase="filing",
                        idle_s=idle,
                        pending_changelog=len(worker.pending_changelog()),
                        pending_inbox=len(worker.pending_inbox()),
                    )
                    result = worker.run_filing()
                    if not result.get("ok"):
                        self._next_ok_at = time.monotonic() + self.error_backoff_s
                        logger.log(
                            "backoff",
                            seconds=self.error_backoff_s,
                            reason=result.get("error") or "filing failed",
                        )
                    continue

                if (
                    not worker.has_filing_work()
                    and worker.needs_read_refresh()
                    and idle >= self.config.soi_read_refresh_idle_seconds
                ):
                    logger.log(
                        "idle_wake",
                        phase="read_refresh",
                        idle_s=idle,
                        stale_count=len(worker.list_stale_read_paths()),
                    )
                    result = worker.run_read_refresh()
                    if not result.get("ok"):
                        self._next_ok_at = time.monotonic() + self.error_backoff_s
                        logger.log(
                            "backoff",
                            seconds=self.error_backoff_s,
                            reason=str(result.get("errors") or "read refresh failed"),
                        )
            except Exception as exc:  # noqa: BLE001 — keep chat loop alive
                self._next_ok_at = time.monotonic() + self.error_backoff_s
                logger.log("error", level="error", error=str(exc))
                logger.log("backoff", seconds=self.error_backoff_s, reason=str(exc))
            finally:
                self._busy.release()
