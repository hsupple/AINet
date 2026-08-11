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
        self._busy = threading.Lock()
        self._next_ok_at = 0.0
        self._thread = threading.Thread(target=self._loop, name="soi-idle", daemon=True)

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    def start(self) -> None:
        if not self.config.soi_enabled:
            return
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        logger = SOILogger(self.config.db_root, on_status=self.on_status)
        worker = SOIWorker(self.config, on_status=self.on_status, logger=logger)
        logger.log(
            "watcher_start",
            soi_idle_seconds=self.config.soi_idle_seconds,
            soi_read_refresh_idle_seconds=self.config.soi_read_refresh_idle_seconds,
            soi_timeout_s=self.config.soi_timeout_s,
        )
        while not self._stop.wait(3.0):
            if time.monotonic() < self._next_ok_at:
                continue
            idle = self.session.idle_seconds()
            if not self._busy.acquire(blocking=False):
                continue
            try:
                # Phase 1: file changelog/inbox after short idle
                if worker.has_filing_work() and idle >= self.config.soi_idle_seconds:
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

                # Phase 2: refresh Read.json files after long idle + filing clear
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
