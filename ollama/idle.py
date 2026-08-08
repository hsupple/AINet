"""Background idle watcher — SOI Phase 1 (filing) then Phase 2 (Read refresh)."""

from __future__ import annotations

import threading
from typing import Callable

from ollama.config import OllamaConfig
from ollama.session import ChatSession
from ollama.soi_worker import SOIWorker


class IdleSOIWatcher:
    """Daemon thread driven by OAC idle time (SOI runs do not reset OAC idle)."""

    def __init__(
        self,
        session: ChatSession,
        config: OllamaConfig,
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.session = session
        self.config = config
        self.on_status = on_status or (lambda _msg: None)
        self._stop = threading.Event()
        self._busy = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="soi-idle", daemon=True)

    def start(self) -> None:
        if not self.config.soi_enabled:
            return
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        worker = SOIWorker(self.config)
        while not self._stop.wait(3.0):
            idle = self.session.idle_seconds()
            if not self._busy.acquire(blocking=False):
                continue
            try:
                # Phase 1: file changelog/inbox after short idle
                if worker.has_filing_work() and idle >= self.config.soi_idle_seconds:
                    self.on_status(
                        f"(SOI filing after {self.config.soi_idle_seconds:.0f}s OAC idle…)"
                    )
                    result = worker.run_filing()
                    if result.get("ran"):
                        self.on_status(
                            f"(SOI filing done: changelog={result.get('processed_changelog', 0)} "
                            f"inbox={result.get('seen_inbox', 0)})"
                        )
                    # Do NOT touch session — Phase 2 needs continued OAC idle toward 10m.
                    continue

                # Phase 2: refresh Read.json files after long idle + filing clear
                if (
                    not worker.has_filing_work()
                    and worker.needs_read_refresh()
                    and idle >= self.config.soi_read_refresh_idle_seconds
                ):
                    self.on_status(
                        f"(SOI Read refresh after "
                        f"{self.config.soi_read_refresh_idle_seconds:.0f}s OAC idle…)"
                    )
                    result = worker.run_read_refresh()
                    if result.get("ran"):
                        self.on_status(
                            f"(SOI Read refresh done: domains={result.get('domains', [])})"
                        )
            except Exception as exc:  # noqa: BLE001 — keep chat loop alive
                self.on_status(f"(SOI error: {exc})")
            finally:
                self._busy.release()
