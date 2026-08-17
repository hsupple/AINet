"""Background idle watcher — SOI filing after OAC idle."""

from __future__ import annotations

import threading
import time
from typing import Callable

from ainet.logstore import decay_knowledge
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
        self._kick_phase: str | None = None
        self._cancel_job = threading.Event()
        self._active = threading.Event()
        self._busy = threading.Lock()
        self._next_ok_at = 0.0
        self._last_decay_at = 0.0
        self._thread = threading.Thread(target=self._loop, name="soi-idle", daemon=True)
        self.worker: SOIWorker | None = None

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    @property
    def running(self) -> bool:
        return self._active.is_set() or self.busy or self._kick.is_set()

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
        """Start filing now (Start SOI) — bypasses the automatic idle wait."""
        if not self.start(force=True):
            return {"started": False, "reason": "watcher failed to start"}
        if self.busy:
            return {"started": False, "reason": "already running"}
        self._cancel_job.clear()
        self._kick_phase = "filing"
        self._active.set()
        self._kick.set()
        return {"started": True}

    def request_read_refresh(self) -> dict[str, bool | str]:
        return {"started": False, "reason": "phase 2 removed"}

    def cancel_job(self) -> dict[str, bool | str]:
        self._cancel_job.set()
        self._kick.clear()
        self._kick_phase = None
        worker = self.worker
        if worker is not None:
            try:
                worker.interrupt()
            except Exception:
                pass
        if not self.busy:
            self._active.clear()
            return {"stopped": True, "reason": "SOI was idle"}
        return {"stopped": True, "reason": "cancel requested"}

    def stop(self) -> None:
        self._stop.set()
        self._cancel_job.set()
        self._kick_phase = None
        self._kick.set()
        self._active.clear()
        worker = self.worker
        if worker is not None:
            try:
                worker.interrupt()
            except Exception:
                pass

    def _loop(self) -> None:
        logger = SOILogger(self.config.db_root, on_status=self.on_status)
        worker = SOIWorker(self.config, on_status=self.on_status, logger=logger)
        self.worker = worker
        logger.log(
            "watcher_start",
            soi_idle_seconds=self.config.soi_idle_seconds,
            soi_timeout_s=self.config.soi_timeout_s,
        )
        while not self._stop.is_set():
            self._kick.wait(3.0)
            if self._stop.is_set():
                break
            kicked = self._kick.is_set()
            if kicked:
                self._kick.clear()
                self._kick_phase = None
            if time.monotonic() < self._next_ok_at and not kicked:
                continue
            idle = self.session.idle_seconds()
            if not self._busy.acquire(blocking=False):
                if kicked:
                    self._kick.set()
                continue
            try:
                worker.cancel_event = self._cancel_job
                idle_ready = kicked or idle >= self.config.soi_idle_seconds
                if idle_ready and time.monotonic() - self._last_decay_at >= 900:
                    try:
                        pruned = decay_knowledge(self.session.db)
                        self._last_decay_at = time.monotonic()
                        if (
                            pruned.get("dropped_keys")
                            or pruned.get("dropped_entries")
                            or pruned.get("files")
                        ):
                            logger.log(
                                "decay",
                                dropped_keys=pruned.get("dropped_keys"),
                                dropped_entries=pruned.get("dropped_entries"),
                                files=pruned.get("files"),
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.log("decay_error", level="error", error=str(exc))
                can_file = worker.has_filing_work() and idle_ready
                if not can_file:
                    continue
                logger.log(
                    "idle_wake",
                    phase="filing",
                    idle_s=idle,
                    kicked=bool(kicked),
                    pending_changelog=len(worker.pending_changelog()),
                )
                result = worker.run_filing()
                if result.get("cancelled"):
                    logger.log("filing_skip", reason="cancelled")
                    continue
                if not result.get("ok"):
                    self._next_ok_at = time.monotonic() + self.error_backoff_s
                    logger.log(
                        "backoff",
                        seconds=self.error_backoff_s,
                        reason=result.get("error") or "filing failed",
                    )
                elif int(result.get("left_pending") or 0) and not int(
                    result.get("marked_filed") or 0
                ) and not int(result.get("marked_discarded") or 0):
                    self._next_ok_at = time.monotonic() + self.error_backoff_s
                    logger.log(
                        "backoff",
                        seconds=self.error_backoff_s,
                        reason="filing produced no placements",
                    )
            except Exception as exc:  # noqa: BLE001 — keep chat loop alive
                self._next_ok_at = time.monotonic() + self.error_backoff_s
                logger.log("error", level="error", error=str(exc))
                logger.log("backoff", seconds=self.error_backoff_s, reason=str(exc))
            finally:
                self._busy.release()
                if kicked:
                    self._active.clear()
