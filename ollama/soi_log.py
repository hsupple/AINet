"""SOI logging — JSONL under db/runtime/soi/ plus optional console lines."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


StatusPayload = str | dict[str, Any]


def status_line(msg: StatusPayload) -> str:
    if isinstance(msg, dict):
        return str(msg.get("text") or "")
    return str(msg or "")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class SOILogger:
    """Append-only events.jsonl + optional human status callback."""

    # Events that also print to the chat console (everything still hits JSONL).
    _CONSOLE = {
        "idle_wake",
        "filing_start",
        "filing_done",
        "filing_skip",
        "filing_error",
        "read_refresh_start",
        "read_refresh_done",
        "read_refresh_skip",
        "read_refresh_error",
        "tool_start",
        "tool_done",
        "model_ask",
        "model_reply",
        "model_no_tools",
        "model_error",
        "model_retry",
        "error",
        "backoff",
        "watcher_start",
    }

    def __init__(
        self,
        db_root: Path,
        *,
        on_status: Callable[[StatusPayload], None] | None = None,
    ) -> None:
        self.dir = Path(db_root) / "runtime" / "soi"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.jsonl"
        self.on_status = on_status

    def log(self, event: str, *, level: str = "info", console: bool | None = None, **fields: Any) -> None:
        record = {
            "ts": _utc_now(),
            "level": level,
            "event": event,
            **{k: v for k, v in fields.items() if v is not None},
        }
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

        show = self._CONSOLE.__contains__(event) if console is None else console
        if level == "error":
            show = True
        if not show or not self.on_status:
            return
        line = self._format_status(event, level=level, **fields)
        if line:
            payload: dict[str, Any] = {
                "text": line,
                "event": event,
                "level": level,
                **{k: v for k, v in fields.items() if v is not None},
            }
            self.on_status(payload)

    def _format_status(self, event: str, *, level: str, **fields: Any) -> str:
        if event == "filing_start":
            return (
                f"(SOI filing... changelog={fields.get('pending_changelog', 0)} "
                f"inbox={fields.get('pending_inbox', 0)})"
            )
        if event == "filing_done":
            return (
                f"(SOI filing done: filed={fields.get('marked_filed', 0)} "
                f"discarded={fields.get('marked_discarded', 0)} "
                f"pending={fields.get('left_pending', 0)} "
                f"inbox={fields.get('seen_inbox', 0)})"
            )
        if event == "filing_skip":
            return f"(SOI filing skip: {fields.get('reason', 'no work')})"
        if event == "filing_error":
            return f"(SOI filing error: {fields.get('error', 'unknown')})"
        if event == "read_refresh_start":
            paths = fields.get("stale_count", 0)
            domains = fields.get("domains") or []
            return f"(SOI Read refresh... stale={paths} domains={domains})"
        if event == "read_refresh_done":
            return (
                f"(SOI Read refresh done: refreshed={len(fields.get('refreshed') or [])} "
                f"remaining={len(fields.get('stale_remaining') or [])})"
            )
        if event == "read_refresh_skip":
            return f"(SOI Read refresh skip: {fields.get('reason', 'no work')})"
        if event == "read_refresh_error":
            return f"(SOI Read refresh error: {fields.get('error', 'unknown')})"
        if event == "tool_start":
            name = fields.get("name") or "?"
            hint = fields.get("hint") or ""
            return f"(SOI tool -> {name}{hint})"
        if event == "tool_done":
            name = fields.get("name") or "?"
            mark = "ok" if fields.get("ok", True) else "FAIL"
            summary = fields.get("summary") or ""
            return f"(SOI tool {mark} {name}: {summary})"
        if event == "model_no_tools":
            return "(SOI emitted prose with zero native tool_calls)"
        if event == "model_ask":
            return (
                f"(SOI model ask phase={fields.get('phase', 'soi')} "
                f"think={fields.get('think', False)})"
            )
        if event == "model_reply":
            preview = str(fields.get("preview") or "").strip()
            n = fields.get("chars", len(preview))
            tools = fields.get("mutating_calls", 0)
            head = f"(SOI model reply {n} chars, mutating={tools})"
            return f"{head}\n{preview}" if preview else head
        if event == "model_error":
            return f"(SOI model error: {fields.get('error', 'unknown')})"
        if event == "model_retry":
            return (
                f"(SOI model retry {fields.get('attempt', '?')}/"
                f"{fields.get('max_attempts', '?')}: {fields.get('error', 'unknown')})"
            )
        if event == "idle_wake":
            return (
                f"(SOI wake: phase={fields.get('phase')} "
                f"idle={fields.get('idle_s', 0):.0f}s)"
            )
        if event == "watcher_start":
            return (
                f"(SOI watcher on: file@{fields.get('soi_idle_seconds', '?')}s "
                f"read@{fields.get('soi_read_refresh_idle_seconds', '?')}s "
                f"timeout@{fields.get('soi_timeout_s', '?')}s)"
            )
        if event == "backoff":
            return (
                f"(SOI backing off {fields.get('seconds', '?')}s -- "
                f"{fields.get('reason', 'error')})"
            )
        if event == "error":
            return f"(SOI error: {fields.get('error', 'unknown')})"
        if level == "error":
            return f"(SOI {event}: {fields.get('error') or fields})"
        return f"(SOI {event})"
