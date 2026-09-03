"""SOI test harness — in-memory overlay on live db/, ephemeral sandbox for runs."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ainet.tools import changelog as changelog_mod
from ainet.tools.paths import DbPaths
from ollama.config import OllamaConfig
from ollama.modes import get_mode
from ollama.prompts import soi_test as soi_test_prompt
from ollama.session import ChatSession
from ollama.soi_worker import SOIWorker


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _preview(obj: Any, limit: int = 500) -> Any:
    try:
        raw = json.dumps(obj, ensure_ascii=False, default=str)
    except TypeError:
        raw = str(obj)
    if len(raw) <= limit:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw[:limit] + "…"


class ToolTrace:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def clear(self) -> None:
        self.calls.clear()

    def record(
        self,
        name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        *,
        blocked: bool,
    ) -> None:
        self.calls.append(
            {
                "ts": _utc_now(),
                "tool": name,
                "blocked": blocked,
                "args": args,
                "result_ok": result.get("ok", True),
                "result_preview": _preview(result),
            }
        )


class TestSOIWorker(SOIWorker):
    """SOI worker that uses the test harness prompt instead of production SOI."""

    def _build_filing_payload(self, batch_changelog: list[dict[str, Any]]) -> dict[str, Any]:
        from ollama.filing_payload import build_test_filing_payload
        from ollama.soi_worker import _entry_for_soi

        payload = build_test_filing_payload(
            self.db,
            batch_changelog,
            [],
            entry_for_soi=_entry_for_soi,
        )
        payload["phase"] = "filing"
        return payload

    def _ask_soi(self, payload: dict[str, Any]) -> tuple[str | None, str | None, dict[str, Any]]:
        from dataclasses import replace as dc_replace

        from ollama.client import OllamaError
        from ollama.filing_payload import format_test_user_message
        from ollama.prompts.soi_test import FILING_INSTRUCTIONS
        from ollama.soi_worker import _SOI_MAX_HISTORY, _SOI_MIN_TOOL_ROUNDS

        phase = payload.get("phase") or "soi_test"
        is_p2 = phase == "read_refresh"
        mode_id = "soi_test"

        soi_config = dc_replace(
            self.config,
            max_tool_rounds=max(self.config.max_tool_rounds, _SOI_MIN_TOOL_ROUNDS),
            max_history_messages=max(self.config.max_history_messages, _SOI_MAX_HISTORY),
            max_tool_result_chars=max(self.config.max_tool_result_chars, 12000),
            persist_oac_conversation=False,
            auto_mode=False,
        )
        session = ChatSession(
            mode=get_mode(mode_id),
            config=soi_config,
            client=self.client,
            auto_mode=False,
            persist_conversation=False,
        )
        if is_p2:
            folder = str(payload.get("folder") or "").strip()
            session.soi_folder_scope = folder or None
        stats: dict[str, Any] = {"mutating_calls": [], "tool_names": [], "tool_rounds": 0}

        def on_tool(phase_str: str, name: str, detail: dict[str, Any]) -> None:
            if phase_str == "start":
                args_obj = detail.get("arguments") or {}
                hint = ""
                for key in ("about", "query", "path", "url", "q", "dest", "entry_id"):
                    if key in args_obj and args_obj[key]:
                        hint = f" {key}={args_obj[key]!r}"
                        break
                self.log.log("tool_start", name=name, hint=hint, arguments=args_obj)
            elif phase_str == "done":
                self.log.log(
                    "tool_done",
                    name=name,
                    ok=bool(detail.get("ok", True)),
                    summary=detail.get("summary") or "",
                )

        stream = bool(self.config.soi_think)
        self.log.log(
            "model_ask",
            phase=phase,
            domain=payload.get("domain"),
            timeout_s=self.config.soi_timeout_s,
            think=self.config.soi_think,
            stream=stream,
        )
        try:
            user_text = format_test_user_message(FILING_INSTRUCTIONS, payload)
            reply = session.ask(
                user_text,
                stream=stream,
                on_thinking=None,
                on_tool=on_tool,
            )
            stats["mutating_calls"] = list(getattr(session, "last_mutating_calls", []) or [])
            stats["tool_names"] = list(getattr(session, "last_tool_names", []) or [])
            stats["tool_rounds"] = int(getattr(session, "last_tool_rounds", 0) or 0)
            self.log.log(
                "model_reply",
                phase=phase,
                chars=len(reply or ""),
                preview=(reply or "")[:2000],
                mutating_calls=len(stats["mutating_calls"]),
                tool_rounds=stats["tool_rounds"],
            )
            return reply, None, stats
        except OllamaError as exc:
            self.log.log("model_error", level="error", phase=phase, error=str(exc))
            return None, str(exc), stats


def _load_file_map(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        try:
            out[rel] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            out[rel] = path.read_bytes().decode("utf-8", errors="replace")
    return out


def _diff_maps(before: dict[str, str], after: dict[str, str]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    all_paths = sorted(set(before) | set(after))
    for rel in all_paths:
        b = before.get(rel)
        a = after.get(rel)
        if b == a:
            continue
        if rel not in before:
            kind = "created"
        elif rel not in after:
            kind = "deleted"
        else:
            kind = "modified"
        changes.append(
            {
                "path": rel,
                "kind": kind,
                "before": _preview_text(b),
                "after": _preview_text(a),
            }
        )
    return changes


def _preview_text(text: str | None, limit: int = 1200) -> str | None:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…"


class SOITestApp:
    """Ephemeral SOI filing lab — reads live db/, never writes back to it."""

    def __init__(self, config: OllamaConfig) -> None:
        self.source_db = Path(config.db_root).resolve()
        self.config = replace(
            config,
            persist_oac_conversation=False,
            soi_enabled=False,
            max_tool_rounds=max(config.max_tool_rounds, 24),
        )
        self.lock = threading.RLock()
        self.running = False
        self.events: list[dict[str, Any]] = []
        self._event_seq = 0
        self.trace = ToolTrace()
        self.seed_meta: dict[str, Any] = {}
        self.last_result: dict[str, Any] = {}
        self.baseline: dict[str, str] = {}
        self.current: dict[str, str] = {}
        self.changes: list[dict[str, Any]] = []
        self.sandbox: Path | None = None
        self._restore_hooks: list[Callable[[], None]] = []
        self.reset_session(extra_turns=True)

    def _emit(self, event: str, **fields: Any) -> None:
        with self.lock:
            self._event_seq += 1
            row = {"id": self._event_seq, "event": event, "ts": _utc_now(), **fields}
            self.events.append(row)
            if len(self.events) > 800:
                self.events = self.events[-600:]

    def events_after(self, after_id: int = 0) -> list[dict[str, Any]]:
        with self.lock:
            return [e for e in self.events if int(e.get("id") or 0) > after_id]

    def _teardown_sandbox(self) -> None:
        for restore in self._restore_hooks:
            try:
                restore()
            except Exception:
                pass
        self._restore_hooks.clear()
        if self.sandbox and self.sandbox.is_dir():
            try:
                shutil.rmtree(self.sandbox)
            except OSError:
                pass
        self.sandbox = None

    def reset_session(self, *, extra_turns: bool = True) -> dict[str, Any]:
        with self.lock:
            self._teardown_sandbox()
            self.trace.clear()
            self.last_result = {}
            self.changes = []
            self.running = False

            if not self.source_db.is_dir():
                raise FileNotFoundError(f"Database not found: {self.source_db}")

            self.sandbox = Path(tempfile.mkdtemp(prefix="ainet-soi-test-")).resolve()

            from scripts.test_soi import copy_sandbox, seed_pending_work

            copy_sandbox(self.source_db, self.sandbox)
            self.seed_meta = seed_pending_work(self.sandbox, extra_turns=extra_turns)
            self.baseline = _load_file_map(self.sandbox)
            self.current = dict(self.baseline)
            self.changes = []

            run_config = replace(self.config, db_root=self.sandbox)
            self._install_trace_hooks(run_config)

            self._emit("reset", seed=self.seed_meta, sandbox=str(self.sandbox))
            return self.status()

    def _install_trace_hooks(self, run_config: OllamaConfig) -> None:
        original_run = ChatSession._run_tool_call

        def patched_run(self_session: ChatSession, call: dict[str, Any]) -> dict[str, Any]:
            fn = call.get("function") or {}
            name = fn.get("name") or ""
            raw_args = fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}

            result = original_run(self_session, call)
            if not isinstance(result, dict):
                result = {"ok": True, "value": result}
            self.trace.record(name, args, result, blocked=False)
            self._emit(
                "tool",
                tool=name,
                args=args,
                ok=bool(result.get("ok", True)),
                summary=_preview(result, 300),
            )
            return result

        ChatSession._run_tool_call = patched_run  # type: ignore[method-assign]
        self._restore_hooks.append(lambda: setattr(ChatSession, "_run_tool_call", original_run))

    def _refresh_current(self) -> None:
        if self.sandbox:
            self.current = _load_file_map(self.sandbox)
            self.changes = _diff_maps(self.baseline, self.current)

    def status(self) -> dict[str, Any]:
        with self.lock:
            pending = []
            if self.sandbox:
                for entry in changelog_mod.pending_oac_entries(DbPaths(self.sandbox)):
                    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
                    pending.append(
                        {
                            "id": entry.get("id"),
                            "ts": entry.get("ts"),
                            "user_text": details.get("user_text") or entry.get("summary") or "",
                            "mode_id": details.get("mode_id"),
                            "soi_status": entry.get("soi_status"),
                        }
                    )
            return {
                "ok": True,
                "running": self.running,
                "source_db": str(self.source_db),
                "sandbox": str(self.sandbox) if self.sandbox else None,
                "model": self.config.model,
                "prompt": soi_test_prompt.PROMPT,
                "seed": self.seed_meta,
                "pending": pending,
                "pending_count": len(pending),
                "inbox_count": int(self.seed_meta.get("inbox_unfiled") or 0),
                "changes_count": len(self.changes),
                "tool_calls_count": len(self.trace.calls),
                "last_result": self.last_result or None,
                "event_seq": self._event_seq,
            }

    def tree(self, path: str = ".", max_depth: int = 3) -> dict[str, Any]:
        with self.lock:
            if not self.sandbox:
                return {"ok": False, "error": "No active session"}
            from ainet.tools.ops import DatabaseTools

            db = DatabaseTools(self.sandbox)
            try:
                return db.tree(path, max_depth=max_depth)
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

    def read_file(self, path: str, *, view: str = "current") -> dict[str, Any]:
        with self.lock:
            store = self.current if view == "current" else self.baseline
            norm = path.replace("\\", "/").strip()
            if norm not in store:
                return {"ok": False, "error": f"Not found: {path}", "view": view}
            text = store[norm]
            parsed: Any = None
            if norm.endswith(".json"):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = None
            return {
                "ok": True,
                "path": norm,
                "view": view,
                "content": text,
                "json": parsed,
            }

    def changes_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "ok": True,
                "count": len(self.changes),
                "changes": self.changes,
            }

    def tool_trace(self) -> dict[str, Any]:
        with self.lock:
            return {"ok": True, "calls": list(self.trace.calls)}

    def run_filing(self) -> dict[str, Any]:
        with self.lock:
            if self.running:
                return {"ok": False, "error": "Already running"}
            if not self.sandbox:
                return {"ok": False, "error": "No active session — reset first"}
            self.running = True

        def _worker() -> None:
            try:
                self._emit("run_start", phase="filing")
                run_config = replace(self.config, db_root=self.sandbox)
                worker = TestSOIWorker(
                    run_config,
                    on_status=lambda msg: self._emit(
                        "soi_log",
                        text=str(msg) if not isinstance(msg, dict) else json.dumps(msg, default=str),
                    ),
                )
                result = worker.run_filing()
                self._refresh_current()
                with self.lock:
                    self.last_result = result
                self._emit("run_done", phase="filing", result=result)
            except Exception as exc:
                tb = traceback.format_exc()
                with self.lock:
                    self.last_result = {"ok": False, "error": str(exc), "traceback": tb}
                self._emit("run_error", error=str(exc), traceback=tb)
            finally:
                with self.lock:
                    self.running = False

        threading.Thread(target=_worker, name="soi-test-run", daemon=True).start()
        return {"ok": True, "started": True}

    def run_read_refresh(self) -> dict[str, Any]:
        result = {
            "ok": True,
            "ran": False,
            "phase": "read_refresh",
            "reason": "phase 2 removed",
        }
        with self.lock:
            self.last_result = result
        self._emit("run_done", phase="read_refresh", result=result)
        return result
