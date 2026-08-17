"""SOI worker — file pending oac_turns via log_item."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ainet.logstore import decay_knowledge, ensure_knowledge_files
from ainet.tools import changelog
from ainet.tools.fsutil import atomic_write_text
from ainet.tools.ops import DatabaseTools
from ollama.client import OllamaCancelled, OllamaClient, OllamaError
from ollama.config import OllamaConfig
from ollama.modes import get_mode
from ollama.content_filing import is_ephemeral_text
from ollama.filing_payload import build_test_filing_payload, format_test_user_message
from ollama.prompts.soi_test import FILING_INSTRUCTIONS
from ollama.session import ChatSession
from ollama.soi_log import SOILogger

_FILING_BATCH_SIZE = 4
_FILING_STREAM_RETRIES = 2
_SOI_MIN_TOOL_ROUNDS = 6
_SOI_MAX_HISTORY = 16


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _entry_user_text(entry: dict[str, Any]) -> str:
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    return str(details.get("user_text") or entry.get("summary") or "").strip()


def _entry_for_soi(entry: dict[str, Any]) -> dict[str, Any]:
    """SOI sees id, user text, and time only — never OAC assistant_text."""
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    return {
        "id": entry.get("id"),
        "ts": entry.get("ts"),
        "session_id": str(details.get("session_id") or "").strip(),
        "user_text": str(details.get("user_text") or entry.get("summary") or "").strip(),
    }


def _is_ephemeral_entry(entry: dict[str, Any]) -> bool:
    return is_ephemeral_text(_entry_user_text(entry))


def _retryable_model_error(err: str | None) -> bool:
    if not err:
        return False
    low = err.lower()
    return any(
        token in low
        for token in (
            "stream interrupted",
            "stream non-json",
            "stream truncated",
            "timed out",
            "cannot reach ollama",
        )
    )


def _add_dest(dest_by_id: dict[str, Any], eid: str, path: str) -> None:
    path = str(path or "").replace("\\", "/").strip()
    if not eid:
        return
    if not path:
        dest_by_id.setdefault(eid, "")
        return
    cur = dest_by_id.get(eid)
    if not cur:
        dest_by_id[eid] = path
        return
    if isinstance(cur, list):
        if path not in cur:
            cur.append(path)
        return
    if cur == path:
        return
    dest_by_id[eid] = [cur, path]


class SOIWorker:
    def __init__(
        self,
        config: OllamaConfig | None = None,
        client: OllamaClient | None = None,
        *,
        on_status: Callable[[str], None] | None = None,
        logger: SOILogger | None = None,
    ) -> None:
        self.config = config or OllamaConfig.from_env()
        self.client = client or OllamaClient(self.config)
        self.db = DatabaseTools(self.config.db_root)
        ensure_knowledge_files(Path(self.config.db_root))
        changelog.ensure_changelog_file(Path(self.config.db_root))
        changelog.ensure_masterlog_file(Path(self.config.db_root))
        changelog.migrate_resolved_to_masterlog(self.db.paths)
        self.state_dir = Path(self.config.db_root) / "runtime" / "soi"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "state.json"
        self.cursor_path = self.state_dir / "cursor.json"
        self.log = logger or SOILogger(self.config.db_root, on_status=on_status)
        self.cancel_event = threading.Event()
        self._active_session: ChatSession | None = None

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def interrupt(self) -> None:
        self.cancel_event.set()
        try:
            self.client.cancel_active()
        except Exception:
            pass
        active = self._active_session
        if active is not None:
            try:
                active.request_cancel()
            except Exception:
                pass

    def pending_changelog(self) -> list[dict[str, Any]]:
        return changelog.pending_oac_entries(self.db.paths)

    def pending_inbox(self) -> list[dict[str, Any]]:
        return []

    def has_filing_work(self) -> bool:
        return bool(self.pending_changelog())

    def has_work(self) -> bool:
        return self.has_filing_work()

    def needs_read_refresh(self) -> bool:
        return False

    def list_stale_read_paths(self) -> list[str]:
        return []

    def run_filing(self) -> dict[str, Any]:
        try:
            decay_knowledge(self.db)
        except Exception as exc:  # noqa: BLE001
            self.log.log("decay_error", level="error", error=str(exc))
        changelog_pending = self.pending_changelog()
        if not changelog_pending:
            self._merge_state({"status": "idle", "phase": "filing", "reason": "no pending work"})
            self.log.log("filing_skip", reason="no pending work")
            return {"ok": True, "ran": False, "phase": "filing", "reason": "no pending work"}

        self.log.log(
            "filing_start",
            pending_changelog=len(changelog_pending),
            entry_ids=[e.get("id") for e in changelog_pending[:40] if e.get("id")],
        )
        totals = {
            "processed_changelog": 0,
            "marked_filed": 0,
            "marked_discarded": 0,
            "left_pending": 0,
            "mutating_tool_calls": 0,
            "batches": 0,
            "retries": 0,
            "replies": [],
        }
        errors: list[str] = []
        seen_ids: set[str] = set()
        cancelled = False
        while True:
            if self.cancelled():
                cancelled = True
                break
            batch_changelog = self.pending_changelog()[:_FILING_BATCH_SIZE]
            if not batch_changelog:
                break
            if totals["batches"] >= 6:
                break
            batch_ids = [str(e.get("id") or "") for e in batch_changelog if e.get("id")]
            if batch_ids and all(eid in seen_ids for eid in batch_ids):
                break
            seen_ids.update(batch_ids)
            totals["batches"] += 1
            result = self._run_filing_batch(batch_changelog)
            if result.get("cancelled"):
                cancelled = True
                break
            if result.get("error"):
                errors.append(str(result["error"]))
            totals["processed_changelog"] += int(result.get("processed_changelog") or 0)
            totals["marked_filed"] += int(result.get("marked_filed") or 0)
            totals["marked_discarded"] += int(result.get("marked_discarded") or 0)
            totals["left_pending"] += int(result.get("left_pending") or 0)
            totals["mutating_tool_calls"] += int(result.get("mutating_tool_calls") or 0)
            totals["retries"] += int(result.get("retries") or 0)
            if result.get("reply"):
                totals["replies"].append(str(result["reply"])[:400])
            if (
                int(result.get("marked_filed") or 0) == 0
                and int(result.get("marked_discarded") or 0) == 0
                and int(result.get("mutating_tool_calls") or 0) == 0
            ):
                break
            if int(result.get("left_pending") or 0) >= len(batch_changelog) and len(batch_changelog) > 0:
                if int(result.get("marked_discarded") or 0) == 0 and int(result.get("marked_filed") or 0) == 0:
                    break

        ok = not errors
        self._merge_state(
            {
                "status": "cancelled" if cancelled else ("ok" if ok else "error"),
                "phase": "filing",
                "processed_changelog": totals["processed_changelog"],
                "marked_filed": totals["marked_filed"],
                "marked_discarded": totals["marked_discarded"],
                "left_pending": totals["left_pending"],
                "mutating_tool_calls": totals["mutating_tool_calls"],
                "batches": totals["batches"],
                "last_filing_at": _utc_now(),
                "reply_preview": (totals["replies"][-1] if totals["replies"] else "")[:400],
                "errors": errors or None,
            }
        )
        out = {
            "ok": ok,
            "ran": True,
            "phase": "filing",
            "cancelled": cancelled,
            "processed_changelog": totals["processed_changelog"],
            "marked_filed": totals["marked_filed"],
            "marked_discarded": totals["marked_discarded"],
            "left_pending": totals["left_pending"],
            "mutating_tool_calls": totals["mutating_tool_calls"],
            "batches": totals["batches"],
            "retries": totals["retries"],
            "pending_remaining": len(self.pending_changelog()),
            "replies": totals["replies"],
            "errors": errors or None,
            "processed_any": totals["processed_changelog"] > 0,
        }
        self.log.log(
            "filing_done",
            level="error" if not ok else "info",
            marked_filed=totals["marked_filed"],
            marked_discarded=totals["marked_discarded"],
            left_pending=totals["left_pending"],
            cancelled=cancelled or None,
            errors=errors or None,
        )
        return out

    def _build_filing_payload(self, batch_changelog: list[dict[str, Any]]) -> dict[str, Any]:
        payload = build_test_filing_payload(
            self.db,
            batch_changelog,
            [],
            entry_for_soi=_entry_for_soi,
        )
        payload["phase"] = "filing"
        return payload

    def _run_filing_batch(self, batch_changelog: list[dict[str, Any]]) -> dict[str, Any]:
        payload = self._build_filing_payload(batch_changelog)
        reply, err, stats = self._ask_soi(payload)
        retries = 0
        if err == "cancelled" or stats.get("cancelled"):
            return {
                "ok": True,
                "cancelled": True,
                "processed_changelog": len(batch_changelog),
                "mutating_calls": stats.get("mutating_calls") or [],
            }
        while (
            err
            and not (stats.get("mutating_calls") or [])
            and retries < _FILING_STREAM_RETRIES
            and _retryable_model_error(err)
            and not self.cancelled()
        ):
            retries += 1
            self.log.log(
                "model_retry",
                phase="filing",
                attempt=retries,
                max_attempts=_FILING_STREAM_RETRIES,
                error=err,
            )
            reply, err, stats = self._ask_soi(payload)
            if err == "cancelled" or stats.get("cancelled"):
                return {
                    "ok": True,
                    "cancelled": True,
                    "processed_changelog": len(batch_changelog),
                    "mutating_calls": stats.get("mutating_calls") or [],
                    "retries": retries,
                }

        if err and not (stats.get("mutating_calls") or []):
            return {
                "ok": False,
                "error": err,
                "processed_changelog": len(batch_changelog),
                "retries": retries,
            }

        entry_ids = [str(e["id"]) for e in batch_changelog if e.get("id")]
        discarded_ids = self._parse_id_list(reply, entry_ids, keys=("discarded", "discarded_ids"))
        known = set(entry_ids)
        handled_by_id: set[str] = set()
        dest_by_id: dict[str, Any] = {}
        by_entry = {str(e.get("id") or ""): e for e in batch_changelog}
        discarded_ids = {
            eid
            for eid in discarded_ids
            if (by_entry.get(eid) and _is_ephemeral_entry(by_entry[eid]))
        }
        for call in stats.get("mutating_calls") or []:
            if call.get("tool") not in {"log_item", "file_note"}:
                continue
            if call.get("ok") is False:
                continue
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            result = call.get("result") if isinstance(call.get("result"), dict) else {}
            dest = str(args.get("dest") or result.get("action") or result.get("dest") or "").strip().lower()
            ids = [str(x) for x in (args.get("entry_ids") or result.get("entry_ids") or []) if x]
            eid = str(args.get("entry_id") or args.get("id") or "").strip()
            if eid:
                ids.append(eid)
            for item in ids:
                if item not in known:
                    continue
                if dest in {"discard", "ephemeral", "drop"}:
                    handled_by_id.add(item)
                    discarded_ids.add(item)
                    dest_by_id[item] = ""
                    continue
                dest_path = str(
                    result.get("path")
                    or result.get("folder")
                    or result.get("filed_to")
                    or args.get("dest")
                    or ""
                ).replace("\\", "/")
                handled_by_id.add(item)
                _add_dest(dest_by_id, item, dest_path)

        for entry in batch_changelog:
            eid = str(entry.get("id") or "")
            if eid and _is_ephemeral_entry(entry):
                discarded_ids.add(eid)

        host_filed_ids = {k for k, v in dest_by_id.items() if v}
        filed_ids: list[str] = []
        left_pending: list[str] = []
        for entry in batch_changelog:
            eid = str(entry.get("id") or "")
            if not eid:
                continue
            if eid in discarded_ids:
                continue
            if eid in host_filed_ids or eid in handled_by_id:
                filed_ids.append(eid)
                continue
            left_pending.append(eid)

        filed_ids = [eid for eid in filed_ids if eid not in discarded_ids]
        marked_filed = changelog.mark_soi_status(
            self.db.paths, entry_ids=filed_ids, status="filed", dest_by_id=dest_by_id
        )
        marked_discarded = changelog.mark_soi_status(
            self.db.paths,
            entry_ids=list(discarded_ids),
            status="discarded",
            dest_by_id=dest_by_id,
        )
        if batch_changelog:
            self._save_cursor(batch_changelog[-1]["index"])
        return {
            "ok": True,
            "processed_changelog": len(batch_changelog),
            "marked_filed": marked_filed,
            "marked_discarded": marked_discarded,
            "left_pending": len(left_pending),
            "left_pending_ids": left_pending,
            "mutating_tool_calls": len(stats.get("mutating_calls") or []),
            "tool_names": stats.get("tool_names") or [],
            "retries": retries,
            "reply": reply,
        }

    def run_once(self) -> dict[str, Any]:
        return self.run_filing()

    def run_read_refresh(self) -> dict[str, Any]:
        self.log.log("read_refresh_skip", reason="phase 2 removed")
        return {"ok": True, "ran": False, "phase": "read_refresh", "reason": "removed"}

    def _ask_soi(self, payload: dict[str, Any]) -> tuple[str | None, str | None, dict[str, Any]]:
        from dataclasses import replace

        soi_config = replace(
            self.config,
            max_tool_rounds=max(self.config.max_tool_rounds, _SOI_MIN_TOOL_ROUNDS),
            max_history_messages=max(self.config.max_history_messages, _SOI_MAX_HISTORY),
            max_tool_result_chars=max(self.config.max_tool_result_chars, 12000),
            persist_oac_conversation=False,
            auto_mode=False,
        )
        session = ChatSession(
            mode=get_mode("soi"),
            config=soi_config,
            client=self.client,
            auto_mode=False,
            persist_conversation=False,
        )
        session.cancel_event = self.cancel_event
        self._active_session = session
        stats: dict[str, Any] = {"mutating_calls": [], "tool_names": [], "tool_rounds": 0}

        def on_tool(phase: str, name: str, detail: dict[str, Any]) -> None:
            if phase == "start":
                args_obj = detail.get("arguments") or {}
                hint = ""
                for key in ("dest", "label", "entry_id", "query", "path"):
                    if key in args_obj and args_obj[key]:
                        hint = f" {key}={args_obj[key]!r}"
                        break
                self.log.log("tool_start", name=name, hint=hint, arguments=args_obj)
            elif phase == "done":
                self.log.log(
                    "tool_done",
                    name=name,
                    ok=bool(detail.get("ok", True)),
                    summary=detail.get("summary") or "",
                )

        phase = payload.get("phase") or "soi"
        self.log.log(
            "model_ask",
            phase=phase,
            timeout_s=self.config.soi_timeout_s,
            think=self.config.soi_think,
            stream=False,
        )
        try:
            user_text = format_test_user_message(FILING_INSTRUCTIONS, payload)
            reply = session.ask(
                user_text,
                stream=False,
                on_thinking=None,
                on_tool=on_tool,
            )
            stats["mutating_calls"] = list(getattr(session, "last_mutating_calls", []) or [])
            stats["tool_names"] = list(getattr(session, "last_tool_names", []) or [])
            stats["tool_rounds"] = int(getattr(session, "last_tool_rounds", 0) or 0)
            if not stats["tool_names"]:
                self.log.log(
                    "model_no_tools",
                    phase=phase,
                    chars=len(reply or ""),
                    preview=(reply or "")[:240],
                )
            self.log.log(
                "model_reply",
                phase=phase,
                chars=len(reply or ""),
                preview=(reply or "")[:2000],
                mutating_calls=len(stats["mutating_calls"]),
                tool_rounds=stats["tool_rounds"],
            )
            return reply, None, stats
        except OllamaCancelled:
            stats["mutating_calls"] = list(getattr(session, "last_mutating_calls", []) or [])
            stats["tool_names"] = list(getattr(session, "last_tool_names", []) or [])
            stats["tool_rounds"] = int(getattr(session, "last_tool_rounds", 0) or 0)
            stats["cancelled"] = True
            self.log.log("model_error", level="info", phase=phase, error="cancelled")
            return None, "cancelled", stats
        except OllamaError as exc:
            stats["mutating_calls"] = list(getattr(session, "last_mutating_calls", []) or [])
            stats["tool_names"] = list(getattr(session, "last_tool_names", []) or [])
            stats["tool_rounds"] = int(getattr(session, "last_tool_rounds", 0) or 0)
            self.log.log("model_error", level="error", phase=phase, error=str(exc))
            return None, str(exc), stats
        finally:
            self._active_session = None

    def _parse_id_list(
        self,
        reply: str | None,
        known_ids: list[str],
        *,
        keys: tuple[str, ...],
    ) -> set[str]:
        found: set[str] = set()
        if not reply or not known_ids:
            return found
        known = set(known_ids)
        start = reply.find("{")
        end = reply.rfind("}")
        if start >= 0 and end > start:
            try:
                blob = json.loads(reply[start : end + 1])
            except json.JSONDecodeError:
                blob = None
            if isinstance(blob, dict):
                for key in keys:
                    raw = blob.get(key) or []
                    if isinstance(raw, list):
                        for item in raw:
                            token = str(item).strip()
                            token = token.split()[0].strip(" ,;:") if token else ""
                            if token in known:
                                found.add(token)
        return found

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _merge_state(self, patch: dict[str, Any]) -> None:
        state = self._load_state()
        state.update(patch)
        state["updated_at"] = _utc_now()
        atomic_write_text(
            self.state_path,
            json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        )

    def _save_cursor(self, last_index: int) -> None:
        atomic_write_text(
            self.cursor_path,
            json.dumps({"last_index": last_index, "updated_at": _utc_now()}, indent=2) + "\n",
        )
