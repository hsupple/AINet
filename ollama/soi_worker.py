"""SOI worker — Phase 1 filing (changelog/inbox), Phase 2 Read.json refresh."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ainet.tools import changelog
from ainet.tools import readlog
from ainet.tools.fsutil import atomic_write_text
from ainet.tools.ops import DatabaseTools
from ainet.tools.registry import dispatch
from ollama.client import OllamaCancelled, OllamaClient, OllamaError
from ollama.config import OllamaConfig
from ollama.modes import get_mode
from ollama.content_filing import (
    cop_name_in_text,
    is_ephemeral_text,
)
from ollama.filing_payload import (
    build_read_refresh_folders,
    build_test_filing_payload,
    format_read_refresh_message,
    format_test_user_message,
)
from ollama.prompts.soi_test import FILING_INSTRUCTIONS, READ_REFRESH_INSTRUCTIONS
from ollama.session import ChatSession
from ollama.soi_log import SOILogger

_FILING_BATCH_SIZE = 4
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


def _inbox_for_soi(cap: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": cap.get("id"),
        "suggested_home": cap.get("suggested_home") or "",
        "text": str(cap.get("text") or "").strip(),
    }


def _is_ephemeral_entry(entry: dict[str, Any]) -> bool:
    return is_ephemeral_text(_entry_user_text(entry))


def _add_dest(dest_by_id: dict[str, Any], eid: str, path: str) -> None:
    """Record one or more filing destinations for a changelog id."""
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
        changelog.ensure_changelog_file(Path(self.config.db_root))
        changelog.ensure_masterlog_file(Path(self.config.db_root))
        changelog.migrate_resolved_to_masterlog(self.db.paths)
        self.state_dir = Path(self.config.db_root) / "runtime" / "soi"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "state.json"
        # Legacy cursor kept for migration/debug; pending uses per-entry soi_status.
        self.cursor_path = self.state_dir / "cursor.json"
        self.log = logger or SOILogger(self.config.db_root, on_status=on_status)
        self.cancel_event = threading.Event()
        self._active_session: ChatSession | None = None

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def interrupt(self) -> None:
        """Unblock an in-flight SOI model call (Stop / Reset AI)."""
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

    # ---- pending queues ----------------------------------------------------

    def pending_changelog(self) -> list[dict[str, Any]]:
        return changelog.pending_oac_entries(self.db.paths)

    def pending_inbox(self) -> list[dict[str, Any]]:
        path = "Hayden/Inbox/Captures.json"
        if not self.db.paths.resolve(path).exists():
            return []
        data = self.db.read_json(path)["data"]
        captures = data.get("captures") or []
        if not isinstance(captures, list):
            return []
        return [c for c in captures if isinstance(c, dict) and c.get("status") == "unfiled"]

    def has_filing_work(self) -> bool:
        return bool(self.pending_changelog() or self.pending_inbox())

    def has_work(self) -> bool:
        """Back-compat alias: filing work only."""
        return self.has_filing_work()

    def needs_read_refresh(self) -> bool:
        if self.list_stale_read_paths():
            return True
        state = self._load_state()
        return bool(state.get("needs_read_refresh"))

    def list_stale_read_paths(self) -> list[str]:
        return list(self.db.list_stale_reads().get("paths") or [])

    # ---- phase 1: filing ---------------------------------------------------

    def run_filing(self) -> dict[str, Any]:
        changelog_pending = self.pending_changelog()
        inbox = self.pending_inbox()
        if not changelog_pending and not inbox:
            self._merge_state({"status": "idle", "phase": "filing", "reason": "no pending work"})
            self.log.log("filing_skip", reason="no pending work")
            return {"ok": True, "ran": False, "phase": "filing", "reason": "no pending work"}

        self.log.log(
            "filing_start",
            pending_changelog=len(changelog_pending),
            pending_inbox=len(inbox),
            entry_ids=[e.get("id") for e in changelog_pending[:40] if e.get("id")],
        )
        totals = {
            "processed_changelog": 0,
            "marked_filed": 0,
            "marked_discarded": 0,
            "left_pending": 0,
            "seen_inbox": 0,
            "mutating_tool_calls": 0,
            "batches": 0,
            "retries": 0,
            "replies": [],
        }
        errors: list[str] = []
        seen_ids: set[str] = set()
        cancelled = False
        # Process small batches so the model can actually call tools per item.
        while True:
            if self.cancelled():
                cancelled = True
                break
            batch_changelog = self.pending_changelog()[:_FILING_BATCH_SIZE]
            batch_inbox = self.pending_inbox()[:_FILING_BATCH_SIZE]
            if not batch_changelog and not batch_inbox:
                break
            if totals["batches"] >= 6:
                break
            batch_ids = [str(e.get("id") or "") for e in batch_changelog if e.get("id")]
            if batch_ids and all(eid in seen_ids for eid in batch_ids):
                break
            seen_ids.update(batch_ids)
            totals["batches"] += 1
            result = self._run_filing_batch(batch_changelog, batch_inbox)
            if result.get("cancelled"):
                cancelled = True
                break
            if result.get("error"):
                errors.append(str(result["error"]))
            totals["processed_changelog"] += int(result.get("processed_changelog") or 0)
            totals["marked_filed"] += int(result.get("marked_filed") or 0)
            totals["marked_discarded"] += int(result.get("marked_discarded") or 0)
            totals["left_pending"] += int(result.get("left_pending") or 0)
            totals["seen_inbox"] += int(result.get("seen_inbox") or 0)
            totals["mutating_tool_calls"] += int(result.get("mutating_tool_calls") or 0)
            totals["retries"] += int(result.get("retries") or 0)
            if result.get("reply"):
                totals["replies"].append(str(result["reply"])[:400])
            # If nothing was resolved and no tools ran, stop to avoid infinite loops.
            if (
                int(result.get("marked_filed") or 0) == 0
                and int(result.get("marked_discarded") or 0) == 0
                and int(result.get("mutating_tool_calls") or 0) == 0
            ):
                break
            # If this batch left everything pending, stop (model failed to file).
            if int(result.get("left_pending") or 0) >= len(batch_changelog) and len(batch_changelog) > 0:
                if int(result.get("marked_discarded") or 0) == 0 and int(
                    result.get("marked_filed") or 0
                ) == 0:
                    break

        ok = not errors
        processed_any = totals["processed_changelog"] > 0 or totals["seen_inbox"] > 0
        self._merge_state(
            {
                "status": "cancelled" if cancelled else ("ok" if ok else "error"),
                "phase": "filing",
                "processed_changelog": totals["processed_changelog"],
                "marked_filed": totals["marked_filed"],
                "marked_discarded": totals["marked_discarded"],
                "left_pending": totals["left_pending"],
                "seen_inbox": totals["seen_inbox"],
                "mutating_tool_calls": totals["mutating_tool_calls"],
                "batches": totals["batches"],
                "needs_read_refresh": True if totals["marked_filed"] else self.needs_read_refresh(),
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
            "seen_inbox": totals["seen_inbox"],
            "mutating_tool_calls": totals["mutating_tool_calls"],
            "batches": totals["batches"],
            "retries": totals["retries"],
            "pending_remaining": len(self.pending_changelog()),
            "inbox_remaining": len(self.pending_inbox()),
            "replies": totals["replies"],
            "errors": errors or None,
            "processed_any": processed_any,
        }
        self.log.log(
            "filing_done",
            level="error" if not ok else "info",
            marked_filed=totals["marked_filed"],
            marked_discarded=totals["marked_discarded"],
            left_pending=totals["left_pending"],
            seen_inbox=totals["seen_inbox"],
            cancelled=cancelled or None,
            errors=errors or None,
        )
        return out

    def _build_filing_payload(
        self,
        batch_changelog: list[dict[str, Any]],
        batch_inbox: list[dict[str, Any]],
        *,
        layout: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = layout
        payload = build_test_filing_payload(
            self.db,
            batch_changelog,
            batch_inbox,
            entry_for_soi=_entry_for_soi,
            inbox_for_soi=_inbox_for_soi,
        )
        payload["phase"] = "filing"
        return payload

    def _run_filing_batch(
        self,
        batch_changelog: list[dict[str, Any]],
        batch_inbox: list[dict[str, Any]],
    ) -> dict[str, Any]:
        layout = {}
        if self.db.paths.resolve("Folderrules.json").exists():
            raw = self.db.read_json("Folderrules.json")["data"]
            if isinstance(raw, dict):
                layout = {
                    "domains": list((raw.get("domains") or {}).keys())
                    if isinstance(raw.get("domains"), dict)
                    else raw.get("domains"),
                    "create_under": (raw.get("ai_may_create") or {}).get("under"),
                    "course_cop": "School/Courses/<Code>",
                    "project_cop": "Work/Projects/<Name>",
                }
        payload = self._build_filing_payload(batch_changelog, batch_inbox, layout=layout)

        reply, err, stats = self._ask_soi(payload)
        retries = 0
        if err == "cancelled" or stats.get("cancelled"):
            return {
                "ok": True,
                "cancelled": True,
                "processed_changelog": len(batch_changelog),
                "mutating_calls": stats.get("mutating_calls") or [],
            }
        if err:
            return {"ok": False, "error": err, "processed_changelog": len(batch_changelog)}

        # Do not re-ask. Qwen often reprints the same fake upsert JSON instead of
        # calling tools; host safety nets below file lasting content once.

        narrated: list[dict[str, Any]] = []

        if err and not stats.get("mutating_calls"):
            return {
                "ok": False,
                "error": err,
                "processed_changelog": len(batch_changelog),
                "retries": retries,
            }

        entry_ids = [str(e["id"]) for e in batch_changelog if e.get("id")]
        discarded_ids = self._parse_id_list(reply, entry_ids, keys=("discarded", "discarded_ids"))
        claimed_filed = self._parse_id_list(reply, entry_ids, keys=("filed", "filed_ids"))
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
            if call.get("tool") not in {"file_by_id", "file_note"}:
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
                    # Honor successful discard tool calls — model chose discard.
                    handled_by_id.add(item)
                    discarded_ids.add(item)
                    dest_by_id[item] = ""
                    continue
                dest_path = str(
                    result.get("folder")
                    or result.get("filed_to")
                    or result.get("path")
                    or result.get("notes_path")
                    or args.get("dest")
                    or ""
                ).replace("\\", "/")
                if "inbox" in dest or "inbox/" in dest_path.lower():
                    continue
                handled_by_id.add(item)
                claimed_filed.add(item)
                _add_dest(dest_by_id, item, dest_path)

        domain_plans: set[str] = set()
        for call in stats.get("mutating_calls") or []:
            if call.get("tool") not in {"create_cop", "create_folder", "write_json", "patch_json"}:
                continue
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            path = str(args.get("path") or args.get("folder_path") or "").replace("\\", "/")
            if not path.startswith(("School/", "Work/", "Household/")):
                continue
            for entry in batch_changelog:
                eid = str(entry.get("id") or "")
                if not eid or eid in handled_by_id or eid in discarded_ids:
                    continue
                if _is_ephemeral_entry(entry):
                    continue
                text = _entry_user_text(entry)
                if ("/Courses/" in path or "/Projects/" in path) and not cop_name_in_text(
                    path, text
                ):
                    continue
                handled_by_id.add(eid)
                _add_dest(dest_by_id, eid, path)
                domain = path.split("/", 1)[0]
                domain_plans.add(f"{domain}/Plan.json")
        for entry in batch_changelog:
            eid = str(entry.get("id") or "")
            if not eid or eid not in handled_by_id:
                continue
            text = _entry_user_text(entry)
            if not text:
                continue
            for plan in domain_plans:
                if not self.db.paths.resolve(plan).exists():
                    continue
                data = self.db.read_json(plan)["data"]
                if not isinstance(data, dict):
                    continue
                plans = data.get("plans") if isinstance(data.get("plans"), list) else []
                if not any(isinstance(row, dict) and row.get("id") == eid for row in plans):
                    plans.append(
                        {
                            "id": eid,
                            "objective": text,
                            "status": "active",
                            "last_updated": _utc_now(),
                        }
                    )
                    data["plans"] = plans
                    data["last_updated"] = _utc_now()
                    self.db.write_json(plan, data, summary=f"File oac_turn {eid} into {plan}")
                _add_dest(dest_by_id, eid, plan)

        # Leftovers stay pending — host does not regex-dump them into folders.
        host_inbox = self._host_file_inbox(batch_inbox)

        # Host-auto discard ephemeral if model forgot. Do not auto-file leftovers.
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
            "seen_inbox": len(batch_inbox),
            "host_general_filed": list(host_filed_ids),
            "host_inbox_filed": host_inbox.get("filed") or 0,
            "narrated_tools_applied": len(narrated),
            "mutating_tool_calls": len(stats.get("mutating_calls") or []),
            "tool_names": stats.get("tool_names") or [],
            "retries": retries,
            "reply": reply,
        }

    def _domain_snapshot(self) -> dict[str, Any]:
        """Short path lists only — full Read.json makes Qwen write essays."""
        snap: dict[str, Any] = {}
        for domain in ("School", "Work", "Household"):
            item: dict[str, Any] = {"children": []}
            if self.db.paths.resolve(domain).exists():
                try:
                    listing = self.db.list_dir(domain)
                    item["children"] = [
                        str(c.get("path") or c.get("name") or "")
                        for c in (listing.get("children") or [])
                        if isinstance(c, dict)
                    ]
                except Exception:
                    item["children"] = []
            extra = f"{domain}/Courses" if domain == "School" else f"{domain}/Projects"
            if self.db.paths.resolve(extra).exists():
                try:
                    listing = self.db.list_dir(extra)
                    item["cops"] = [
                        str(c.get("path") or c.get("name") or "")
                        for c in (listing.get("children") or [])
                        if isinstance(c, dict)
                    ]
                except Exception:
                    item["cops"] = []
            else:
                item["cops"] = []
            snap[domain] = item
        return snap

    def _ids_placed_by_file_by_id(
        self,
        stats: dict[str, Any],
        batch_changelog: list[dict[str, Any]],
    ) -> set[str]:
        known = {str(e.get("id") or "") for e in batch_changelog if e.get("id")}
        placed: set[str] = set()
        for call in stats.get("mutating_calls") or []:
            if call.get("tool") != "file_by_id" or call.get("ok") is False:
                continue
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            result = call.get("result") if isinstance(call.get("result"), dict) else {}
            dest = str(args.get("dest") or result.get("action") or "").replace("\\", "/").lower()
            if dest in {"discard", "ephemeral", "drop"} or "inbox" in dest:
                continue
            ids = [str(x) for x in (args.get("entry_ids") or result.get("entry_ids") or []) if x]
            eid = str(args.get("entry_id") or args.get("id") or "").strip()
            if eid:
                ids.append(eid)
            for item in ids:
                if item in known:
                    placed.add(item)
        for call in stats.get("mutating_calls") or []:
            if call.get("ok") is False:
                continue
            if call.get("tool") in {"create_cop", "create_folder", "write_json", "patch_json"}:
                # Domain tools ran — still need file_by_id or host leftover for ids.
                continue
        return placed


    def run_once(self) -> dict[str, Any]:
        """Back-compat: run filing phase."""
        return self.run_filing()

    # ---- phase 2: Read.json refresh ----------------------------------------

    def list_read_json_paths(self) -> list[str]:
        root = Path(self.config.db_root)
        paths: list[str] = []
        for path in sorted(root.rglob("Read.json")):
            if "runtime" in path.parts:
                continue
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                continue
            paths.append(rel)
        return paths

    def reads_by_domain(self, paths: list[str] | None = None) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for rel in paths if paths is not None else self.list_read_json_paths():
            top = rel.split("/", 1)[0] if "/" in rel else rel
            grouped.setdefault(top, []).append(rel)
        return grouped

    def run_read_refresh(self) -> dict[str, Any]:
        if self.has_filing_work():
            self.log.log("read_refresh_skip", reason="filing still pending")
            return {
                "ok": True,
                "ran": False,
                "phase": "read_refresh",
                "reason": "filing still pending",
            }

        stale = self.list_stale_read_paths()
        if not stale:
            self._merge_state(
                {
                    "status": "idle",
                    "phase": "read_refresh",
                    "needs_read_refresh": False,
                    "reason": "no stale Reads",
                }
            )
            self.log.log("read_refresh_skip", reason="no stale Reads (needs_update=false)")
            return {
                "ok": True,
                "ran": False,
                "phase": "read_refresh",
                "reason": "no stale Reads (needs_update=false)",
            }

        folder_payloads = build_read_refresh_folders(self.db)
        if not folder_payloads:
            self._merge_state(
                {
                    "status": "idle",
                    "phase": "read_refresh",
                    "needs_read_refresh": False,
                    "reason": "no stale folders",
                }
            )
            self.log.log("read_refresh_skip", reason="no stale folders")
            return {
                "ok": True,
                "ran": False,
                "phase": "read_refresh",
                "reason": "no stale folders",
            }

        domains = sorted({str(fp.get("folder") or "").split("/", 1)[0] for fp in folder_payloads if fp.get("folder")})
        self.log.log(
            "read_refresh_start",
            stale_count=len(stale),
            domains=domains,
            stale=stale,
        )
        replies: dict[str, str] = {}
        errors: dict[str, str] = {}
        refreshed: list[str] = []

        for fp in folder_payloads:
            folder = str(fp.get("folder") or "").strip()
            if not folder:
                continue
            read_path = f"{folder}/Read.json"
            notes_path = f"{folder}/Notes.json"
            hist_path = f"{folder}/History.json"
            new_entries = int(fp.get("new_entries_since_last_refresh") or 0)
            domain = folder.split("/", 1)[0]

            try:
                before_read = self.db.read_json(read_path)["data"]
            except (OSError, ValueError, KeyError):
                before_read = {}
            try:
                before_notes = self.db.read_json(notes_path)["data"]
            except (OSError, ValueError, KeyError):
                before_notes = None
            try:
                before_hist = self.db.read_json(hist_path)["data"]
            except (OSError, ValueError, KeyError):
                before_hist = None

            payload = {**fp, "phase": "read_refresh", "domain": domain}
            reply, err, stats = self._ask_soi(payload)
            if err:
                errors[folder] = err
                continue

            replies[folder] = (reply or "")[:400]
            mut_count = len(stats.get("mutating_calls") or [])
            if mut_count == 0:
                errors[folder] = "model produced no mutating tool calls"
                continue

            try:
                after_read = self.db.read_json(read_path)["data"]
            except (OSError, ValueError, KeyError):
                after_read = {}
            try:
                after_notes = self.db.read_json(notes_path)["data"]
            except (OSError, ValueError, KeyError):
                after_notes = None
            try:
                after_hist = self.db.read_json(hist_path)["data"]
            except (OSError, ValueError, KeyError):
                after_hist = None

            digest_keys = (
                "summary",
                "state",
                "important_context",
                "recent_changes",
                "active_items",
                "known_facts",
                "uncertainties",
            )
            digest_changed = any(
                (before_read.get(k) if isinstance(before_read, dict) else None)
                != (after_read.get(k) if isinstance(after_read, dict) else None)
                for k in digest_keys
            )
            notes_changed = before_notes != after_notes
            hist_changed = before_hist != after_hist
            cleanup_changed = notes_changed or hist_changed

            if new_entries > 0 and not (digest_changed or cleanup_changed):
                errors[folder] = (
                    "no Read digest update detected despite new entries; "
                    "mark_read_refreshed-only run rejected"
                )
                continue

            try:
                still = self.db.read_json(read_path)["data"]
                if readlog.read_needs_refresh(still):
                    self.db.mark_read_refreshed(read_path)
            except (OSError, ValueError, KeyError):
                pass
            refreshed.append(read_path)

        self._host_observe_voice_from_masterlog()

        remaining = self.list_stale_read_paths()
        ok = not errors
        self._merge_state(
            {
                "status": "ok" if ok else "partial_error",
                "phase": "read_refresh",
                "needs_read_refresh": bool(remaining) or bool(errors),
                "last_read_refresh_at": _utc_now(),
                "read_domains": domains,
                "stale_before": stale,
                "refreshed": refreshed,
                "stale_remaining": remaining,
                "read_errors": errors or None,
                "reply_previews": replies,
            }
        )
        out = {
            "ok": ok,
            "ran": True,
            "phase": "read_refresh",
            "domains": domains,
            "stale": stale,
            "refreshed": refreshed,
            "stale_remaining": remaining,
            "errors": errors,
            "replies": replies,
        }
        self.log.log(
            "read_refresh_done",
            level="error" if not ok else "info",
            refreshed=refreshed,
            stale_remaining=remaining,
            errors=errors or None,
        )
        return out

    # ---- helpers -----------------------------------------------------------

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
        phase_for_mode = str(payload.get("phase") or "filing")
        if phase_for_mode == "read_refresh":
            mode_id = "soi_test_p2"
        elif phase_for_mode == "filing":
            mode_id = "soi_test"
        else:
            mode_id = "soi"
        session = ChatSession(
            mode=get_mode(mode_id),
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
                for key in ("query", "path", "url", "q", "dest", "entry_id", "title", "slug"):
                    if key in args_obj and args_obj[key]:
                        hint = f" {key}={args_obj[key]!r}"
                        break
                if not hint and args_obj:
                    raw = json.dumps(args_obj, ensure_ascii=False)
                    hint = f" {raw[:100]}{'...' if len(raw) > 100 else ''}"
                self.log.log("tool_start", name=name, hint=hint, arguments=args_obj)
            elif phase == "done":
                self.log.log(
                    "tool_done",
                    name=name,
                    ok=bool(detail.get("ok", True)),
                    summary=detail.get("summary") or "",
                )

        phase = payload.get("phase") or "soi"
        # Always stream so Stop/Reset can interrupt; soi_think still controls /think.
        stream = True
        self.log.log(
            "model_ask",
            phase=phase,
            domain=payload.get("domain"),
            timeout_s=self.config.soi_timeout_s,
            think=self.config.soi_think,
            stream=stream,
        )
        try:
            if phase == "filing":
                user_text = format_test_user_message(FILING_INSTRUCTIONS, payload)
                reply = session.ask(
                    user_text,
                    stream=stream,
                    on_thinking=None,
                    on_tool=on_tool,
                )
            elif phase == "read_refresh":
                user_text = format_read_refresh_message(READ_REFRESH_INSTRUCTIONS, payload)
                reply = session.ask(
                    user_text,
                    stream=stream,
                    on_thinking=None,
                    on_tool=on_tool,
                )
            else:
                reply = session.ask(
                    "SOI job — process this batch with TOOL CALLS (mutations required):\n"
                    + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    stream=stream,
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
            self.log.log("model_error", level="error", phase=phase, error=str(exc))
            return None, str(exc), stats
        finally:
            self._active_session = None

    def _execute_narrated_tool_plan(self, reply: str) -> list[dict[str, Any]]:
        """Disabled: models echo changelog rows as fake upsert JSON and loop."""
        return []

        plans: list[tuple[str, dict[str, Any]]] = []
        for key in ("tool_calls", "actions", "tools"):
            raw = blob.get(key)
            if isinstance(raw, list):
                for item in raw:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("tool") or item.get("name") or "").strip()
                    args = item.get("params") or item.get("arguments") or item.get("args") or {}
                    if name and isinstance(args, dict):
                        plans.append((name, args))

        # Shape: {"patch_json": {"Hayden/Preferences/Food.json": {...}}}
        patch_map = blob.get("patch_json")
        if isinstance(patch_map, dict):
            for path, patch in patch_map.items():
                if isinstance(path, str) and isinstance(patch, (dict, list)):
                    plans.append(("patch_json", {"path": path, "patch": patch if isinstance(patch, dict) else {"items": patch}}))

        write_map = blob.get("write_json")
        if isinstance(write_map, dict) and "path" in write_map:
            plans.append(("write_json", write_map))

        applied: list[dict[str, Any]] = []
        allowed = {
            "write_json",
            "create_json",
            "patch_json",
            "set_json_path",
            "create_folder",
            "create_cop",
            "create_project",
            "capture_inbox",
            "mark_read_stale",
            "mark_read_refreshed",
        }
        for name, args in plans[:20]:
            if name not in allowed:
                continue
            # Normalize common narrated aliases
            if name == "write_json" and "file" in args and "path" not in args:
                args = {**args, "path": args.get("file")}
            if name == "patch_json" and "file" in args and "path" not in args:
                args = {**args, "path": args.get("file")}
            # Prefer Hayden-qualified paths when model omits domain root for prefs/inbox.
            path = args.get("path")
            if isinstance(path, str) and path and not path.startswith(
                ("Hayden/", "School/", "Work/", "Household/", "runtime/")
            ):
                if path.startswith(("Preferences/", "Habits/", "Inbox/", "Relationships/")):
                    args = {**args, "path": f"Hayden/{path}"}
            try:
                result = dispatch(self.db, name, args)
            except Exception as exc:  # noqa: BLE001 — keep filing batch alive
                result = {"ok": False, "error": str(exc)}
            applied.append({"tool": name, "args": args, "ok": bool(result.get("ok", True)), "result": result})
        return applied

    def _host_append_note(self, path: str, *, field: str, note: str, summary: str) -> bool:
        """Append a short note string into a list field on an existing JSON object leaf."""
        if not self.db.paths.resolve(path).exists():
            return False
        data = self.db.read_json(path)["data"]
        if not isinstance(data, dict):
            return False
        bucket = data.get(field)
        if bucket is None:
            # Prefer a sensible existing list field
            for candidate in ("notes", "likes", "environments", "routines_i_like", "favorites", "items", "goals", "active"):
                if isinstance(data.get(candidate), list):
                    field = candidate
                    bucket = data[candidate]
                    break
            else:
                data["notes"] = []
                field = "notes"
                bucket = data["notes"]
        if not isinstance(bucket, list):
            return False
        if note not in bucket:
            bucket.append(note)
        data[field] = bucket
        data["last_updated"] = _utc_now()
        self.db.write_json(path, data, summary=summary)
        return True

    def _host_observe_voice_from_masterlog(self) -> dict[str, Any]:
        """Phase 2: speech/tone/swearing evidence from how Hayden actually prompted."""
        path = "Masterlog.json"
        if not self.db.paths.resolve(path).exists():
            return {"ok": True, "observed": 0}
        data = self.db.read_json(path)["data"]
        entries = data.get("entries") if isinstance(data, dict) else []
        if not isinstance(entries, list):
            return {"ok": True, "observed": 0}
        users: list[str] = []
        for entry in entries[-40:]:
            if not isinstance(entry, dict):
                continue
            if entry.get("action") != "oac_turn":
                continue
            details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
            text = str(details.get("user_text") or entry.get("summary") or "").strip()
            if text:
                users.append(text)
        if not users:
            return {"ok": True, "observed": 0}
        from ollama.file_by_id import _append_voice_evidence

        for text in users[-12:]:
            _append_voice_evidence(self.db, text)
        voice_path = "Hayden/Identity/Voice.json"
        if self.db.paths.resolve(voice_path).exists():
            voice = self.db.read_json(voice_path)["data"]
            if isinstance(voice, dict):
                q = sum(1 for t in users if "?" in t)
                if q >= 3 and not str(voice.get("directness") or "").strip():
                    voice["directness"] = "asks blunt factual questions in short spoken bursts"
                greet = sum(1 for t in users if re.match(r"^\s*(hi|hey|hello)\b", t, re.I))
                if greet and not str(voice.get("humor") or "").strip():
                    voice["humor"] = "casual buddy greetings (hi bud / hey pal) then jumps into the question"
                voice["last_updated"] = _utc_now()
                self.db.write_json(voice_path, voice, summary="Phase 2 voice observation from Masterlog")
        return {"ok": True, "observed": len(users)}

    def _host_file_inbox(self, batch_inbox: list[dict[str, Any]]) -> dict[str, Any]:
        """File unfiled captures that already have a suggested_home."""
        if not batch_inbox:
            return {"filed": 0}
        path = "Hayden/Inbox/Captures.json"
        if not self.db.paths.resolve(path).exists():
            return {"filed": 0}
        data = self.db.read_json(path)["data"]
        if not isinstance(data, dict):
            return {"filed": 0}
        captures = data.get("captures") if isinstance(data.get("captures"), list) else []
        wanted = {str(c.get("id")) for c in batch_inbox if isinstance(c, dict) and c.get("id")}
        filed = 0
        for cap in captures:
            if not isinstance(cap, dict):
                continue
            cid = str(cap.get("id") or "")
            if cid not in wanted or cap.get("status") != "unfiled":
                continue
            home = str(cap.get("suggested_home") or "").strip()
            text = str(cap.get("text") or "").strip()
            if not home or not text:
                continue
            dest = home if home.startswith(("Hayden/", "School/", "Work/", "Household/")) else f"Hayden/{home}"
            # Directory suggestions → skip leaf write; still mark filed_to pointer.
            wrote = False
            if dest.endswith(".json") and self.db.paths.resolve(dest).exists():
                # Prefer list-ish fields
                wrote = self._host_append_note(
                    dest,
                    field="notes",
                    note=text,
                    summary="Host-file inbox capture into suggested_home",
                )
            cap["status"] = "filed"
            cap["filed_to"] = dest
            filed += 1
            if not wrote and dest.endswith(".json"):
                # Leaf missing — leave filed_to as intent pointer anyway.
                pass
        if filed:
            data["captures"] = captures
            data["last_updated"] = _utc_now()
            self.db.write_json(path, data, summary=f"Host-file {filed} inbox captures")
        return {"filed": filed}

    def _parse_id_list(
        self,
        reply: str | None,
        known_ids: list[str],
        *,
        keys: tuple[str, ...],
    ) -> set[str]:
        """Extract real changelog entry ids listed under the given JSON keys."""
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
                            # Allow "id — note" forms; take leading id-like token.
                            token = token.split()[0].strip(" ,;:") if token else ""
                            if token in known:
                                found.add(token)
        # Also accept bare known ids near the key name (conservative: JSON path above preferred).
        return found

    def _parse_discarded_ids(self, reply: str | None, known_ids: list[str]) -> set[str]:
        """Back-compat wrapper."""
        return self._parse_id_list(reply, known_ids, keys=("discarded", "discarded_ids"))

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
