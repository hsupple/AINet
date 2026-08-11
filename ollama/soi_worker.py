"""SOI worker — Phase 1 filing (changelog/inbox), Phase 2 Read.json refresh."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ainet.tools import changelog
from ainet.tools import readlog
from ainet.tools.fsutil import atomic_write_text
from ainet.tools.ops import DatabaseTools
from ainet.tools.registry import dispatch
from ollama.client import OllamaClient, OllamaError
from ollama.config import OllamaConfig
from ollama.modes import get_mode
from ollama.research_sessions import INDEX_PATH, upsert_research_session
from ollama.content_filing import content_kind, entry_kind, is_ephemeral_text, topic_title_from_text
from ollama.session import ChatSession, _guess_topic_title
from ollama.soi_log import SOILogger
from ollama.topics import (
    ensure_topic,
    latest_open_research_subject,
    record_personal_filing,
    record_topic_filing,
    slugify_topic,
)

_FILING_BATCH_SIZE = 4
_SOI_MIN_TOOL_ROUNDS = 24
_SOI_MAX_HISTORY = 40


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _entry_user_text(entry: dict[str, Any]) -> str:
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    return str(details.get("user_text") or entry.get("summary") or "").strip()


def _entry_for_soi(entry: dict[str, Any]) -> dict[str, Any]:
    """Full turn text so SOI can choose tools. id is for file_by_id copies."""
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    return {
        "id": entry.get("id"),
        "summary": entry.get("summary"),
        "suggested_filing": entry.get("suggested_filing"),
        "user_text": str(details.get("user_text") or entry.get("summary") or "").strip(),
        "assistant_text": str(details.get("assistant_text") or "").strip(),
    }


def _inbox_for_soi(cap: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": cap.get("id"),
        "suggested_home": cap.get("suggested_home") or "",
        "text": str(cap.get("text") or "").strip(),
    }


def _is_ephemeral_entry(entry: dict[str, Any]) -> bool:
    return is_ephemeral_text(_entry_user_text(entry))


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
            "host_research_sessions": [],
            "mutating_tool_calls": 0,
            "batches": 0,
            "retries": 0,
            "replies": [],
        }
        errors: list[str] = []
        seen_ids: set[str] = set()
        # Process small batches so the model can actually call tools per item.
        while True:
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
            if result.get("error"):
                errors.append(str(result["error"]))
            totals["processed_changelog"] += int(result.get("processed_changelog") or 0)
            totals["marked_filed"] += int(result.get("marked_filed") or 0)
            totals["marked_discarded"] += int(result.get("marked_discarded") or 0)
            totals["left_pending"] += int(result.get("left_pending") or 0)
            totals["seen_inbox"] += int(result.get("seen_inbox") or 0)
            totals["mutating_tool_calls"] += int(result.get("mutating_tool_calls") or 0)
            totals["retries"] += int(result.get("retries") or 0)
            totals["host_research_sessions"].extend(result.get("host_research_sessions") or [])
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
                "status": "ok" if ok else "error",
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
                "host_research_sessions": totals["host_research_sessions"],
                "reply_preview": (totals["replies"][-1] if totals["replies"] else "")[:400],
                "errors": errors or None,
            }
        )
        out = {
            "ok": ok,
            "ran": True,
            "phase": "filing",
            "processed_changelog": totals["processed_changelog"],
            "marked_filed": totals["marked_filed"],
            "marked_discarded": totals["marked_discarded"],
            "left_pending": totals["left_pending"],
            "seen_inbox": totals["seen_inbox"],
            "mutating_tool_calls": totals["mutating_tool_calls"],
            "batches": totals["batches"],
            "retries": totals["retries"],
            "host_research_sessions": totals["host_research_sessions"],
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
            errors=errors or None,
        )
        return out

    def _run_filing_batch(
        self,
        batch_changelog: list[dict[str, Any]],
        batch_inbox: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self._annotate_suggested_filing(batch_changelog)
        layout = {}
        if self.db.paths.resolve("Folderrules.json").exists():
            raw = self.db.read_json("Folderrules.json")["data"]
            if isinstance(raw, dict):
                layout = raw
        payload = {
            "phase": "filing",
            "folderrules": layout,
            "changelog_entries": [_entry_for_soi(e) for e in batch_changelog],
            "inbox_unfiled": [_inbox_for_soi(c) for c in batch_inbox],
            "instructions": (
                "You have the full user_text/assistant_text for each id, and Folderrules. "
                "Use create_cop / create_folder / write_json / patch_json / file_by_id to put "
                "content where that map already says it belongs. "
                "file_by_id copies stored text by id — do not retype bodies. "
                "One turn may need several tool calls. Do not invent domains outside Folderrules."
            ),
        }

        reply, err, stats = self._ask_soi(payload)
        retries = 0
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
        dest_by_id: dict[str, str] = {}
        by_entry = {str(e.get("id") or ""): e for e in batch_changelog}
        discarded_ids = {
            eid
            for eid in discarded_ids
            if (by_entry.get(eid) and _is_ephemeral_entry(by_entry[eid]))
        }
        for call in stats.get("mutating_calls") or []:
            if call.get("tool") != "file_by_id":
                continue
            if call.get("ok") is False:
                continue
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            result = call.get("result") if isinstance(call.get("result"), dict) else {}
            dest = str(args.get("dest") or result.get("action") or "").strip().lower()
            ids = [str(x) for x in (args.get("entry_ids") or result.get("entry_ids") or []) if x]
            eid = str(args.get("entry_id") or "").strip()
            if eid:
                ids.append(eid)
            for item in ids:
                if item not in known:
                    continue
                if dest in {"discard", "ephemeral", "drop"}:
                    src = by_entry.get(item)
                    if src and _is_ephemeral_entry(src):
                        handled_by_id.add(item)
                        discarded_ids.add(item)
                        dest_by_id[item] = ""
                    # Lasting content: ignore model discard so host can store it.
                    continue
                handled_by_id.add(item)
                claimed_filed.add(item)
                dest_by_id[item] = str(
                    result.get("filed_to")
                    or result.get("path")
                    or args.get("dest")
                    or ""
                )

        # Host safety nets only for ids SOI did not already place by id.
        leftover = [
            e for e in batch_changelog if str(e.get("id") or "") not in handled_by_id
        ]
        host_research = self._host_file_research_turns(leftover)
        if isinstance(host_research, dict):
            host_sessions = list(host_research.get("sessions") or [])
            dest_by_id.update(host_research.get("dest_by_id") or {})
        else:
            host_sessions = list(host_research or [])
        host_general = self._host_file_general_turns(leftover)
        dest_by_id.update(host_general.get("dest_by_id") or {})
        host_personal = self._host_file_personal_turns(leftover)
        dest_by_id.update(host_personal.get("dest_by_id") or {})
        host_inbox = self._host_file_inbox(batch_inbox)

        # Host-auto discard ephemeral if model forgot.
        for entry in batch_changelog:
            eid = str(entry.get("id") or "")
            if eid and _is_ephemeral_entry(entry):
                discarded_ids.add(eid)

        # Host general filing marks those entry ids as evidenced.
        host_filed_ids = {str(x) for x in (host_general.get("filed_ids") or []) if x}
        host_filed_ids.update(str(x) for x in (host_personal.get("filed_ids") or []) if x)
        host_filed_ids.update(k for k, v in dest_by_id.items() if v)

        had_mutations = bool(stats.get("mutating_calls"))
        filed_ids: list[str] = []
        left_pending: list[str] = []
        for entry in batch_changelog:
            eid = str(entry.get("id") or "")
            if not eid:
                continue
            if eid in discarded_ids:
                continue
            in_session = self._already_in_research_session(eid)
            if in_session or eid in host_filed_ids:
                filed_ids.append(eid)
                continue
            if had_mutations and eid in claimed_filed:
                filed_ids.append(eid)
                continue
            if entry.get("suggested_filing") == "research" and self._already_in_research_session(eid):
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
            "host_research_sessions": host_sessions,
            "host_general_filed": list(host_filed_ids),
            "host_inbox_filed": host_inbox.get("filed") or 0,
            "narrated_tools_applied": len(narrated),
            "mutating_tool_calls": len(stats.get("mutating_calls") or []),
            "tool_names": stats.get("tool_names") or [],
            "retries": retries,
            "reply": reply,
        }

    def _annotate_suggested_filing(self, batch_changelog: list[dict[str, Any]]) -> None:
        for entry in batch_changelog:
            entry["suggested_filing"] = entry_kind(entry)

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

        grouped = self.reads_by_domain(stale)
        self.log.log(
            "read_refresh_start",
            stale_count=len(stale),
            domains=list(grouped.keys()),
            stale=stale,
        )
        replies: dict[str, str] = {}
        errors: dict[str, str] = {}
        refreshed: list[str] = []

        size_rules = (
            "Read.json is a SHORT hot index only — never a folder dump. "
            "Caps: summary≤400 chars, state≤160, items≤180 chars each; "
            "important_context≤12, active_items≤10, recent_changes≤8, "
            "known_facts≤12, uncertainties≤8; whole file ≤~12KB. "
            "Prefer path pointers to sibling leaves over inlining detail. "
            "Roll excess into the correct leaf files or History before/while refreshing. "
            "Use pending read_changelog entries as the change digest."
        )

        for domain, reads in grouped.items():
            payload = {
                "phase": "read_refresh",
                "domain": domain,
                "read_paths": reads,
                "instructions": (
                    "You are SOI Phase 2 (Read refresh). "
                    "ONLY the listed Read.json paths need updates (needs_update=true / pending read_changelog). "
                    "For each, inspect that COP's hot surface and nearby Profile/Plan/History/Notes only as needed. "
                    "Rewrite/patch the Read so summary/state/important_context/active_items/"
                    "known_facts/recent_changes stay a lean digest of newest relevant info. "
                    f"{size_rules} "
                    "Do not dump unrelated domains. Prefer patch_json. "
                    "Do NOT dump read_changelog into Research Notes.json — Notes/History are filled at filing time. "
                    "Also observe Hayden's speech from recent Masterlog turns: curse rate, tone, "
                    "buddy greetings, how questions are asked. Patch Hayden/Identity/Voice.json and "
                    "Personality.json with evidence only (no invented traits). "
                    "After each successful Read rewrite, call mark_read_refreshed on that Read path "
                    "(or folder) so needs_update=false and pending changelog entries are consumed."
                ),
            }
            reply, err, _stats = self._ask_soi(payload)
            if err:
                errors[domain] = err
            else:
                replies[domain] = (reply or "")[:400]
                # Host safety net: clear gate for paths still marked stale after a successful run.
                for rel in reads:
                    try:
                        still = self.db.read_json(rel)["data"]
                    except (OSError, ValueError, KeyError):
                        continue
                    if readlog.read_needs_refresh(still):
                        try:
                            self.db.mark_read_refreshed(rel)
                            refreshed.append(rel)
                        except (OSError, ValueError) as exc:
                            errors[domain] = (
                                (errors.get(domain) or "") + f" mark_read_refreshed({rel}): {exc}"
                            ).strip()
                    else:
                        refreshed.append(rel)

        self._host_observe_voice_from_masterlog()

        remaining = self.list_stale_read_paths()
        ok = not errors
        self._merge_state(
            {
                "status": "ok" if ok else "partial_error",
                "phase": "read_refresh",
                "needs_read_refresh": bool(remaining) or bool(errors),
                "last_read_refresh_at": _utc_now(),
                "read_domains": list(grouped.keys()),
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
            "domains": list(grouped.keys()),
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
        session = ChatSession(
            mode=get_mode("soi"),
            config=soi_config,
            client=self.client,
            auto_mode=False,
            persist_conversation=False,
        )
        stats: dict[str, Any] = {"mutating_calls": [], "tool_names": [], "tool_rounds": 0}

        def on_tool(phase: str, name: str, detail: dict[str, Any]) -> None:
            if phase == "start":
                args_obj = detail.get("arguments") or {}
                hint = ""
                for key in ("query", "path", "url", "q", "title", "slug"):
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
            reply = session.ask(
                "SOI job — process this batch with TOOL CALLS (mutations required):\n"
                + json.dumps(payload, ensure_ascii=False, indent=2),
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
                preview=(reply or "")[:240],
                mutating_calls=len(stats["mutating_calls"]),
            )
            return reply, None, stats
        except OllamaError as exc:
            self.log.log("model_error", level="error", phase=phase, error=str(exc))
            return None, str(exc), stats

    def _already_in_research_session(self, entry_id: str) -> bool:
        if not entry_id or not self.db.paths.resolve(INDEX_PATH).exists():
            return False
        index = self.db.read_json(INDEX_PATH)["data"]
        sessions = index.get("sessions") if isinstance(index, dict) else None
        if not isinstance(sessions, list):
            return False
        for row in sessions:
            if not isinstance(row, dict):
                continue
            path = str(row.get("path") or "")
            if not path or not self.db.paths.resolve(path).exists():
                continue
            data = self.db.read_json(path)["data"]
            ids = data.get("changelog_entry_ids") if isinstance(data, dict) else None
            if isinstance(ids, list) and entry_id in ids:
                return True
        return False

    def _host_file_research_turns(self, batch_changelog: list[dict[str, Any]]) -> dict[str, Any]:
        """If SOI skipped upsert_research_session, file research-hinted turns here."""
        created: list[str] = []
        dest_by_id: dict[str, str] = {}
        groups: dict[str, list[dict[str, Any]]] = {}
        open_subject = latest_open_research_subject(self.db)
        for entry in batch_changelog:
            kind = entry.get("suggested_filing") or entry_kind(entry)
            if kind != "research":
                continue
            if _is_ephemeral_entry(entry):
                continue
            user_text = _entry_user_text(entry)
            if not user_text:
                continue
            eid = str(entry.get("id") or "")
            if eid and self._already_in_research_session(eid):
                continue
            title = topic_title_from_text(user_text) or _guess_topic_title(user_text)
            if (not title or content_kind(user_text) == "research") and open_subject:
                if not title or len(user_text) < 80:
                    title = open_subject
            if not title:
                continue
            groups.setdefault(title, []).append(entry)
            open_subject = title

        for title, entries in groups.items():
            ensure_topic(self.db, title)
            slug = slugify_topic(title)
            details_covered: list[dict[str, str]] = []
            entry_ids: list[str] = []
            for entry in entries:
                details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
                user_text = _entry_user_text(entry)
                assistant_text = str(details.get("assistant_text") or "").strip()
                eid = str(entry.get("id") or "")
                if eid:
                    entry_ids.append(eid)
                if user_text:
                    details_covered.append({"kind": "qa", "text": f"Q: {user_text}"})
                if assistant_text:
                    clip = assistant_text if len(assistant_text) <= 900 else assistant_text[:900] + "…"
                    details_covered.append({"kind": "mechanism", "text": clip})
            result = upsert_research_session(
                self.db,
                subject=title,
                title=title,
                topic_slug=slug,
                details_covered=details_covered,
                length_turns=len(entries),
                changelog_entry_ids=entry_ids,
                status="open",
                summary=f"Host-filed research session for {title}",
            )
            if result.get("ok") is False:
                continue
            session_obj = result.get("session") if isinstance(result.get("session"), dict) else {}
            sid = str(
                result.get("session_id")
                or result.get("id")
                or session_obj.get("id")
                or ""
            )
            if sid:
                created.append(sid)
            session_path = str(result.get("path") or "")
            for eid in entry_ids:
                if eid:
                    dest_by_id[eid] = session_path or f"Hayden/Research/Sessions/{sid}.json"
            for entry in entries:
                details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
                record_topic_filing(
                    self.db,
                    slug,
                    title,
                    user=_entry_user_text(entry),
                    assistant=str(details.get("assistant_text") or ""),
                    entry_ids=[str(entry.get("id") or "")] if entry.get("id") else [],
                )
        return {"sessions": created, "dest_by_id": dest_by_id}

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
            "capture_inbox",
            "upsert_research_session",
            "complete_research_session",
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
                if path.startswith(("Preferences/", "Habits/", "Inbox/", "Research/", "Relationships/")):
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

    def _host_file_general_turns(self, batch_changelog: list[dict[str, Any]]) -> dict[str, Any]:
        """Deterministic leaf filing when SOI skips non-research lasting turns."""
        filed_ids: list[str] = []
        actions: list[str] = []
        dest_by_id: dict[str, str] = {}
        for entry in batch_changelog:
            if entry.get("suggested_filing") in {"research", "discard"}:
                continue
            if _is_ephemeral_entry(entry):
                continue
            eid = str(entry.get("id") or "")
            if not eid:
                continue
            text = _entry_user_text(entry)
            low = text.lower()
            filed = False
            dest_path = ""

            if re.search(r"\b(coffee shop|outlet table|library|deep-?work spot|place)\b", low):
                dest_path = "Hayden/Preferences/Lifestyle.json"
                filed = self._host_append_note(
                    dest_path,
                    field="environments",
                    note=text,
                    summary="Host-file place/lifestyle pref from oac_turn",
                )
            elif re.search(r"\b(matcha|espresso|latte|drip coffee|food|ramen|eat)\b", low):
                dest_path = "Hayden/Preferences/Food.json"
                filed = self._host_append_note(
                    dest_path,
                    field="likes",
                    note=text,
                    summary="Host-file food pref from oac_turn",
                )
            elif re.search(r"\b(focus block|pomodoro|stretch|habit|routine|every afternoon)\b", low):
                dest_path = "Hayden/Habits/Routines.json"
                filed = self._host_append_note(
                    dest_path,
                    field="rituals",
                    note=text,
                    summary="Host-file habit from oac_turn",
                )
            elif re.search(r"\b(met|makerspace)\b", low):
                m = re.search(r"\b(?:Met|met)\s+([A-Z][a-z]+)\b", text)
                if not m:
                    m = re.search(r"\b([A-Z][a-z]{2,})\b", text)
                name = m.group(1) if m else ""
                if not name or name.lower() in {"the", "for", "met", "i", "bio", "pcb"}:
                    continue
                dest_path = f"Hayden/Relationships/People/{name}.json"
                person_path = dest_path
                if not self.db.paths.resolve(person_path).exists():
                    self.db.create_json(
                        person_path,
                        {
                            "name": name,
                            "aliases": [],
                            "how_we_met": text,
                            "relationship_type": "acquaintance",
                            "status": "active",
                            "closeness": 0.3,
                            "how_i_feel": "",
                            "how_i_act_around_them": "",
                            "what_they_know_about_me": [],
                            "what_i_know_about_them": [text],
                            "shared_history": [],
                            "boundaries": [],
                            "secrets_involving_them": [],
                            "triggers_around_them": [],
                            "tags": [],
                            "related_paths": [],
                            "last_updated": _utc_now(),
                        },
                        summary=f"Host-create person dossier for {name}",
                    )
                else:
                    self._host_append_note(
                        person_path,
                        field="what_i_know_about_them",
                        note=text,
                        summary=f"Host-update person dossier for {name}",
                    )
                filed = True
            elif re.search(r"\b(oat milk|dish soap|pantry|restock|household)\b", low):
                dest_path = "Household/Pantry/Staples.json"
                path = dest_path
                if not self.db.paths.resolve(path).exists():
                    self.db.create_json(
                        path,
                        {"staples": [], "low": [], "last_updated": _utc_now()},
                        summary="Host-create pantry staples",
                    )
                data = self.db.read_json(path)["data"]
                if isinstance(data, dict):
                    low_list = data.get("low") if isinstance(data.get("low"), list) else []
                    if text not in low_list:
                        low_list.append(text)
                    data["low"] = low_list
                    data["last_updated"] = _utc_now()
                    self.db.write_json(path, data, summary="Host-file household restock")
                    filed = True
            elif re.search(r"\b(wrist|trackpad|ergonomic|sleep|sore)\b", low):
                dest_path = "Hayden/Body/Health.json"
                filed = self._host_append_note(
                    dest_path,
                    field="notes",
                    note=text,
                    summary="Host-file body/health note",
                )
            elif re.search(r"\b(anxious|open loops?|trigger)\b", low):
                dest_path = "Hayden/Psychology/Triggers.json"
                filed = self._host_append_note(
                    dest_path,
                    field="triggers",
                    note=text,
                    summary="Host-file psychology trigger",
                )
            elif re.search(r"\b(goal|this month|trust(?:ing)? the (?:personal )?db)\b", low):
                dest_path = "Hayden/Desires/Goals.json"
                filed = self._host_append_note(
                    dest_path,
                    field="goals",
                    note=text,
                    summary="Host-file desire/goal",
                )
            elif re.search(r"\b(craftsmanship|half-baked|values?|identity)\b", low):
                dest_path = "Hayden/Values/Principles.json"
                filed = self._host_append_note(
                    dest_path,
                    field="values",
                    note=text,
                    summary="Host-file values/identity note",
                )
            elif re.search(r"\b(ainet|soi filing|project)\b", low):
                dest_path = "Work/Projects/AINet/Plan.json"
                path = dest_path
                if self.db.paths.resolve(path).exists():
                    data = self.db.read_json(path)["data"]
                    if isinstance(data, dict):
                        plans = data.get("plans") if isinstance(data.get("plans"), list) else []
                        if text not in plans:
                            plans.append(text)
                        data["plans"] = plans
                        self.db.write_json(path, data, summary="Host-file AINet plan note")
                        filed = True
            elif re.search(r"\b(esp32|milestone|huge win|remember when)\b", low):
                dest_path = "Hayden/Memories/Milestones/Log.json"
                path = dest_path
                if self.db.paths.resolve(path).exists():
                    data = self.db.read_json(path)["data"]
                    if isinstance(data, dict):
                        entries = data.get("entries") if isinstance(data.get("entries"), list) else []
                        entries.append(
                            {
                                "id": eid[:12],
                                "title": text[:80],
                                "when": "",
                                "why_it_matters": text,
                            }
                        )
                        data["entries"] = entries[-40:]
                        data["last_updated"] = _utc_now()
                        self.db.write_json(path, data, summary="Host-file memory milestone")
                        filed = True

            if filed:
                filed_ids.append(eid)
                actions.append(f"{eid}:{text[:40]}")
                if dest_path:
                    dest_by_id[eid] = dest_path

        return {"filed_ids": filed_ids, "actions": actions, "dest_by_id": dest_by_id}

    def _host_file_personal_turns(self, batch_changelog: list[dict[str, Any]]) -> dict[str, Any]:
        """Identity / psychology / habits / voice — same id-copy path as research."""
        filed_ids: list[str] = []
        dest_by_id: dict[str, str] = {}
        for entry in batch_changelog:
            kind = entry.get("suggested_filing") or entry_kind(entry)
            if kind not in {"identity", "personality", "voice", "psychology", "habits"}:
                continue
            if _is_ephemeral_entry(entry):
                continue
            eid = str(entry.get("id") or "")
            if not eid:
                continue
            details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
            path = record_personal_filing(
                self.db,
                kind,
                user=_entry_user_text(entry),
                assistant=str(details.get("assistant_text") or ""),
                entry_ids=[eid],
            )
            if kind in {"voice", "identity", "personality"}:
                from ollama.file_by_id import _append_voice_evidence

                _append_voice_evidence(self.db, _entry_user_text(entry))
            filed_ids.append(eid)
            dest_by_id[eid] = path or f"Hayden/{kind}/Notes.json"
        return {"filed_ids": filed_ids, "dest_by_id": dest_by_id}

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
