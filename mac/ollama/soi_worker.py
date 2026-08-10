"""SOI worker — Phase 1 filing (changelog/inbox), Phase 2 Read.json refresh."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ainet.tools import changelog
from ainet.tools import readlog
from ainet.tools.fsutil import atomic_write_text
from ainet.tools.ops import DatabaseTools
from ollama.client import OllamaClient, OllamaError
from ollama.config import OllamaConfig
from ollama.modes import get_mode
from ollama.session import ChatSession


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class SOIWorker:
    def __init__(self, config: OllamaConfig | None = None, client: OllamaClient | None = None) -> None:
        self.config = config or OllamaConfig.from_env()
        self.client = client or OllamaClient(self.config)
        self.db = DatabaseTools(self.config.db_root)
        changelog.ensure_changelog_file(Path(self.config.db_root))
        self.state_dir = Path(self.config.db_root) / "runtime" / "soi"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "state.json"
        # Legacy cursor kept for migration/debug; pending uses per-entry soi_status.
        self.cursor_path = self.state_dir / "cursor.json"

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
            return {"ok": True, "ran": False, "phase": "filing", "reason": "no pending work"}

        batch_changelog = changelog_pending[:40]
        batch_inbox = inbox[:40]
        payload = {
            "phase": "filing",
            "changelog_entries": batch_changelog,
            "inbox_unfiled": batch_inbox,
            "instructions": (
                "You are SOI Phase 1 (filing). "
                "For each oac_turn: lasting content must be filed into the correct DB leaf "
                "(or capture_inbox if still unclear). Ephemeral noise may be discarded. "
                "If details.mode_id is 'research' OR details.topic is set: call upsert_research_session "
                "with subject/title, topic_slug when known, length_turns, and details_covered "
                "(every mechanism/point/QA from the turn). Append across related turns; "
                "when the rabbit hole ends, call complete_research_session. "
                "Also update Topics/<Slug>/Notes for lasting topic facts. "
                "Update Hayden/Inbox/Captures.json statuses to filed/discarded with filed_to when handled. "
                "Do NOT rewrite Changelog.json yourself — the host marks entries after this run. "
                "Return a short JSON status listing filed vs discarded entry ids if possible."
            ),
        }
        reply, err = self._ask_soi(payload)
        if err:
            self._merge_state({"status": "error", "phase": "filing", "error": err})
            return {"ok": False, "ran": True, "phase": "filing", "error": err}

        # Host marks changelog handoffs processed so they leave the pending set.
        # Default: filed. If model lists discarded ids in a simple way, honor them.
        entry_ids = [e["id"] for e in batch_changelog if e.get("id")]
        discarded_ids = self._parse_discarded_ids(reply, entry_ids)
        filed_ids = [eid for eid in entry_ids if eid not in discarded_ids]
        marked_filed = changelog.mark_soi_status(
            self.db.paths, entry_ids=filed_ids, status="filed"
        )
        marked_discarded = changelog.mark_soi_status(
            self.db.paths, entry_ids=list(discarded_ids), status="discarded"
        )

        # Advance legacy cursor for debugging
        if batch_changelog:
            self._save_cursor(batch_changelog[-1]["index"])

        processed_any = bool(batch_changelog or batch_inbox)
        self._merge_state(
            {
                "status": "ok",
                "phase": "filing",
                "processed_changelog": len(batch_changelog),
                "marked_filed": marked_filed,
                "marked_discarded": marked_discarded,
                "seen_inbox": len(batch_inbox),
                "needs_read_refresh": True if processed_any else self.needs_read_refresh(),
                "last_filing_at": _utc_now(),
                "reply_preview": (reply or "")[:400],
            }
        )
        return {
            "ok": True,
            "ran": True,
            "phase": "filing",
            "processed_changelog": len(batch_changelog),
            "marked_filed": marked_filed,
            "marked_discarded": marked_discarded,
            "seen_inbox": len(batch_inbox),
            "reply": reply,
        }

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
            return {
                "ok": True,
                "ran": False,
                "phase": "read_refresh",
                "reason": "no stale Reads (needs_update=false)",
            }

        grouped = self.reads_by_domain(stale)
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
                    "After each successful Read rewrite, call mark_read_refreshed on that Read path "
                    "(or folder) so needs_update=false and pending changelog entries are consumed."
                ),
            }
            reply, err = self._ask_soi(payload)
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
        return {
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

    # ---- helpers -----------------------------------------------------------

    def _ask_soi(self, payload: dict[str, Any]) -> tuple[str | None, str | None]:
        session = ChatSession(
            mode=get_mode("soi"),
            config=self.config,
            client=self.client,
            auto_mode=False,
            persist_conversation=False,
        )
        try:
            reply = session.ask(
                "SOI job — process this batch:\n"
                + json.dumps(payload, ensure_ascii=False, indent=2)
            )
            return reply, None
        except OllamaError as exc:
            return None, str(exc)

    def _parse_discarded_ids(self, reply: str | None, known_ids: list[str]) -> set[str]:
        """Best-effort: if SOI mentions discarded ids, honor them; else none."""
        if not reply or not known_ids:
            return set()
        discarded: set[str] = set()
        # Look for a JSON object in the reply
        start = reply.find("{")
        end = reply.rfind("}")
        if start >= 0 and end > start:
            try:
                blob = json.loads(reply[start : end + 1])
            except json.JSONDecodeError:
                blob = None
            if isinstance(blob, dict):
                raw = blob.get("discarded") or blob.get("discarded_ids") or []
                if isinstance(raw, list):
                    for item in raw:
                        if item in known_ids:
                            discarded.add(str(item))
        # Also accept bare id tokens after the word discarded
        lower = reply.lower()
        if "discard" in lower:
            for eid in known_ids:
                if eid in reply:
                    # only if near discard context — keep conservative: require JSON path above
                    pass
        return discarded

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
