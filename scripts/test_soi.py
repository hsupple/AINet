#!/usr/bin/env python3
"""SOI filing harness — sandbox copy of real db/, mutations blocked by default.

Copies the live database into a sandbox, then injects pending SOI work the same way
OAC does: ConversationStore / changelog.append_entry → oac_turn with details
{session_id, mode_id, topic, user_text, assistant_text} and soi_status=pending.

By default, existing oac_turn rows from your Changelog.json are re-pended (same shape
and content). Optional extra turns are appended through that same injector.

Default mode is dry-run: full SOI tool catalog, mutations do not write. Real db/ is
never touched.

Examples (from repo root):
  python scripts/test_soi.py
  python scripts/test_soi.py --phase both
  python scripts/test_soi.py --apply
  python scripts/test_soi.py --keep
  python scripts/test_soi.py --seed-only
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Allow `python scripts/test_soi.py` from repo root.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ainet.tools import changelog as changelog_mod
from ainet.tools.paths import DbPaths
from ainet.tools.registry import catalog_tools
from ollama.config import OllamaConfig, default_db_root
from ollama.conversation_store import ConversationStore
from ollama.session import ChatSession, _MUTATING
from ollama.soi_worker import SOIWorker

# Tools that change durable or runtime state (beyond the session _MUTATING set).
_EXTRA_MUTATING = frozenset(
    {
        "mark_read_stale",
        "mark_read_refreshed",
        "start_quiz",
        "record_quiz_answer",
    }
)
_ALL_MUTATING = _MUTATING | _EXTRA_MUTATING


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Sandbox + seed (inject like OAC Changelog handoffs)
# ---------------------------------------------------------------------------


def copy_sandbox(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".git")
    shutil.copytree(source, dest, ignore=ignore)
    # Fresh SOI runtime so host state from prod does not leak in.
    runtime_soi = dest / "runtime" / "soi"
    runtime_soi.mkdir(parents=True, exist_ok=True)
    _write_json(runtime_soi / "state.json", {"status": "harness", "seeded_at": _utc_now()})
    _write_json(runtime_soi / "cursor.json", {"last_index": -1, "updated_at": _utc_now()})


def _is_oac_turn(entry: dict[str, Any]) -> bool:
    return entry.get("action") == "oac_turn" or entry.get("actor") == "oac"


def repend_existing_oac_turns(root: Path) -> list[dict[str, Any]]:
    """Reset copied Changelog oac_turn rows to pending — keep exact live entry shape."""
    path = root / "Changelog.json"
    data = _read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise ValueError("Changelog.json must be an object with an entries array")

    pending: list[dict[str, Any]] = []
    for entry in data["entries"]:
        if not isinstance(entry, dict) or not _is_oac_turn(entry):
            continue
        entry["soi_status"] = "pending"
        entry.pop("soi_processed_at", None)
        # Normalize details to the OAC injector keys only (drop any stray extras).
        details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
        entry["details"] = {
            "session_id": details.get("session_id"),
            "mode_id": details.get("mode_id") or "companion",
            "topic": details.get("topic"),
            "user_text": details.get("user_text") or entry.get("summary") or "",
            "assistant_text": details.get("assistant_text") or "",
        }
        pending.append(copy.deepcopy(entry))

    _write_json(path, data)
    return pending


def inject_oac_turn(
    store: ConversationStore,
    *,
    user_text: str,
    assistant_text: str,
    mode_id: str = "companion",
    topic: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Inject one pending handoff exactly like live OAC (ConversationStore.append_turn)."""
    if session_id is None:
        session_id = store.ensure_session(mode_id=mode_id, topic=topic)
    else:
        # Ensure session file exists so path in changelog is real.
        sess_path = store.sessions_dir / f"{session_id}.json"
        if not sess_path.exists():
            store.sessions_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                sess_path,
                {
                    "id": session_id,
                    "mode_id": mode_id,
                    "topic": topic,
                    "created_at": _utc_now(),
                    "updated_at": _utc_now(),
                    "turns": [],
                },
            )
        store._set_current(session_id)  # noqa: SLF001 — mirror live session pointer

    store.append_turn(
        session_id,
        user_text=user_text,
        assistant_text=assistant_text,
        mode_id=mode_id,
        topic=topic,
    )
    # Return the changelog row just appended (last oac_turn).
    entries = changelog_mod.pending_oac_entries(store.paths)
    return entries[-1] if entries else {}


def reopen_inbox_captures(root: Path) -> list[dict[str, Any]]:
    """Re-open existing Captures.json rows as unfiled (keep live capture shape/text)."""
    path = root / "Hayden" / "Inbox" / "Captures.json"
    if not path.exists():
        return []
    data = _read_json(path)
    if not isinstance(data, dict):
        return []
    captures = data.get("captures") if isinstance(data.get("captures"), list) else []
    reopened: list[dict[str, Any]] = []
    for cap in captures:
        if not isinstance(cap, dict):
            continue
        cap["status"] = "unfiled"
        cap.pop("filed_to", None)
        # Live captures use source=conversation; keep whatever was there.
        reopened.append(
            {
                "id": cap.get("id"),
                "suggested_home": cap.get("suggested_home") or "",
                "text": cap.get("text") or "",
            }
        )
    data["captures"] = captures
    data["last_updated"] = _utc_now()
    _write_json(path, data)
    return reopened


def harness_conversations() -> list[dict[str, Any]]:
    """Natural OAC-style turns covering a range of filing targets (Changelog shape only)."""
    return [
        # Continue nuclear research thread
        {
            "user_text": "ok so the water is both coolant and moderator?",
            "assistant_text": (
                "Yep — in light-water reactors the same water usually does both jobs: "
                "it removes heat from the fuel rods and slows neutrons so fission stays "
                "efficient. Different designs can separate those roles, but PWR/BWR style "
                "plants lean on that dual use."
            ),
            "mode_id": "research",
            "topic": None,
        },
        {
            "user_text": "what happens to spent fuel rods after they pull them out?",
            "assistant_text": (
                "Usually they cool in a spent-fuel pool for years, then move to dry cask "
                "storage. The pellets stay inside the cladding; fission products are trapped "
                "in the ceramic until reprocessing or long-term disposal."
            ),
            "mode_id": "research",
            "topic": None,
        },
        # Preferences / places / food (NOT research)
        {
            "user_text": "cool cool. also that campus coffee shop table with the outlets still slaps",
            "assistant_text": (
                "Ha — noted. Campus coffee shop outlet table is still the move. "
                "I'll treat that as a place pref, not research."
            ),
            "mode_id": "companion",
            "topic": None,
        },
        {
            "user_text": "I've been switching to matcha in the afternoon — espresso wrecks my sleep",
            "assistant_text": (
                "Makes sense. Matcha afternoon / skip late espresso is a solid food+energy pref."
            ),
            "mode_id": "companion",
            "topic": None,
        },
        # Habits
        {
            "user_text": "trying 25 min focus then 5 min stretch every afternoon this week",
            "assistant_text": (
                "Got it — afternoon 25/5 focus+stretch blocks. Habit territory."
            ),
            "mode_id": "companion",
            "topic": None,
        },
        # Relationships
        {
            "user_text": (
                "Met Jordan at the makerspace yesterday — they do PCB layout and offered "
                "to review my board this weekend"
            ),
            "assistant_text": (
                "Jordan from the makerspace, PCB review this weekend — worth a People note."
            ),
            "mode_id": "companion",
            "topic": None,
        },
        # Work / AINet
        {
            "user_text": (
                "for AINet the next concrete step is making SOI filing trustworthy again"
            ),
            "assistant_text": (
                "AINet focus: trustworthy SOI filing. I'll keep that on the project plan surface."
            ),
            "mode_id": "planner",
            "topic": None,
        },
        # School
        {
            "user_text": (
                "BIO problem set on cellular respiration is due Thursday, I still need "
                "the Krebs cycle questions"
            ),
            "assistant_text": (
                "School reminder: BIO PS due Thursday — Krebs cycle still open."
            ),
            "mode_id": "planner",
            "topic": None,
        },
        # Body / ergonomics
        {
            "user_text": "right wrist gets sore after long trackpad sessions, need more keyboard shortcuts",
            "assistant_text": (
                "Wrist strain from trackpad use — ergonomics + shortcuts. Body/health leaf."
            ),
            "mode_id": "companion",
            "topic": None,
        },
        # Psychology
        {
            "user_text": "vague open loops still make me anxious until I write a next action",
            "assistant_text": (
                "Open loops → anxiety until there's a concrete next action. Useful trigger note."
            ),
            "mode_id": "companion",
            "topic": None,
        },
        # Content-based research (companion mode, should still file as research)
        {
            "user_text": "wait quick aside — what other muscles stabilize the wrist in a reverse curl?",
            "assistant_text": (
                "Wrist stabilizers in a reverse curl include extensor carpi radialis longus/"
                "brevis and ulnaris; brachioradialis is doing a lot of the elbow flexion "
                "with that grip."
            ),
            "mode_id": "companion",
            "topic": None,
        },
        # Research with explicit topic
        {
            "user_text": "how does the proton gradient actually drive ATP synthase?",
            "assistant_text": (
                "Protons flow back through ATP synthase (F0/F1). The gradient's potential "
                "energy turns the rotor; that mechanical rotation drives ADP + Pi → ATP "
                "in the catalytic head. Chemiosmosis."
            ),
            "mode_id": "research",
            "topic": "Mitochondria",
        },
        # Household
        {
            "user_text": "we're basically out of oat milk and dish soap btw",
            "assistant_text": "Restock list: oat milk + dish soap.",
            "mode_id": "companion",
            "topic": None,
        },
        # Memories / wins
        {
            "user_text": (
                "remember when we finally got the ESP32 mic pipeline working last winter? "
                "still feels like a huge win"
            ),
            "assistant_text": (
                "Yeah — ESP32 mic pipeline coming alive was a real milestone. Worth keeping."
            ),
            "mode_id": "companion",
            "topic": None,
        },
        # Identity / values
        {
            "user_text": "I care more about craftsmanship than shipping half-baked systems",
            "assistant_text": (
                "Craft over half-baked ship — that's an identity/values signal, not a todo."
            ),
            "mode_id": "conversation",
            "topic": None,
        },
        # Desires / goals
        {
            "user_text": "goal for this month is trusting the personal DB again",
            "assistant_text": "Monthly goal: trust the personal DB again via reliable filing.",
            "mode_id": "planner",
            "topic": None,
        },
        # Ephemeral — should discard
        {
            "user_text": "lol ok thanks",
            "assistant_text": "Anytime!",
            "mode_id": "companion",
            "topic": None,
        },
        {
            "user_text": "gg",
            "assistant_text": "Later 👋",
            "mode_id": "companion",
            "topic": None,
        },
    ]


def seed_pending_work(root: Path, *, extra_turns: bool = True) -> dict[str, Any]:
    """Re-pend live changelog oac_turns; append range conversations via OAC injector."""
    repended = repend_existing_oac_turns(root)
    store = ConversationStore(root)

    injected: list[dict[str, Any]] = []
    if extra_turns:
        session_id = None
        if repended:
            details = repended[0].get("details") if isinstance(repended[0].get("details"), dict) else {}
            session_id = details.get("session_id")

        for row in harness_conversations():
            entry = inject_oac_turn(
                store,
                user_text=row["user_text"],
                assistant_text=row["assistant_text"],
                mode_id=row["mode_id"],
                topic=row["topic"],
                session_id=session_id,
            )
            if entry:
                injected.append(entry)
                if session_id is None:
                    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
                    session_id = details.get("session_id")

    captures = reopen_inbox_captures(root)
    pending = changelog_mod.pending_oac_entries(DbPaths(root))

    return {
        "changelog_pending": len(pending),
        "inbox_unfiled": len(captures),
        "repended": len(repended),
        "injected_extra": len(injected),
        "turns": [
            {
                "id": e.get("id"),
                "mode_id": (e.get("details") or {}).get("mode_id"),
                "topic": (e.get("details") or {}).get("topic"),
                "user": (e.get("details") or {}).get("user_text") or e.get("summary"),
                "path": e.get("path"),
                "soi_status": e.get("soi_status"),
            }
            for e in pending
        ],
        "captures": captures,
    }


# ---------------------------------------------------------------------------
# Dry-run / logging wrappers
# ---------------------------------------------------------------------------


class ToolTrace:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record(self, name: str, args: dict[str, Any], result: dict[str, Any], *, blocked: bool) -> None:
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


def _preview(obj: Any, limit: int = 400) -> Any:
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


def install_harness(
    session_ask_hook: Callable[[], None] | None,
    *,
    dry_run: bool,
    trace: ToolTrace,
    allow_web: bool,
) -> list[Callable[[], None]]:
    """Monkeypatch ChatSession + host SOI mutators. Returns restore callables."""
    restores: list[Callable[[], None]] = []

    original_run = ChatSession._run_tool_call

    def patched_run(self: ChatSession, call: dict[str, Any]) -> dict[str, Any]:
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

        if name in {"web_search", "web_fetch"} and not allow_web:
            result = {
                "ok": False,
                "error": "web tools disabled in harness (pass --allow-web)",
                "harness": True,
            }
            trace.record(name, args, result, blocked=True)
            return result

        if dry_run and name in _ALL_MUTATING:
            result = {
                "ok": True,
                "dry_run": True,
                "harness": True,
                "tool": name,
                "would_apply": args,
                "note": "Mutation blocked — sandbox dry-run. Pass --apply to write inside sandbox.",
            }
            trace.record(name, args, result, blocked=True)
            return result

        result = original_run(self, call)
        if not isinstance(result, dict):
            result = {"ok": True, "value": result}
        trace.record(name, args, result, blocked=False)
        return result

    ChatSession._run_tool_call = patched_run  # type: ignore[method-assign]
    restores.append(lambda: setattr(ChatSession, "_run_tool_call", original_run))

    # Host safety nets that write outside the tool loop.
    import ainet.tools.changelog as changelog_mod
    import ollama.research_sessions as sessions_mod
    import ollama.soi_worker as worker_mod

    orig_mark = changelog_mod.mark_soi_status
    orig_host = worker_mod.SOIWorker._host_file_research_turns
    orig_merge = worker_mod.SOIWorker._merge_state
    orig_cursor = worker_mod.SOIWorker._save_cursor
    orig_upsert = sessions_mod.upsert_research_session

    def mark_wrapper(paths, *, entry_ids, status, dest_by_id=None, **_kw):  # type: ignore[no-untyped-def]
        if dry_run:
            result = {
                "ok": True,
                "dry_run": True,
                "would_mark": list(entry_ids),
                "status": status,
                "dest_by_id": dest_by_id or {},
                "count": len(entry_ids),
            }
            trace.record(
                "host.mark_soi_status",
                {"entry_ids": list(entry_ids), "status": status},
                result,
                blocked=True,
            )
            return len(entry_ids)
        count = orig_mark(paths, entry_ids=entry_ids, status=status, dest_by_id=dest_by_id)
        trace.record(
            "host.mark_soi_status",
            {"entry_ids": list(entry_ids), "status": status, "dest_by_id": dest_by_id or {}},
            {"ok": True, "count": count},
            blocked=False,
        )
        return count

    def host_research_wrapper(self, batch):  # type: ignore[no-untyped-def]
        if dry_run:
            hinted = [
                e.get("id")
                for e in batch
                if isinstance(e, dict) and e.get("suggested_filing") == "research"
            ]
            result = {"ok": True, "dry_run": True, "would_host_file": hinted}
            trace.record("host._host_file_research_turns", {"count": len(batch)}, result, blocked=True)
            return []
        created = orig_host(self, batch)
        trace.record(
            "host._host_file_research_turns",
            {"count": len(batch)},
            {"ok": True, "created": created},
            blocked=False,
        )
        return created

    def merge_wrapper(self, patch):  # type: ignore[no-untyped-def]
        if dry_run:
            trace.record("host._merge_state", patch, {"ok": True, "dry_run": True}, blocked=True)
            return None
        return orig_merge(self, patch)

    def cursor_wrapper(self, last_index):  # type: ignore[no-untyped-def]
        if dry_run:
            trace.record(
                "host._save_cursor",
                {"last_index": last_index},
                {"ok": True, "dry_run": True},
                blocked=True,
            )
            return None
        return orig_cursor(self, last_index)

    changelog_mod.mark_soi_status = mark_wrapper  # type: ignore[assignment]
    worker_mod.SOIWorker._host_file_research_turns = host_research_wrapper  # type: ignore[method-assign]
    worker_mod.SOIWorker._merge_state = merge_wrapper  # type: ignore[method-assign]
    worker_mod.SOIWorker._save_cursor = cursor_wrapper  # type: ignore[method-assign]

    restores.append(lambda: setattr(changelog_mod, "mark_soi_status", orig_mark))
    restores.append(lambda: setattr(worker_mod.SOIWorker, "_host_file_research_turns", orig_host))
    restores.append(lambda: setattr(worker_mod.SOIWorker, "_merge_state", orig_merge))
    restores.append(lambda: setattr(worker_mod.SOIWorker, "_save_cursor", orig_cursor))
    restores.append(lambda: setattr(sessions_mod, "upsert_research_session", orig_upsert))

    if session_ask_hook:
        session_ask_hook()

    return restores


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def summarize(trace: ToolTrace, seed_meta: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    by_tool: dict[str, int] = {}
    blocked = 0
    for c in trace.calls:
        by_tool[c["tool"]] = by_tool.get(c["tool"], 0) + 1
        if c["blocked"]:
            blocked += 1

    intended_paths: list[str] = []
    for c in trace.calls:
        if not c["blocked"]:
            continue
        args = c.get("args") or {}
        for key in ("path", "src", "dest", "folder_or_path", "read_path", "suggested_home"):
            val = args.get(key)
            if isinstance(val, str) and val:
                intended_paths.append(val)
        would = (c.get("result_preview") or {}).get("would_apply") if isinstance(c.get("result_preview"), dict) else None
        if isinstance(would, dict):
            for key in ("path", "src", "dest", "folder_or_path", "read_path"):
                val = would.get(key)
                if isinstance(val, str) and val:
                    intended_paths.append(val)

    catalog = catalog_tools(detail=False, include_meta=True, read_only=False)
    return {
        "seed": {
            "changelog_pending": seed_meta.get("changelog_pending"),
            "inbox_unfiled": seed_meta.get("inbox_unfiled"),
            "repended": seed_meta.get("repended"),
            "injected_extra": seed_meta.get("injected_extra"),
            "turns_by_mode": _count_by(seed_meta.get("turns") or [], "mode_id"),
        },
        "tool_catalog_count": catalog.get("count"),
        "tool_names": [t["name"] for t in catalog.get("tools") or []],
        "trace": {
            "total_calls": len(trace.calls),
            "blocked_mutations": blocked,
            "by_tool": by_tool,
            "intended_paths": sorted(set(intended_paths)),
        },
        "phase_results": results,
        "calls": trace.calls,
    }


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        k = str(row.get(key) or "?")
        out[k] = out.get(k, 0) + 1
    return out


def print_report(report: dict[str, Any], sandbox: Path, dry_run: bool) -> None:
    print("\n=== SOI harness report ===")
    print(f"sandbox: {sandbox}")
    print(f"dry_run: {dry_run}  (real db/ never touched)")
    seed = report["seed"]
    print(
        f"pending: {seed['changelog_pending']} oac_turns "
        f"(repended={seed.get('repended')}, extra={seed.get('injected_extra')}), "
        f"{seed['inbox_unfiled']} unfiled captures"
    )
    modes = seed.get("turns_by_mode") or {}
    if modes:
        print(f"modes: {', '.join(f'{k}={v}' for k, v in sorted(modes.items()))}")
    print(f"full tool catalog: {report['tool_catalog_count']} tools available to SOI")
    trace = report["trace"]
    print(
        f"tool calls: {trace['total_calls']} "
        f"(blocked mutations: {trace['blocked_mutations']})"
    )
    if trace["by_tool"]:
        print("by tool:")
        for name, n in sorted(trace["by_tool"].items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {n:3d}  {name}")
    if trace["intended_paths"]:
        print("intended write paths (dry-run):")
        for p in trace["intended_paths"]:
            print(f"  - {p}")
    for phase, result in (report.get("phase_results") or {}).items():
        print(f"\n-- {phase} --")
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str)[:2000])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sandbox SOI test harness (dry-run by default)")
    p.add_argument("--source-db", type=Path, default=None, help="Source DB to copy (default: repo db/)")
    p.add_argument("--sandbox", type=Path, default=None, help="Sandbox path (default: temp dir)")
    p.add_argument("--keep", action="store_true", help="Do not delete sandbox when done")
    p.add_argument(
        "--apply",
        action="store_true",
        help="Allow mutations inside the sandbox (still never touches real db/)",
    )
    p.add_argument(
        "--phase",
        choices=("filing", "read_refresh", "both", "seed-only"),
        default="filing",
        help="Which SOI phase(s) to run",
    )
    p.add_argument("--seed-only", action="store_true", help="Alias for --phase seed-only")
    p.add_argument(
        "--extra-turns",
        dest="extra_turns",
        action="store_true",
        default=True,
        help="Append range conversations via ConversationStore (default: on)",
    )
    p.add_argument(
        "--no-extra-turns",
        dest="extra_turns",
        action="store_false",
        help="Only re-pend existing Changelog oac_turns",
    )
    p.add_argument("--model", default=None, help="Override Ollama model")
    p.add_argument("--host", default=None, help="Override Ollama host")
    p.add_argument("--max-tool-rounds", type=int, default=24, help="Tool rounds for SOI turns")
    p.add_argument("--allow-web", action="store_true", help="Allow web_search/web_fetch")
    p.add_argument("--report", type=Path, default=None, help="Write full JSON report here")
    return p


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = build_arg_parser().parse_args(argv)
    if args.seed_only:
        args.phase = "seed-only"

    source = (args.source_db or default_db_root()).resolve()
    if not source.is_dir():
        print(f"Source DB not found: {source}", file=sys.stderr)
        return 2

    own_temp = False
    if args.sandbox:
        sandbox = args.sandbox.resolve()
        sandbox.parent.mkdir(parents=True, exist_ok=True)
    else:
        sandbox = Path(tempfile.mkdtemp(prefix="ainet-soi-harness-")).resolve()
        own_temp = True

    dry_run = not args.apply
    print(f"Copying {source} -> {sandbox}")
    copy_sandbox(source, sandbox)

    print("Re-pending live Changelog oac_turns (+ inbox) via OAC injection shape…")
    pending = seed_pending_work(sandbox, extra_turns=args.extra_turns)
    print(
        f"  repended={pending['repended']} extra={pending['injected_extra']} "
        f"pending={pending['changelog_pending']} inbox={pending['inbox_unfiled']}"
    )

    config = OllamaConfig.from_env()
    updates: dict[str, Any] = {
        "db_root": sandbox,
        "persist_oac_conversation": False,
        "soi_enabled": False,
        "max_tool_rounds": args.max_tool_rounds,
    }
    if args.model:
        updates["model"] = args.model
    if args.host:
        updates["host"] = args.host.rstrip("/")
    from dataclasses import replace

    config = replace(config, **updates)

    report_path = args.report or (sandbox / "harness-report.json")
    results: dict[str, Any] = {}
    trace = ToolTrace()

    if args.phase == "seed-only":
        report = summarize(trace, pending, {"seed_only": True})
        _write_json(report_path, report)
        print_report(report, sandbox, dry_run=True)
        print(f"\nReport: {report_path}")
        print("Pending oac_turns (Changelog shape):")
        for t in pending["turns"]:
            user = (t.get("user") or "")[:80]
            print(f"  [{t.get('mode_id')}] {t.get('id')}: {user}")
        # Show one raw entry so you can verify injection matches live Changelog.
        raw = changelog_mod.pending_oac_entries(DbPaths(sandbox))
        if raw:
            sample = {k: raw[0][k] for k in ("id", "actor", "action", "path", "summary", "details", "soi_status") if k in raw[0]}
            print("\nSample pending entry:")
            print(json.dumps(sample, indent=2, ensure_ascii=False)[:1500])
        if own_temp and not args.keep:
            print(f"\nSandbox kept for seed-only inspection: {sandbox}")
            args.keep = True
        return 0

    restores = install_harness(None, dry_run=dry_run, trace=trace, allow_web=args.allow_web)
    worker = SOIWorker(config=config)

    try:
        if args.phase in {"filing", "both"}:
            print("\nRunning SOI Phase 1 (filing)…")
            try:
                results["filing"] = worker.run_filing()
            except Exception as exc:
                results["filing"] = {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }

        if args.phase in {"read_refresh", "both"}:
            print("\nRunning SOI Phase 2 (read_refresh)…")
            # In dry-run, filing still leaves pending work in the sandbox files,
            # so force phase-2 by clearing the filing-work gate for this call.
            if dry_run and args.phase == "both":
                worker.has_filing_work = lambda: False  # type: ignore[method-assign]
            try:
                results["read_refresh"] = worker.run_read_refresh()
            except Exception as exc:
                results["read_refresh"] = {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
    finally:
        for restore in restores:
            try:
                restore()
            except Exception:
                pass

    report = summarize(trace, pending, results)
    report["sandbox"] = str(sandbox)
    report["dry_run"] = dry_run
    _write_json(report_path, report)
    print_report(report, sandbox, dry_run)
    print(f"\nFull report: {report_path}")

    if own_temp and not args.keep:
        print(f"Cleaning sandbox {sandbox} (pass --keep to retain)")
        shutil.rmtree(sandbox, ignore_errors=True)
    else:
        print(f"Sandbox retained at {sandbox}")

    for result in results.values():
        if isinstance(result, dict) and result.get("ok") is False:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())