#!/usr/bin/env python3
"""End-to-end stress test: fake OAC turns → SOI filing → OAC query_db retrieval.

Uses an isolated sandbox copy of db/. Never touches live db/ unless --source-db points there
and you pass --apply (writes stay inside sandbox only).

Examples:
  python scripts/stress_test_kb.py
  python scripts/stress_test_kb.py --apply --keep
  python scripts/stress_test_kb.py --apply --skip-oac
  python scripts/stress_test_kb.py --apply --report stress-report.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ainet.logstore import (
    HAYDEN_FILE,
    ROOT_FILES,
    decay_knowledge,
    empty_hayden,
    empty_log,
    ensure_knowledge_files,
    query_db,
)
from ainet.tools import changelog as changelog_mod
from ainet.tools.ops import DatabaseTools
from ainet.tools.registry import dispatch
from dataclasses import replace
from ollama.client import OllamaClient, OllamaError
from ollama.config import OllamaConfig, default_db_root
from ollama.conversation_store import ConversationStore
from ollama.modes import get_mode
from ollama.session import ChatSession
from ollama.soi_worker import SOIWorker

from scripts.test_soi import copy_sandbox, inject_oac_turn, install_harness, ToolTrace


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _reset_changelog(root: Path) -> None:
    _write_json(root / "Changelog.json", {"version": 1, "entries": []})
    _write_json(root / "Masterlog.json", {"version": 1, "entries": [], "last_updated": ""})


def _empty_knowledge(root: Path) -> None:
    _write_json(root / HAYDEN_FILE, empty_hayden())
    for name in ROOT_FILES:
        if name == HAYDEN_FILE:
            continue
        _write_json(root / name, empty_log())
    leftover = root / "planner.json"
    if leftover.is_file():
        leftover.unlink()


# ---------------------------------------------------------------------------
# Curated fake turns + filing expectations
# ---------------------------------------------------------------------------


@dataclass
class TurnSpec:
    user_text: str
    expect_dests: list[str]
    note: str = ""
    allow_discard_only: bool = False


def stress_turns() -> list[TurnSpec]:
    return [
        TurnSpec(
            "Alex came over last night, we're friends from the climbing gym",
            ["people"],
            "person → people.json key Alex",
        ),
        TurnSpec(
            "I've been doing 25 minute focus blocks with a 5 minute stretch after lunch",
            ["habits"],
        ),
        TurnSpec(
            "switching to matcha in the afternoon because espresso wrecks my sleep",
            ["preferences", "habits"],
        ),
        TurnSpec(
            "we're basically out of oat milk and the dish soap is almost gone",
            ["household"],
        ),
        TurnSpec(
            "open loops still make me anxious until I write a concrete next action",
            ["psychology"],
        ),
        TurnSpec(
            "I care more about craftsmanship than shipping half-baked systems",
            ["values", "characteristics", "hayden"],
        ),
        TurnSpec(
            "long term I want to trust my personal database again",
            ["desires"],
        ),
        TurnSpec(
            "what happens to spent fuel rods after they pull them out of a reactor?",
            ["questions"],
        ),
        TurnSpec(
            "remember when the ESP32 mic pipeline finally worked? still a huge win",
            ["memories"],
        ),
        TurnSpec(
            "my right wrist gets sore after long trackpad sessions",
            ["body"],
        ),
        TurnSpec(
            "Sam texted about lunch and I felt weirdly anxious about it",
            ["people", "psychology"],
            "split: Sam → people, anxious → psychology",
        ),
        TurnSpec(
            "dentist appointment Tuesday at 3 and pick up oat milk on the way home",
            [],
            allow_discard_only=True,
            note="schedule + errand → discard until Calendar exists",
        ),
        TurnSpec("thanks, that's all for now", [], allow_discard_only=True),
        TurnSpec("gg", [], allow_discard_only=True),
    ]


@dataclass
class OACQuerySpec:
    question: str
    must_find: list[str]
    hint: str = ""


def oac_queries() -> list[OACQuerySpec]:
    """Vague questions — no filenames, no dest hints."""
    return [
        OACQuerySpec(
            "who have I been hanging out with lately?",
            ["Alex", "Sam", "climbing", "lunch"],
        ),
        OACQuerySpec(
            "what's running low at home?",
            ["oat milk", "dish soap"],
        ),
        OACQuerySpec(
            "anything about how I focus or take breaks?",
            ["focus", "25", "stretch"],
        ),
        OACQuerySpec(
            "do you remember anything about sleep or caffeine for me?",
            ["matcha", "espresso", "sleep"],
        ),
        OACQuerySpec(
            "what makes me anxious?",
            ["anxious", "open loops"],
        ),
        OACQuerySpec(
            "what kind of person am I when it comes to quality?",
            ["craftsmanship", "half-baked"],
        ),
        OACQuerySpec(
            "did I ask about nuclear stuff?",
            ["fuel", "reactor", "spent"],
        ),
        OACQuerySpec(
            "any big wins I mentioned?",
            ["ESP32", "mic", "pipeline"],
        ),
    ]


# ---------------------------------------------------------------------------
# Inspect sandbox after SOI
# ---------------------------------------------------------------------------


def _flatten_knowledge(root: Path) -> dict[str, list[str]]:
    """Return section/file → list of 'key: snippet' strings."""
    out: dict[str, list[str]] = {}

    hayden = _read_json(root / HAYDEN_FILE)
    if isinstance(hayden, dict):
        for section, mapping in hayden.items():
            if section == "version" or not isinstance(mapping, dict):
                continue
            rows: list[str] = []
            for key, entries in mapping.items():
                if not isinstance(entries, list):
                    continue
                texts = [
                    str(e.get("text") or "")[:80]
                    for e in entries
                    if isinstance(e, dict)
                ]
                if texts:
                    rows.append(f"{key}: {texts[-1]}")
            if rows:
                out[f"hayden/{section}"] = rows

    for name in ROOT_FILES:
        if name == HAYDEN_FILE:
            continue
        path = root / name
        if not path.is_file():
            continue
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        rows = []
        for key, entries in data.items():
            if not isinstance(entries, list):
                continue
            texts = [
                str(e.get("text") or "")[:80] for e in entries if isinstance(e, dict)
            ]
            if texts:
                rows.append(f"{key}: {texts[-1]}")
        if rows:
            out[name] = rows
    return out


def _dest_aliases(dest: str) -> set[str]:
    d = dest.casefold()
    aliases = {d}
    if d in {"characteristics", "hayden", "identity", "personality"}:
        aliases |= {"characteristics", "hayden", "preferences", "values"}
    if d == "people":
        aliases.add("people.json")
    if d == "household":
        aliases.add("household.json")
    if d == "habits":
        aliases.add("habits")
    if d == "preferences":
        aliases.add("preferences")
    if d == "psychology":
        aliases.add("psychology")
    if d == "desires":
        aliases.add("desires")
    if d == "questions":
        aliases.add("questions.json")
    if d == "memories":
        aliases.add("memories.json")
    if d == "body":
        aliases.add("body")
    if d == "values":
        aliases.add("values")
    return aliases


def _log_calls_from_trace(trace: ToolTrace) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for row in trace.calls:
        if row.get("tool") != "log_item" or row.get("blocked"):
            continue
        args = row.get("args") or {}
        result = row.get("result_preview")
        if not isinstance(result, dict):
            result = {}
        calls.append(
            {
                "dest": str(args.get("dest") or ""),
                "label": str(args.get("label") or ""),
                "reason": str(args.get("reason") or "")[:120],
                "action": result.get("action"),
                "path": result.get("path"),
            }
        )
    return calls


def evaluate_filing(trace: ToolTrace, turns: list[TurnSpec]) -> dict[str, Any]:
    log_calls = _log_calls_from_trace(trace)
    by_user: dict[str, list[dict[str, Any]]] = {}
    # We don't have entry_id in trace easily tied to user — evaluate globally + per expect
    all_dests = {c["dest"].casefold() for c in log_calls if c.get("dest")}
    discard_calls = [c for c in log_calls if c.get("dest").casefold() == "discard"]

    checks: list[dict[str, Any]] = []
    for spec in turns:
        flat_want: set[str] = set()
        for d in spec.expect_dests:
            flat_want |= _dest_aliases(d)
        if spec.allow_discard_only and not spec.expect_dests:
            ok = bool(discard_calls) or not log_calls
            checks.append(
                {
                    "user": spec.user_text[:60],
                    "expect": "discard",
                    "ok": ok,
                    "note": spec.note,
                }
            )
            continue
        if not spec.expect_dests:
            continue
        hit = any(
            c["dest"].casefold() in flat_want or any(a in c["dest"].casefold() for a in flat_want)
            for c in log_calls
        )
        # Also check label/reason keywords from user text
        if not hit:
            tokens = [t for t in re.split(r"\W+", spec.user_text.casefold()) if len(t) > 3]
            for c in log_calls:
                blob = f"{c.get('label', '')} {c.get('reason', '')}".casefold()
                if any(t in blob for t in tokens[:4]):
                    hit = True
                    break
        checks.append(
            {
                "user": spec.user_text[:60],
                "expect_dests": spec.expect_dests,
                "ok": hit,
                "note": spec.note,
            }
        )

    passed = sum(1 for c in checks if c.get("ok"))
    return {
        "log_item_calls": len(log_calls),
        "discard_calls": len(discard_calls),
        "dests_used": sorted(all_dests),
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "ok": passed == len(checks) and len(log_calls) > 0,
        "calls": log_calls,
    }


class OACTrace:
    def __init__(self) -> None:
        self.query_calls: list[dict[str, Any]] = []
        self.other_tools: list[str] = []

    def hook(self, session: ChatSession) -> None:
        original = session._run_tool_call

        def wrapped(call: dict[str, Any]) -> dict[str, Any]:
            fn = call.get("function") or {}
            name = fn.get("name") or ""
            raw_args = fn.get("arguments") or {}
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}
            result = original(call)
            if name == "query_db":
                self.query_calls.append({"args": args, "result": result})
            elif name not in {"get_tools", "getTools"}:
                self.other_tools.append(name)
            return result

        session._run_tool_call = wrapped  # type: ignore[method-assign]


def evaluate_oac_retrieval(
    trace: OACTrace,
    specs: list[OACQuerySpec],
    db: DatabaseTools,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        found_in_tools = False
        blob = ""
        for call in trace.query_calls:
            args = call.get("args") or {}
            qtext = str(args.get("q") or args.get("name") or args.get("dest") or "")
            if any(w in qtext.casefold() for w in spec.question.casefold().split()[:3]):
                pass
            result = call.get("result") or {}
            blob += json.dumps(result, ensure_ascii=False).casefold()
        # Also run host query_db as ground truth for the vague question keywords
        keywords = [w for w in re.split(r"\W+", spec.question.casefold()) if len(w) > 3][:5]
        ground = query_db(db, q=" ".join(keywords[:3]))
        ground_blob = json.dumps(ground, ensure_ascii=False).casefold()
        blob += ground_blob

        hits = [m for m in spec.must_find if m.casefold() in blob]
        ok = len(hits) >= max(1, len(spec.must_find) // 2)
        used_query = any(
            call.get("args") for call in trace.query_calls
        )
        rows.append(
            {
                "question": spec.question,
                "must_find": spec.must_find,
                "hits": hits,
                "ok": ok,
                "model_used_query_db": used_query,
            }
        )
    passed = sum(1 for r in rows if r["ok"])
    return {
        "query_db_calls": len(trace.query_calls),
        "other_tools": sorted(set(trace.other_tools)),
        "checks": rows,
        "passed": passed,
        "total": len(rows),
        "ok": passed >= max(1, len(rows) - 1),  # allow one miss on 8B
    }


def evaluate_decay(db: DatabaseTools, root: Path) -> dict[str, Any]:
    from ainet.logstore import observation_remaining, decay_profile, resistance_days
    from datetime import timedelta

    # Inject a stale household observation and prune
    doc = _read_json(root / "household.json")
    old = (datetime.now(timezone.utc) - timedelta(days=60)).replace(microsecond=0).isoformat()
    doc["stale milk"] = [{"time": old, "text": "was out of oat milk"}]
    _write_json(root / "household.json", doc)

    pruned = decay_knowledge(db)
    after = _read_json(root / "household.json")
    stale_gone = "stale milk" not in after

    profile = decay_profile("household", db)
    fresh = observation_remaining(
        {"time": _utc_now(), "text": "just bought soap"},
        1,
        profile,
        datetime.now(timezone.utc),
        db,
    )
    resistance = resistance_days(profile, 1, db)
    personality = decay_profile("characteristics", db)
    long_res = resistance_days(personality, 5, db)

    return {
        "pruned": pruned,
        "stale_household_removed": stale_gone,
        "household_resistance_1_hit": round(resistance, 1),
        "personality_resistance_5_hit": round(long_res, 1),
        "fresh_observation_remaining": round(fresh, 3),
        "ok": stale_gone and resistance <= 10 and long_res >= 300,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _safe_print(obj: Any, limit: int = 4000) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=True, default=str)
    if len(text) > limit:
        text = text[:limit] + "..."
    print(text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stress test SOI filing + OAC query_db")
    p.add_argument("--source-db", type=Path, default=None)
    p.add_argument("--sandbox", type=Path, default=None)
    p.add_argument("--apply", action="store_true", help="Write inside sandbox (required for real test)")
    p.add_argument("--keep", action="store_true")
    p.add_argument("--skip-oac", action="store_true")
    p.add_argument("--skip-decay", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--max-tool-rounds", type=int, default=32)
    p.add_argument("--report", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = build_parser().parse_args(argv)
    source = (args.source_db or default_db_root()).resolve()
    if not source.is_dir():
        print(f"Source db not found: {source}", file=sys.stderr)
        return 2

    own_temp = False
    if args.sandbox:
        sandbox = args.sandbox.resolve()
        try:
            sandbox.relative_to(source)
            print(
                f"--sandbox must not be inside --source-db ({sandbox}). "
                "Use a temp path or e.g. Desktop/ainet-stress-sandbox",
                file=sys.stderr,
            )
            return 2
        except ValueError:
            pass
    else:
        sandbox = Path(tempfile.mkdtemp(prefix="ainet-kb-stress-")).resolve()
        own_temp = True

    print(f"Sandbox: {sandbox}")
    copy_sandbox(source, sandbox)
    _reset_changelog(sandbox)
    _empty_knowledge(sandbox)
    ensure_knowledge_files(sandbox)

    turns = stress_turns()
    store = ConversationStore(sandbox)
    session_id = store.ensure_session(mode_id="companion")
    injected: list[dict[str, Any]] = []
    for spec in turns:
        entry = inject_oac_turn(
            store,
            user_text=spec.user_text,
            assistant_text="(harness — no OAC reply)",
            mode_id="companion",
            session_id=session_id,
        )
        if entry:
            injected.append(entry)
    print(f"Injected {len(injected)} fake OAC turns")

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
    config = replace(config, **updates)

    # Ollama ping
    try:
        models = OllamaClient(config).list_models()
        print(f"Ollama OK @ {config.host} model={config.model} ({len(models)} models)")
    except OllamaError as exc:
        print(f"Ollama unavailable: {exc}", file=sys.stderr)
        print("Start Ollama or set AINET_OLLAMA_HOST / AINET_OLLAMA_MODEL", file=sys.stderr)
        return 3

    dry_run = not args.apply
    if dry_run:
        print("WARNING: pass --apply to actually file into sandbox", file=sys.stderr)

    trace = ToolTrace()
    restores = install_harness(None, dry_run=dry_run, trace=trace, allow_web=False)
    report: dict[str, Any] = {
        "sandbox": str(sandbox),
        "dry_run": dry_run,
        "injected_turns": len(injected),
        "timestamp": _utc_now(),
    }

    try:
        print("\n=== SOI filing ===")
        worker = SOIWorker(config=config)
        try:
            filing = worker.run_filing()
        except Exception as exc:
            filing = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
        report["filing_result"] = filing
        report["knowledge_after_soi"] = _flatten_knowledge(sandbox)
        report["filing_eval"] = evaluate_filing(trace, turns)
        print(json.dumps(report["filing_eval"], indent=2, ensure_ascii=True)[:3000])

        db = DatabaseTools(sandbox)
        if not args.skip_decay:
            print("\n=== Decay ===")
            report["decay_eval"] = evaluate_decay(db, sandbox)
            print(json.dumps(report["decay_eval"], indent=2))

        if not args.skip_oac and args.apply:
            print("\n=== OAC retrieval (vague questions) ===")
            oac_trace = OACTrace()
            oac_session = ChatSession(get_mode("companion"), config=config)
            oac_trace.hook(oac_session)
            oac_replies: list[dict[str, Any]] = []
            for spec in oac_queries():
                print(f"  Q: {spec.question}")
                try:
                    reply = oac_session.ask(spec.question, stream=False)
                except Exception as exc:
                    reply = f"(error: {exc})"
                oac_replies.append({"question": spec.question, "reply": reply[:500]})
            report["oac_replies"] = oac_replies
            report["oac_eval"] = evaluate_oac_retrieval(oac_trace, oac_queries(), db)
            print(json.dumps(report["oac_eval"], indent=2, ensure_ascii=True)[:3000])

        # Host query_db sanity (no model)
        print("\n=== Host query_db sanity ===")
        sanity = []
        for spec in oac_queries():
            q = " ".join(w for w in re.split(r"\W+", spec.question) if len(w) > 3)[:40]
            hit = query_db(db, q=q)
            blob = json.dumps(hit, ensure_ascii=False).casefold()
            found = [m for m in spec.must_find if m.casefold() in blob]
            sanity.append({"question": spec.question, "q": q, "found": found, "ok": bool(found)})
        report["query_sanity"] = sanity
        print(json.dumps(sanity, indent=2, ensure_ascii=True)[:2000])

    finally:
        for restore in restores:
            try:
                restore()
            except Exception:
                pass

    filing_ok = report.get("filing_eval", {}).get("ok", False)
    decay_ok = report.get("decay_eval", {}).get("ok", True)
    oac_ok = report.get("oac_eval", {}).get("ok", True if args.skip_oac else False)
    sanity_ok = all(r.get("ok") for r in report.get("query_sanity") or [])
    report["summary"] = {
        "filing_ok": filing_ok,
        "decay_ok": decay_ok,
        "oac_ok": oac_ok,
        "query_sanity_ok": sanity_ok,
        "overall_ok": filing_ok and decay_ok and sanity_ok and (oac_ok or args.skip_oac or not args.apply),
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(report["summary"], indent=2))

    out = args.report or (sandbox / "stress-report.json")
    slim = {
        "summary": report.get("summary"),
        "filing_eval": {
            k: report["filing_eval"].get(k)
            for k in ("ok", "passed", "total", "log_item_calls", "discard_calls", "dests_used", "checks")
            if report.get("filing_eval")
        },
        "decay_eval": report.get("decay_eval"),
        "oac_eval": {
            k: report["oac_eval"].get(k)
            for k in ("ok", "passed", "total", "query_db_calls", "other_tools", "checks")
            if report.get("oac_eval")
        },
        "query_sanity": report.get("query_sanity"),
        "knowledge_after_soi": report.get("knowledge_after_soi"),
        "sandbox": str(sandbox),
    }
    _write_json(out, slim)
    _write_json(out.with_name("stress-report-full.json"), report)
    print(f"\nFull report: {out}")

    if own_temp and not args.keep:
        print(f"Keeping sandbox for inspection: {sandbox}")
    elif args.keep:
        print(f"Sandbox: {sandbox}")

    return 0 if report["summary"]["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
