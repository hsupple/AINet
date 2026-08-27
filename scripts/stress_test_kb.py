#!/usr/bin/env python3
"""End-to-end stress test: fake OAC turns → SOI filing → score wrong dests/labels/meta.

Resets knowledge + changelog inside an isolated sandbox (never touches live db/ writes
outside the sandbox). Feeds a large curated corpus, drains SOI across multiple
run_filing passes, then reports where things landed and what went wrong.

Examples:
  python scripts/stress_test_kb.py --apply --keep
  python scripts/stress_test_kb.py --apply --keep --size large
  python scripts/stress_test_kb.py --apply --size small
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
    expect_labels: list[str] = field(default_factory=list)
    forbid_labels: list[str] = field(default_factory=list)
    category: str = "fact"  # fact | discard | retrieval | split | trap


_META_REASON_RE = re.compile(
    r"(?:"
    r"^(?:hayden |user )?(?:asked about|asked who|asked what|asked if|inquired|requested)|"
    r"user asked|confirmed identity|looked up|from the database|"
    r"query_db|hayden\.json|retrieved"
    r")",
    re.I,
)

_BAD_PEOPLE_LABELS = frozenset(
    {
        "friend",
        "friends",
        "people",
        "person",
        "family",
        "relationship",
        "relationships",
        "best friend",
        "social",
    }
)

_BAD_HAYDEN_LABELS = frozenset(
    {
        "characteristics",
        "identity",
        "hayden",
        "traits",
        "trait",
        "self",
        "personality traits",
    }
)


def stress_turns(*, size: str = "large") -> list[TurnSpec]:
    """Curated fake OAC turns. size=small (~14) or large (~90)."""
    small = _stress_turns_core()
    if size == "small":
        return small
    return small + _stress_turns_extra()


def _stress_turns_core() -> list[TurnSpec]:
    return [
        TurnSpec(
            "Alex came over last night, we're friends from the climbing gym",
            ["people"],
            note="person → people.json key Alex",
            expect_labels=["Alex"],
            forbid_labels=["friends", "friend"],
            category="fact",
        ),
        TurnSpec(
            "I've been doing 25 minute focus blocks with a 5 minute stretch after lunch",
            ["habits"],
            category="fact",
        ),
        TurnSpec(
            "switching to matcha in the afternoon because espresso wrecks my sleep",
            ["preferences", "habits"],
            category="split",
        ),
        TurnSpec(
            "we're basically out of oat milk and the dish soap is almost gone",
            ["household"],
            category="fact",
        ),
        TurnSpec(
            "open loops still make me anxious until I write a concrete next action",
            ["psychology"],
            category="fact",
        ),
        TurnSpec(
            "I care more about craftsmanship than shipping half-baked systems",
            ["values", "characteristics", "hayden"],
            forbid_labels=["characteristics", "identity"],
            category="fact",
        ),
        TurnSpec(
            "long term I want to trust my personal database again",
            ["desires"],
            category="fact",
        ),
        TurnSpec(
            "what happens to spent fuel rods after they pull them out of a reactor?",
            ["questions"],
            category="fact",
        ),
        TurnSpec(
            "remember when the ESP32 mic pipeline finally worked? still a huge win",
            ["memories"],
            category="fact",
        ),
        TurnSpec(
            "my right wrist gets sore after long trackpad sessions",
            ["body"],
            category="fact",
        ),
        TurnSpec(
            "Sam texted about lunch and I felt weirdly anxious about it",
            ["people", "psychology"],
            note="split: Sam → people, anxious → psychology",
            expect_labels=["Sam"],
            forbid_labels=["friends"],
            category="split",
        ),
        TurnSpec(
            "dentist appointment Tuesday at 3 and pick up oat milk on the way home",
            [],
            allow_discard_only=True,
            note="schedule + errand → discard until Calendar exists",
            category="discard",
        ),
        TurnSpec(
            "thanks, that's all for now",
            [],
            allow_discard_only=True,
            category="discard",
        ),
        TurnSpec("gg", [], allow_discard_only=True, category="discard"),
    ]


def _stress_turns_extra() -> list[TurnSpec]:
    """Large corpus: more facts, splits, retrieval traps, greetings, schedule noise."""
    people = [
        TurnSpec(
            "Powell Williams is my best friend but I never call him that out loud",
            ["people"],
            expect_labels=["Powell Williams", "Powell"],
            forbid_labels=["friends", "best friend", "friend"],
            category="fact",
        ),
        TurnSpec(
            "Maya and I pair-programmed on the ROS2 stack this weekend",
            ["people"],
            expect_labels=["Maya"],
            forbid_labels=["friends", "people"],
            category="fact",
        ),
        TurnSpec(
            "Jordan ghosted the group chat after the party and it pissed me off",
            ["people", "psychology"],
            expect_labels=["Jordan"],
            category="split",
        ),
        TurnSpec(
            "I introduced Riley to the climbing gym crowd",
            ["people"],
            expect_labels=["Riley"],
            forbid_labels=["friends"],
            category="fact",
        ),
        TurnSpec(
            "my mom keeps asking when I'm coming home for break",
            ["people"],
            expect_labels=["mom", "Mom"],
            forbid_labels=["family", "people"],
            category="fact",
        ),
        TurnSpec(
            "Chris from Purdue Autonomous Racing asked if I can help with the battery pack",
            ["people"],
            expect_labels=["Chris"],
            category="fact",
        ),
        TurnSpec(
            "Nina brought over dumplings and we talked until 2am",
            ["people"],
            expect_labels=["Nina"],
            category="fact",
        ),
        TurnSpec(
            "I told Theo I'm not dating anyone right now and he was weirdly relieved",
            ["people"],
            expect_labels=["Theo"],
            category="fact",
        ),
    ]

    habits_prefs = [
        TurnSpec(
            "I start every morning with a cold shower then coffee black",
            ["habits", "preferences"],
            category="split",
        ),
        TurnSpec(
            "I refuse to use Notion — plain markdown files only",
            ["preferences", "habits"],
            category="split",
        ),
        TurnSpec(
            "I do a 40-minute run three times a week before dinner",
            ["habits"],
            category="fact",
        ),
        TurnSpec(
            "I hate when podcasts put ads mid-sentence",
            ["preferences"],
            category="fact",
        ),
        TurnSpec(
            "I've been sleeping with earplugs because the apartment vents are loud",
            ["habits", "body"],
            category="split",
        ),
        TurnSpec(
            "I batch laundry on Sundays and refuse to do it midweek",
            ["habits"],
            category="fact",
        ),
        TurnSpec(
            "dark mode everywhere, I can't stand bright UIs at night",
            ["preferences"],
            category="fact",
        ),
        TurnSpec(
            "I keep a standing desk and switch to sitting after 90 minutes",
            ["habits"],
            category="fact",
        ),
    ]

    psychology_values = [
        TurnSpec(
            "social small talk drains me faster than hard technical work",
            ["psychology"],
            category="fact",
        ),
        TurnSpec(
            "I get defensive when people imply my projects are just toys",
            ["psychology"],
            category="fact",
        ),
        TurnSpec(
            "honesty matters more to me than being liked in the moment",
            ["values"],
            category="fact",
        ),
        TurnSpec(
            "I feel guilty when I skip a workout even if I'm sick",
            ["psychology", "habits"],
            category="split",
        ),
        TurnSpec(
            "deadlines stress me out until the first concrete commit lands",
            ["psychology"],
            category="fact",
        ),
        TurnSpec(
            "I value depth over breadth — I'd rather master one system than dabble",
            ["values", "hayden", "characteristics"],
            forbid_labels=["characteristics", "identity"],
            category="fact",
        ),
        TurnSpec(
            "I shut down emotionally when conversations get vague and motivational",
            ["psychology", "preferences"],
            category="split",
        ),
    ]

    hayden_identity = [
        TurnSpec(
            "I'm a mechanical engineering student at Purdue, class of 2027",
            ["hayden", "characteristics"],
            expect_labels=["education"],
            forbid_labels=["characteristics", "identity", "hayden"],
            category="fact",
        ),
        TurnSpec(
            "people say I'm intensely curious — I keep asking why until it bottoms out",
            ["hayden", "characteristics"],
            expect_labels=["curiosity", "personality"],
            category="fact",
        ),
        TurnSpec(
            "I rebuilt a BMW 540i just to learn the powertrain end to end",
            ["hayden", "characteristics", "memories"],
            expect_labels=["experience", "projects", "personality"],
            category="fact",
        ),
        TurnSpec(
            "that's so me — I always decompose systems before I touch the UI",
            ["hayden", "characteristics"],
            expect_labels=["personality"],
            category="fact",
        ),
        TurnSpec(
            "I prefer objective technical answers over pep talks",
            ["hayden", "characteristics", "preferences"],
            expect_labels=["personality", "preferences"],
            category="fact",
        ),
    ]

    desires_body_secrets = [
        TurnSpec(
            "eventually I want to ship a personal AI that actually remembers me correctly",
            ["desires"],
            category="fact",
        ),
        TurnSpec(
            "I want to get strong enough to do 10 clean pull-ups",
            ["desires"],
            category="fact",
        ),
        TurnSpec(
            "my lower back flares up if I sit for more than two hours straight",
            ["body"],
            category="fact",
        ),
        TurnSpec(
            "my left knee still clicks after that trail run last fall",
            ["body"],
            category="fact",
        ),
        TurnSpec(
            "the wifi password for home is in the router sticker — don't put it in chat logs",
            ["secrets"],
            category="fact",
        ),
        TurnSpec(
            "my student ID PIN is private, never store it in plain notes",
            ["secrets"],
            category="fact",
        ),
    ]

    household_questions_memories = [
        TurnSpec(
            "we're out of paper towels and the vacuum bag is full",
            ["household"],
            category="fact",
        ),
        TurnSpec(
            "the bathroom faucet drips unless you twist it past the detent",
            ["household"],
            category="fact",
        ),
        TurnSpec(
            "how does a differential gearset keep the outer wheel spinning faster in a turn?",
            ["questions"],
            category="fact",
        ),
        TurnSpec(
            "why do GPUs win at matrix multiply versus CPUs for inference?",
            ["questions"],
            category="fact",
        ),
        TurnSpec(
            "remember the night the dorm fire alarm went off during finals week?",
            ["memories"],
            category="fact",
        ),
        TurnSpec(
            "winning the hackathon with the ePaper shelf-tag demo still feels unreal",
            ["memories"],
            category="fact",
        ),
    ]

    more_splits = [
        TurnSpec(
            "Kai cancelled plans and now I'm spiraling about whether I'm annoying",
            ["people", "psychology"],
            expect_labels=["Kai"],
            category="split",
        ),
        TurnSpec(
            "I want to learn Mandarin but Duolingo streaks make me anxious",
            ["desires", "psychology", "habits"],
            category="split",
        ),
        TurnSpec(
            "Priya and I argued about ethics in AI and I left feeling drained",
            ["people", "psychology", "values"],
            expect_labels=["Priya"],
            category="split",
        ),
        TurnSpec(
            "I bought better coffee beans and now I brew pour-over every morning",
            ["preferences", "habits", "household"],
            category="split",
        ),
    ]

    # Retrieval / meta traps — MUST discard, not file "user asked..."
    retrieval_traps = [
        TurnSpec(
            "who am I?",
            [],
            allow_discard_only=True,
            note="retrieval — discard",
            category="retrieval",
        ),
        TurnSpec(
            "what am I like?",
            [],
            allow_discard_only=True,
            category="retrieval",
        ),
        TurnSpec(
            "what's my personality like?",
            [],
            allow_discard_only=True,
            category="retrieval",
        ),
        TurnSpec(
            "figure out my characteristics from the database",
            [],
            allow_discard_only=True,
            category="retrieval",
        ),
        TurnSpec(
            "who are my friends?",
            [],
            allow_discard_only=True,
            category="retrieval",
        ),
        TurnSpec(
            "what do you know about me?",
            [],
            allow_discard_only=True,
            category="retrieval",
        ),
        TurnSpec(
            "look up who I am in hayden.json",
            [],
            allow_discard_only=True,
            category="retrieval",
        ),
        TurnSpec(
            "query_db for my curiosity",
            [],
            allow_discard_only=True,
            category="retrieval",
        ),
        TurnSpec(
            "tell me what I'm like again",
            [],
            allow_discard_only=True,
            category="retrieval",
        ),
        TurnSpec(
            "do I have Powell Williams filed as a friend?",
            [],
            allow_discard_only=True,
            note="confirm stored fact — discard, don't re-file confirmation",
            category="retrieval",
        ),
    ]

    greetings_acks = [
        TurnSpec(t, [], allow_discard_only=True, category="discard")
        for t in (
            "hi",
            "hey",
            "hello",
            "thanks",
            "thank you",
            "ok",
            "okay",
            "cool",
            "yeah",
            "yep",
            "go on",
            "sounds good",
            "got it",
            "lol",
            "bye",
            "good night",
            "np",
        )
    ]

    schedule_noise = [
        TurnSpec(
            "meeting with advisor Thursday at 10am",
            [],
            allow_discard_only=True,
            category="discard",
        ),
        TurnSpec(
            "remind me to email the TA tomorrow morning",
            [],
            allow_discard_only=True,
            category="discard",
        ),
        TurnSpec(
            "lab starts at 2 and I need to grab lunch before that",
            [],
            allow_discard_only=True,
            category="discard",
        ),
        TurnSpec(
            "calendar: dentist next Monday, oil change Wednesday",
            [],
            allow_discard_only=True,
            category="discard",
        ),
    ]

    traps = [
        TurnSpec(
            "my friends are important to me — especially the climbing crew",
            ["people", "values"],
            note="temptation to label=friends; should name people or file values",
            forbid_labels=["friends"],
            category="trap",
        ),
        TurnSpec(
            "update my identity / characteristics with how curious I am",
            ["hayden", "characteristics"],
            note="temptation to invent a meta key; fold into personality/curiosity",
            expect_labels=["curiosity", "personality"],
            category="trap",
        ),
        TurnSpec(
            "add a people entry under friends for Alex",
            ["people"],
            note="explicit bad instruction — still must use label=Alex",
            expect_labels=["Alex"],
            forbid_labels=["friends"],
            category="trap",
        ),
    ]

    filler_facts = [
        TurnSpec(
            f"I learned something new about {topic} and want to dig deeper later",
            ["desires", "questions", "hayden", "characteristics"],
            category="fact",
        )
        for topic in (
            "topology optimization",
            "ROS2 lifecycle nodes",
            "battery thermal runaway",
            "FFT windowing",
            "injection molding tolerances",
            "Kalman filters",
            "I2C clock stretching",
            "orbital mechanics Hohmann transfers",
        )
    ]

    return (
        people
        + habits_prefs
        + psychology_values
        + hayden_identity
        + desires_body_secrets
        + household_questions_memories
        + more_splits
        + retrieval_traps
        + greetings_acks
        + schedule_noise
        + traps
        + filler_facts
    )


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
        ids: list[str] = []
        for raw in list(args.get("entry_ids") or []) + [args.get("entry_id") or ""]:
            eid = str(raw or "").strip()
            if eid and eid not in ids:
                ids.append(eid)
        for raw in list(result.get("entry_ids") or []) + [result.get("entry_id") or ""]:
            eid = str(raw or "").strip()
            if eid and eid not in ids:
                ids.append(eid)
        calls.append(
            {
                "dest": str(args.get("dest") or result.get("dest") or ""),
                "label": str(args.get("label") or result.get("name") or ""),
                "reason": str(args.get("reason") or "")[:200],
                "action": result.get("action"),
                "path": result.get("path"),
                "entry_ids": ids,
            }
        )
    return calls


def _normalize_dest(dest: str) -> str:
    d = (dest or "").strip().casefold().replace("\\", "/")
    if d.endswith(".json"):
        d = d[: -len(".json")]
    if d in {"characteristics", "identity", "personality"}:
        return "hayden"
    return d


def evaluate_filing(
    trace: ToolTrace,
    turns: list[TurnSpec],
    injected: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score each injected turn against log_item calls (by entry_id when possible)."""
    log_calls = _log_calls_from_trace(trace)
    id_to_spec: dict[str, TurnSpec] = {}
    id_to_user: dict[str, str] = {}
    for entry, spec in zip(injected, turns):
        eid = str(entry.get("id") or "")
        if not eid:
            continue
        id_to_spec[eid] = spec
        details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
        id_to_user[eid] = str(details.get("user_text") or entry.get("summary") or "")

    calls_by_id: dict[str, list[dict[str, Any]]] = {eid: [] for eid in id_to_spec}
    orphan_calls: list[dict[str, Any]] = []
    for call in log_calls:
        matched = False
        for eid in call.get("entry_ids") or []:
            if eid in calls_by_id:
                calls_by_id[eid].append(call)
                matched = True
        if not matched:
            orphan_calls.append(call)

    checks: list[dict[str, Any]] = []
    wrong: list[dict[str, Any]] = []
    stats = {
        "meta_reasons": 0,
        "bad_people_labels": 0,
        "bad_hayden_labels": 0,
        "narrow_phrase_keys": 0,
        "overlong_labels": 0,
        "wrong_dest": 0,
        "missed_discard": 0,
        "missed_expected_dest": 0,
        "filed_retrieval": 0,
        "ok_turns": 0,
        "fail_turns": 0,
    }

    for eid, spec in id_to_spec.items():
        calls = calls_by_id.get(eid) or []
        dests = {_normalize_dest(c["dest"]) for c in calls if c.get("dest")}
        labels = [str(c.get("label") or "").strip() for c in calls]
        reasons = [str(c.get("reason") or "") for c in calls]
        non_discard = [c for c in calls if _normalize_dest(c.get("dest") or "") != "discard"]
        discarded = any(_normalize_dest(c.get("dest") or "") == "discard" for c in calls)

        issues: list[str] = []

        for reason, call in zip(reasons, calls):
            dest_n = _normalize_dest(call.get("dest") or "")
            if dest_n in {"discard", "questions"}:
                continue
            if _META_REASON_RE.search(reason):
                issues.append(f"meta_reason:{reason[:80]}")
                stats["meta_reasons"] += 1

        for lab, call in zip(labels, calls):
            if _normalize_dest(call.get("dest") or "") == "discard":
                continue
            low = re.sub(r"\s+", " ", lab).casefold()
            if low in _BAD_PEOPLE_LABELS:
                issues.append(f"bad_people_label:{lab}")
                stats["bad_people_labels"] += 1
            if low in _BAD_HAYDEN_LABELS:
                issues.append(f"bad_hayden_label:{lab}")
                stats["bad_hayden_labels"] += 1
            # Narrow phrase-as-key: "preference_for_accuracy", long underscore titles, etc.
            if "_" in lab and lab.count("_") >= 2:
                issues.append(f"narrow_phrase_key:{lab}")
                stats["narrow_phrase_keys"] = stats.get("narrow_phrase_keys", 0) + 1
            elif len(lab.split()) >= 5:
                issues.append(f"overlong_label:{lab}")
                stats["overlong_labels"] = stats.get("overlong_labels", 0) + 1
            for forbid in spec.forbid_labels:
                if low == forbid.casefold():
                    issues.append(f"forbid_label:{lab}")

        if spec.allow_discard_only or spec.category in {"discard", "retrieval"}:
            if non_discard:
                issues.append(
                    "should_discard_but_filed:"
                    + ",".join(sorted({_normalize_dest(c["dest"]) for c in non_discard}))
                )
                stats["missed_discard"] += 1
                if spec.category == "retrieval":
                    stats["filed_retrieval"] += 1
            elif not calls and not discarded:
                # Host may auto-mark ephemeral without log_item — treat as ok for greetings
                if spec.category == "retrieval":
                    issues.append("retrieval_unhandled")
                    stats["missed_discard"] += 1
        else:
            want: set[str] = set()
            for d in spec.expect_dests:
                want.add(_normalize_dest(d))
                want |= {_normalize_dest(a) for a in _dest_aliases(d)}
            want.discard("discard")
            hit = bool(want & dests)
            if not hit and want:
                # soft match via reason/label tokens
                tokens = [t for t in re.split(r"\W+", spec.user_text.casefold()) if len(t) > 3]
                soft = False
                for c in non_discard:
                    blob = f"{c.get('label', '')} {c.get('reason', '')}".casefold()
                    if any(t in blob for t in tokens[:5]):
                        soft = True
                        break
                if not soft:
                    issues.append(f"missed_dest:{sorted(want)}")
                    stats["missed_expected_dest"] += 1
                    stats["wrong_dest"] += 1
            if spec.expect_labels and non_discard:
                label_hit = any(
                    any(want_lab.casefold() in lab.casefold() for want_lab in spec.expect_labels)
                    for lab in labels
                    if lab
                )
                if not label_hit:
                    issues.append(f"missed_label:{spec.expect_labels}")

        ok = not issues
        if ok:
            stats["ok_turns"] += 1
        else:
            stats["fail_turns"] += 1
            wrong.append(
                {
                    "id": eid,
                    "user": (id_to_user.get(eid) or spec.user_text)[:100],
                    "category": spec.category,
                    "expect_dests": spec.expect_dests,
                    "got_dests": sorted(dests),
                    "got_labels": labels,
                    "got_reasons": [r[:100] for r in reasons],
                    "issues": issues,
                    "note": spec.note,
                }
            )

        checks.append(
            {
                "id": eid,
                "user": spec.user_text[:70],
                "category": spec.category,
                "expect_dests": spec.expect_dests,
                "got_dests": sorted(dests),
                "got_labels": labels,
                "ok": ok,
                "issues": issues,
                "note": spec.note,
            }
        )

    # Knowledge-side smell test (independent of entry linkage)
    knowledge_smells: list[str] = []
    all_dests = sorted({_normalize_dest(c["dest"]) for c in log_calls if c.get("dest")})
    discard_calls = [c for c in log_calls if _normalize_dest(c.get("dest") or "") == "discard"]
    filed_calls = [c for c in log_calls if _normalize_dest(c.get("dest") or "") != "discard"]

    passed = stats["ok_turns"]
    total = len(checks)
    return {
        "log_item_calls": len(log_calls),
        "filed_calls": len(filed_calls),
        "discard_calls": len(discard_calls),
        "orphan_calls": len(orphan_calls),
        "dests_used": all_dests,
        "checks": checks,
        "wrong": wrong,
        "wrong_count": len(wrong),
        "passed": passed,
        "total": total,
        "accuracy": round(passed / total, 3) if total else 0.0,
        "error_stats": stats,
        "knowledge_smells": knowledge_smells,
        "ok": stats["fail_turns"] == 0 and len(log_calls) > 0,
        "calls": log_calls,
        "orphan_call_samples": orphan_calls[:20],
    }


def score_knowledge_dump(root: Path) -> dict[str, Any]:
    """Inspect written JSON for classic SOI mistakes."""
    smells: list[dict[str, Any]] = []
    flat = _flatten_knowledge(root)
    people_path = root / "people.json"
    if people_path.is_file():
        people = _read_json(people_path)
        if isinstance(people, dict):
            for key in people:
                low = re.sub(r"\s+", " ", str(key)).casefold()
                if low in _BAD_PEOPLE_LABELS:
                    smells.append({"where": "people.json", "issue": "bad_key", "key": key})
                entries = people.get(key)
                if isinstance(entries, list):
                    for e in entries:
                        if isinstance(e, dict) and _META_REASON_RE.search(str(e.get("text") or "")):
                            smells.append(
                                {
                                    "where": f"people.json/{key}",
                                    "issue": "meta_text",
                                    "text": str(e.get("text") or "")[:120],
                                }
                            )

    hayden_path = root / HAYDEN_FILE
    if hayden_path.is_file():
        hayden = _read_json(hayden_path)
        chars = hayden.get("characteristics") if isinstance(hayden, dict) else None
        if isinstance(chars, dict):
            for key, entries in chars.items():
                low = re.sub(r"\s+", " ", str(key)).casefold()
                if low in _BAD_HAYDEN_LABELS:
                    smells.append({"where": "hayden/characteristics", "issue": "bad_key", "key": key})
                if isinstance(entries, list):
                    for e in entries:
                        if isinstance(e, dict) and _META_REASON_RE.search(str(e.get("text") or "")):
                            smells.append(
                                {
                                    "where": f"hayden/characteristics/{key}",
                                    "issue": "meta_text",
                                    "text": str(e.get("text") or "")[:120],
                                }
                            )

    # Count observations per section
    counts: dict[str, int] = {}
    for section, rows in flat.items():
        counts[section] = len(rows)

    return {
        "smells": smells,
        "smell_count": len(smells),
        "section_counts": counts,
        "flat_preview": {k: v[:8] for k, v in flat.items()},
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
    p.add_argument(
        "--size",
        choices=("small", "large"),
        default="large",
        help="Turn corpus size (default: large ~90 turns)",
    )
    p.add_argument(
        "--max-filing-passes",
        type=int,
        default=20,
        help="SOI run_filing is capped at 6 batches (~24 turns); loop until drained",
    )
    p.add_argument("--model", default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--max-tool-rounds", type=int, default=32)
    p.add_argument("--report", type=Path, default=None)
    return p


def _run_filing_until_drained(
    worker: SOIWorker,
    *,
    max_passes: int,
) -> dict[str, Any]:
    """SOIWorker.run_filing stops after 6 batches; drain the whole queue."""
    passes: list[dict[str, Any]] = []
    totals = {
        "processed_changelog": 0,
        "marked_filed": 0,
        "marked_discarded": 0,
        "mutating_tool_calls": 0,
        "batches": 0,
        "passes": 0,
    }
    for i in range(max(1, max_passes)):
        pending_before = len(worker.pending_changelog())
        if pending_before == 0 and i > 0:
            break
        result = worker.run_filing()
        passes.append(
            {
                "pass": i + 1,
                "pending_before": pending_before,
                "pending_remaining": result.get("pending_remaining"),
                "marked_filed": result.get("marked_filed"),
                "marked_discarded": result.get("marked_discarded"),
                "batches": result.get("batches"),
                "ok": result.get("ok"),
                "error": (result.get("errors") or [None])[0] if result.get("errors") else result.get("error"),
            }
        )
        totals["passes"] += 1
        totals["processed_changelog"] += int(result.get("processed_changelog") or 0)
        totals["marked_filed"] += int(result.get("marked_filed") or 0)
        totals["marked_discarded"] += int(result.get("marked_discarded") or 0)
        totals["mutating_tool_calls"] += int(result.get("mutating_tool_calls") or 0)
        totals["batches"] += int(result.get("batches") or 0)
        if not result.get("ran"):
            break
        if int(result.get("pending_remaining") or 0) == 0:
            break
        # Stuck: no progress this pass — later passes may retry skipped ids
        if (
            int(result.get("marked_filed") or 0) == 0
            and int(result.get("marked_discarded") or 0) == 0
            and int(result.get("mutating_tool_calls") or 0) == 0
            and int(result.get("pending_remaining") or 0) == pending_before
        ):
            break
    totals["pending_remaining"] = len(worker.pending_changelog())
    totals["pass_details"] = passes
    totals["ok"] = all(p.get("ok", True) for p in passes) if passes else False
    return totals


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
    print(f"Corpus size: {args.size}")
    copy_sandbox(source, sandbox)
    _reset_changelog(sandbox)
    _empty_knowledge(sandbox)
    ensure_knowledge_files(sandbox)

    turns = stress_turns(size=args.size)
    by_cat: dict[str, int] = {}
    for t in turns:
        by_cat[t.category] = by_cat.get(t.category, 0) + 1
    print(f"Turn mix: {json.dumps(by_cat, sort_keys=True)} (total={len(turns)})")

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
    print(f"Injected {len(injected)} fake OAC turns (reset knowledge + changelog)")

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

    # Large corpus skips OAC by default (SOI-focused); pass --size small without --skip-oac for OAC.
    skip_oac = bool(args.skip_oac) or args.size == "large"
    if args.size == "large" and not args.skip_oac:
        print("Note: large corpus — skipping OAC retrieval (use --size small to include it)")

    trace = ToolTrace()
    restores = install_harness(None, dry_run=dry_run, trace=trace, allow_web=False)
    report: dict[str, Any] = {
        "sandbox": str(sandbox),
        "dry_run": dry_run,
        "size": args.size,
        "injected_turns": len(injected),
        "turn_mix": by_cat,
        "timestamp": _utc_now(),
    }

    try:
        print("\n=== SOI filing (multi-pass drain) ===")
        worker = SOIWorker(config=config)
        try:
            filing = _run_filing_until_drained(worker, max_passes=args.max_filing_passes)
        except Exception as exc:
            filing = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
        report["filing_result"] = filing
        print(json.dumps({k: filing.get(k) for k in (
            "ok", "passes", "batches", "processed_changelog", "marked_filed",
            "marked_discarded", "mutating_tool_calls", "pending_remaining",
        ) if isinstance(filing, dict)}, indent=2))

        report["knowledge_after_soi"] = _flatten_knowledge(sandbox)
        report["knowledge_score"] = score_knowledge_dump(sandbox)
        report["filing_eval"] = evaluate_filing(trace, turns, injected)

        fe = report["filing_eval"]
        print("\n=== Filing score ===")
        print(
            json.dumps(
                {
                    "accuracy": fe.get("accuracy"),
                    "passed": fe.get("passed"),
                    "total": fe.get("total"),
                    "wrong_count": fe.get("wrong_count"),
                    "error_stats": fe.get("error_stats"),
                    "dests_used": fe.get("dests_used"),
                    "log_item_calls": fe.get("log_item_calls"),
                    "filed_calls": fe.get("filed_calls"),
                    "discard_calls": fe.get("discard_calls"),
                },
                indent=2,
                ensure_ascii=True,
            )
        )
        print("\n=== Wrong filings (up to 40) ===")
        _safe_print(fe.get("wrong") or [], limit=12000)
        print("\n=== Knowledge smells ===")
        _safe_print(report["knowledge_score"], limit=6000)
        print("\n=== Where stuff landed ===")
        _safe_print(report["knowledge_score"].get("section_counts"), limit=2000)
        _safe_print(report["knowledge_after_soi"], limit=8000)

        db = DatabaseTools(sandbox)
        if not args.skip_decay:
            print("\n=== Decay ===")
            report["decay_eval"] = evaluate_decay(db, sandbox)
            print(json.dumps(report["decay_eval"], indent=2))

        if not skip_oac and args.apply:
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
    oac_ok = report.get("oac_eval", {}).get("ok", True if skip_oac else False)
    sanity_ok = all(r.get("ok") for r in report.get("query_sanity") or [])
    smell_ok = (report.get("knowledge_score") or {}).get("smell_count", 0) == 0
    accuracy = float((report.get("filing_eval") or {}).get("accuracy") or 0)
    report["summary"] = {
        "filing_ok": filing_ok,
        "filing_accuracy": accuracy,
        "wrong_count": (report.get("filing_eval") or {}).get("wrong_count"),
        "error_stats": (report.get("filing_eval") or {}).get("error_stats"),
        "knowledge_smells": (report.get("knowledge_score") or {}).get("smell_count"),
        "section_counts": (report.get("knowledge_score") or {}).get("section_counts"),
        "decay_ok": decay_ok,
        "oac_ok": oac_ok,
        "query_sanity_ok": sanity_ok,
        "overall_ok": (
            accuracy >= 0.75
            and smell_ok
            and decay_ok
            and sanity_ok
            and (oac_ok or skip_oac or not args.apply)
        ),
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=True))

    out = args.report or (sandbox / "stress-report.json")
    slim = {
        "summary": report.get("summary"),
        "filing_eval": {
            k: report["filing_eval"].get(k)
            for k in (
                "ok",
                "accuracy",
                "passed",
                "total",
                "wrong_count",
                "error_stats",
                "log_item_calls",
                "filed_calls",
                "discard_calls",
                "dests_used",
                "wrong",
            )
            if report.get("filing_eval")
        },
        "knowledge_score": report.get("knowledge_score"),
        "filing_result": {
            k: (report.get("filing_result") or {}).get(k)
            for k in (
                "ok",
                "passes",
                "batches",
                "processed_changelog",
                "marked_filed",
                "marked_discarded",
                "pending_remaining",
            )
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
    print(f"\nReport: {out}")
    print(f"Full report: {out.with_name('stress-report-full.json')}")
    print(f"Sandbox: {sandbox}")

    if own_temp and not args.keep:
        print("(temp sandbox left on disk for inspection; pass --keep to make that explicit)")
    return 0 if report["summary"]["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
