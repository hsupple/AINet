#!/usr/bin/env python3
"""OAC stress test: seed unique DB facts, ask vague personal questions, prove query_db.

Writes stay inside a sandbox. Scores each question on:
  - query_db called (required for personal questions)
  - web_search / web_fetch NOT used
  - reply (or tool digest) contains seeded evidence tokens

Examples:
  python -u scripts/stress_test_oac.py --apply --keep
  python -u scripts/stress_test_oac.py --apply --keep --sandbox Desktop/ainet-oac-stress
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import traceback
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ainet.logstore import HAYDEN_FILE, ROOT_FILES, empty_hayden, empty_log, ensure_knowledge_files
from ollama.client import OllamaClient, OllamaError
from ollama.config import OllamaConfig, default_db_root
from ollama.convo_memory import (
    host_fallback_memory,
    is_action_confirm,
    is_short_ack,
    named_search_topic,
    reply_looks_like_clarifier,
    strip_leading_clarifier,
    topic_for_search,
    wants_videos,
)
from ollama.db_query_hints import (
    compact_digest,
    extract_query_tokens,
    hayden_asking_about_self,
    looks_like_empty_therapy,
    looks_like_profile_dump,
    personal_memory_question,
    prefetch_personal_context,
    strip_profile_dump,
    wants_open_web,
    _fold,
)
from ollama.modes import get_mode
from ollama.router import suggest_mode
from ollama.session import ChatSession

from scripts.test_soi import copy_sandbox


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _obs(text: str, when: str | None = None) -> dict[str, str]:
    return {"time": when or _utc_now(), "text": text}


def _empty_knowledge(root: Path) -> None:
    _write_json(root / HAYDEN_FILE, empty_hayden())
    for name in ROOT_FILES:
        if name == HAYDEN_FILE:
            continue
        _write_json(root / name, empty_log())


def seed_knowledge(root: Path) -> dict[str, Any]:
    """Deterministic personal facts with unique tokens the open web will not invent."""
    hayden = empty_hayden()
    hayden["characteristics"] = {
        "personality": [
            _obs(
                "Hayden is intensely curious and keeps asking why until it bottoms out "
                "(marker: ZXQ-PERSONA-CURIOUS)."
            )
        ],
        "education": [
            _obs(
                "Hayden is a mechanical engineering student at Purdue, class of 2027 "
                "(marker: ZXQ-EDU-PURDUE27)."
            )
        ],
        "experience": [
            _obs(
                "Hayden rebuilt a BMW 540i to learn the powertrain end to end "
                "(marker: ZXQ-BMW540I-REBUILD)."
            )
        ],
    }
    hayden["preferences"] = {
        "coffee": [
            _obs(
                "Hayden drinks matcha in the afternoon because espresso wrecks sleep "
                "(marker: ZXQ-MATCHA-NO-ESPRESSO)."
            )
        ],
        "tools": [
            _obs(
                "Hayden refuses Notion and uses plain markdown files only "
                "(marker: ZXQ-MARKDOWN-NOT-NOTION)."
            )
        ],
    }
    hayden["habits"] = {
        "focus": [
            _obs(
                "Hayden does 25-minute focus blocks with a 5-minute stretch after lunch "
                "(marker: ZXQ-FOCUS25-STRETCH)."
            )
        ],
        "running": [
            _obs(
                "Hayden runs 40 minutes three times a week before dinner "
                "(marker: ZXQ-RUN40-X3)."
            )
        ],
        "gym": [
            _obs(
                "Hayden lifts at the campus gym but feels anonymous there, like another "
                "worker ant (marker: ZXQ-GYM-ANT)."
            )
        ],
    }
    hayden["values"] = {
        "craftsmanship": [
            _obs(
                "Hayden values craftsmanship over shipping half-baked systems "
                "(marker: ZXQ-CRAFT-OVER-SHIP)."
            )
        ],
    }
    hayden["psychology"] = {
        "anxiety": [
            _obs(
                "Open loops make Hayden anxious until a concrete next action is written "
                "(marker: ZXQ-OPENLOOPS-ANX)."
            )
        ],
        "friends": [
            _obs(
                "Hayden is good at talking once a group project forces it, but otherwise "
                "finds starting conversations hard (marker: ZXQ-FORCED-TALK)."
            )
        ],
    }
    hayden["desires"] = {
        "memory ai": [
            _obs(
                "Hayden wants a personal AI that actually remembers him correctly "
                "(marker: ZXQ-WANT-MEMORY-AI)."
            )
        ],
    }
    hayden["body"] = {
        "right wrist": [
            _obs(
                "Hayden's right wrist gets sore after long trackpad sessions "
                "(marker: ZXQ-WRIST-TRACKPAD)."
            )
        ],
    }

    people = {
        "Alex": [
            _obs(
                "Alex is a climbing-gym friend who came over last night "
                "(marker: ZXQ-ALEX-CLIMB)."
            )
        ],
        "Powell Williams": [
            _obs(
                "Powell Williams is Hayden's closest friend, though Hayden rarely says that aloud "
                "(marker: ZXQ-POWELL-BEST)."
            )
        ],
        "Maya": [
            _obs(
                "Maya pair-programmed on the ROS2 stack with Hayden this weekend "
                "(marker: ZXQ-MAYA-ROS2)."
            )
        ],
        "Sam": [
            _obs(
                "Sam texted about lunch and Hayden felt unusually anxious about it "
                "(marker: ZXQ-SAM-LUNCH-ANX)."
            )
        ],
    }

    household = {
        "supplies": [
            _obs(
                "Oat milk and dish soap are nearly gone at home "
                "(marker: ZXQ-OATMILK-SOAP)."
            )
        ],
        "faucet": [
            _obs(
                "The bathroom faucet drips unless twisted past the detent "
                "(marker: ZXQ-FAUCET-DETENT)."
            )
        ],
    }

    memories = {
        "ESP32 mic": [
            _obs(
                "The ESP32 mic pipeline finally working was a huge personal win "
                "(marker: ZXQ-ESP32-MIC-WIN)."
            )
        ],
        "hackathon": [
            _obs(
                "Winning the hackathon with the ePaper shelf-tag demo still feels unreal "
                "(marker: ZXQ-EPAPER-HACK)."
            )
        ],
    }

    questions = {
        "spent fuel": [
            _obs(
                "Hayden asked what happens to spent fuel rods after reactor removal "
                "(marker: ZXQ-SPENT-FUEL-Q)."
            )
        ],
        "differentials": [
            _obs(
                "Hayden asked how a differential keeps the outer wheel spinning faster in a turn "
                "(marker: ZXQ-DIFF-GEAR-Q)."
            )
        ],
    }

    secrets = {
        "note": [
            _obs(
                "Hayden asked that private PINs never be stored in plain chat notes "
                "(marker: ZXQ-PIN-POLICY)."
            )
        ],
    }

    _write_json(root / HAYDEN_FILE, hayden)
    _write_json(root / "people.json", people)
    _write_json(root / "household.json", household)
    _write_json(root / "memories.json", memories)
    _write_json(root / "questions.json", questions)
    _write_json(root / "secrets.json", secrets)

    return {
        "hayden": hayden,
        "people": people,
        "household": household,
        "memories": memories,
        "questions": questions,
        "secrets": secrets,
    }


@dataclass
class OACAsk:
    question: str
    must_markers: list[str]
    must_words: list[str] = field(default_factory=list)
    category: str = "personal"  # personal | self | secrets | context | external
    allow_web: bool = False
    note: str = ""


def oac_questions() -> list[OACAsk]:
    return [
        OACAsk(
            "who am I?",
            ["ZXQ-PERSONA-CURIOUS", "ZXQ-EDU-PURDUE27"],
            ["curious", "Purdue"],
            category="self",
        ),
        OACAsk(
            "what am I like?",
            ["ZXQ-PERSONA-CURIOUS"],
            ["curious"],
            category="self",
        ),
        OACAsk(
            "what do you know about me?",
            ["ZXQ-PERSONA-CURIOUS", "ZXQ-EDU-PURDUE27"],
            ["curious", "Purdue"],
            category="self",
        ),
        OACAsk(
            "what's my personality / curiosity like?",
            ["ZXQ-PERSONA-CURIOUS"],
            ["curious"],
            category="self",
        ),
        OACAsk(
            "where do I go to school?",
            ["ZXQ-EDU-PURDUE27"],
            ["Purdue"],
            category="self",
        ),
        OACAsk(
            "did I ever rebuild a car?",
            ["ZXQ-BMW540I-REBUILD"],
            ["BMW", "540"],
            category="personal",
        ),
        OACAsk(
            "who have I been hanging out with lately?",
            ["ZXQ-ALEX-CLIMB", "ZXQ-MAYA-ROS2"],
            ["Alex", "Maya"],
            category="personal",
        ),
        OACAsk(
            "who's my closest friend?",
            ["ZXQ-POWELL-BEST"],
            ["Powell"],
            category="personal",
        ),
        OACAsk(
            "what's running low at home?",
            ["ZXQ-OATMILK-SOAP"],
            ["oat", "soap"],
            category="personal",
        ),
        OACAsk(
            "anything broken or annoying around the apartment?",
            ["ZXQ-FAUCET-DETENT"],
            ["faucet"],
            category="personal",
        ),
        OACAsk(
            "how do I focus or take breaks?",
            ["ZXQ-FOCUS25-STRETCH"],
            ["25", "focus"],
            category="personal",
        ),
        OACAsk(
            "do you remember anything about sleep or caffeine for me?",
            ["ZXQ-MATCHA-NO-ESPRESSO"],
            ["matcha", "espresso"],
            category="personal",
        ),
        OACAsk(
            "what note-taking tools do I refuse?",
            ["ZXQ-MARKDOWN-NOT-NOTION"],
            ["Notion", "markdown"],
            category="personal",
        ),
        OACAsk(
            "what makes me anxious?",
            ["ZXQ-OPENLOOPS-ANX"],
            ["open", "anxious"],
            category="personal",
        ),
        OACAsk(
            "what kind of person am I when it comes to quality?",
            ["ZXQ-CRAFT-OVER-SHIP"],
            ["craft"],
            category="personal",
        ),
        OACAsk(
            "any big wins I mentioned?",
            ["ZXQ-ESP32-MIC-WIN"],
            ["ESP32", "mic"],
            category="personal",
        ),
        OACAsk(
            "did I ask about nuclear spent fuel before?",
            ["ZXQ-SPENT-FUEL-Q"],
            ["fuel", "reactor"],
            category="personal",
        ),
        OACAsk(
            "what do I want long-term from a personal AI?",
            ["ZXQ-WANT-MEMORY-AI"],
            ["remember"],
            category="personal",
        ),
        OACAsk(
            "any body issues I mentioned?",
            ["ZXQ-WRIST-TRACKPAD"],
            ["wrist"],
            category="personal",
        ),
        OACAsk(
            "what about Sam and lunch?",
            ["ZXQ-SAM-LUNCH-ANX"],
            ["Sam"],
            category="personal",
        ),
        OACAsk(
            "do I run regularly?",
            ["ZXQ-RUN40-X3"],
            ["40", "run"],
            category="personal",
        ),
        OACAsk(
            "what's my private rule about PINs in notes?",
            ["ZXQ-PIN-POLICY"],
            ["PIN"],
            category="secrets",
            note="needs dest=secrets or include_secrets",
        ),
        # Statements / talk-about-me (not question-shaped) still need DB context
        OACAsk(
            "I've been anxious about open loops again today",
            ["ZXQ-OPENLOOPS-ANX"],
            ["open", "anxious"],
            category="context",
            note="statement — should still query_db for psychology context",
        ),
        OACAsk(
            "Alex came by after climbing, still thinking about that",
            ["ZXQ-ALEX-CLIMB"],
            ["Alex", "climb"],
            category="context",
        ),
        OACAsk(
            "my right wrist is acting up after trackpad work",
            ["ZXQ-WRIST-TRACKPAD"],
            ["wrist"],
            category="context",
        ),
        OACAsk(
            "thinking about that BMW rebuild and how much I learned",
            ["ZXQ-BMW540I-REBUILD"],
            ["BMW"],
            category="context",
        ),
        OACAsk(
            "we're almost out of oat milk again",
            ["ZXQ-OATMILK-SOAP"],
            ["oat"],
            category="context",
        ),
        # Multi-word q should still hit with soft OR matching
        OACAsk(
            "remind me about my personality and curiosity",
            ["ZXQ-PERSONA-CURIOUS"],
            ["curious"],
            category="self",
            note="multi-word intent; q may include both words",
        ),
        # Control: external question SHOULD use web, not DB
        OACAsk(
            "what's the current weather in West Lafayette Indiana?",
            [],
            [],
            category="external",
            allow_web=True,
            note="control — web is OK; query_db not required",
        ),
    ]


@dataclass
class ScenarioTurn:
    question: str
    must_words: list[str] = field(default_factory=list)
    forbid: list[str] = field(default_factory=list)
    allow_web: bool = False
    note: str = ""


def scenario_turns() -> list[ScenarioTurn]:
    """Multi-turn thread matching the live failure: stats → exposure → Purdue ME → gym pushback."""
    dump_forbid = [
        "you're a mechanical engineering student",
        "your interests span",
        "auto → conversation",
        "mode → companion",
        "bmw 540",
        "(auto →",
    ]
    return [
        ScenarioTurn(
            "bring me up some info and more importantly general statistics about social anxiety. Any and all information is welcome, thanks",
            must_words=["anxiety"],
            forbid=dump_forbid,
            allow_web=True,
            note="public stats — web OK, no bio dump",
        ),
        ScenarioTurn(
            "how have other people solved their social anxiety without drugs?",
            must_words=["anxiety"],
            forbid=dump_forbid,
            allow_web=True,
            note="public how-to — stay on social anxiety thread",
        ),
        ScenarioTurn(
            "On campus, what could I do to start exposing myself more to social things? Like go up to random people? What would I even say",
            must_words=["campus"],
            forbid=dump_forbid,
            allow_web=False,
            note="stay on exposure, no biography dump",
        ),
        ScenarioTurn(
            "Ive been really bad at making friends, I go to purdue, im a mechanical engineer and in all my classes, idk, I can meet people in labs n shit but then were more so colleagues. im really good at talking to people once we're forced to talk together like for group projects but otherwise shit I suck. The only class ive made really good friends in is this business class OBHR330 that im in now. were not all friends outside of class, but the 2.5 hours a week are really fun",
            must_words=["purdue", "obhr", "lab", "class"],
            forbid=dump_forbid,
            note="must use Purdue / OBHR330 / labs, not recap the whole hayden.json",
        ),
        ScenarioTurn(
            "Yeah but unfortunately, the kids in engineering classes are weird and i really dont like them. I do go to the gym but in there im lowkey just another worker ant. And Im not gonna pay for friends with the purdue bodybuilding club",
            must_words=["gym", "engineering"],
            forbid=dump_forbid + [
                "if you ever need advice",
                "keep being yourself",
            ],
            note="engage the pushback: engineering kids, gym-as-ant, won't pay for bodybuilding club",
        ),
    ]


def score_scenario_turn(
    turn: ScenarioTurn, reply: str, calls: list[dict[str, Any]]
) -> dict[str, Any]:
    reply_l = _fold(reply or "")
    tools = [str(c.get("tool") or "") for c in calls]
    words_hit = [w for w in turn.must_words if w.casefold() in reply_l]
    forbid_hit = [f for f in turn.forbid if f.casefold() in reply_l]
    issues: list[str] = []
    if turn.must_words:
        need = max(1, (len(turn.must_words) + 1) // 2)
        if len(words_hit) < need:
            issues.append(f"missing_words: want>={need} got={words_hit}")
    if forbid_hit:
        issues.append(f"forbid:{forbid_hit}")
    if looks_like_profile_dump(reply or ""):
        issues.append("profile_dump")
    if looks_like_empty_therapy(reply or ""):
        issues.append("empty_therapy")
    if "(auto →" in (reply or ""):
        issues.append("mode_route_note")
    ok = not issues
    return {
        "question": turn.question[:120],
        "ok": ok,
        "issues": issues,
        "tools": tools,
        "words_hit": words_hit,
        "reply_preview": (reply or "")[:500],
        "note": turn.note,
    }


def run_unit_tests(sandbox: Path | None = None) -> dict[str, Any]:
    """No Ollama — routing, prefetch, dump stripping, personal vs web."""
    fails: list[str] = []

    def check(name: str, cond: bool) -> None:
        if not cond:
            fails.append(name)

    check("alias_conversation", get_mode("conversation").id == "companion")
    check("alias_planner", get_mode("planner").id == "companion")
    check("alias_oac", get_mode("oac").id == "companion")
    check("same_prompt", get_mode("conversation").prompt == get_mode("companion").prompt)

    decision = suggest_mode(
        "I've been really bad at making friends, I go to purdue", "companion"
    )
    check("no_flavor_switch", decision.mode_id == "companion")

    decision = suggest_mode("do a deep research brief on lithium batteries", "companion")
    check("deep_research_still_routes", decision.mode_id == "deep_research")

    check(
        "stats_not_personal",
        not personal_memory_question(
            "bring me up some info and more importantly general statistics about social anxiety"
        ),
    )
    check(
        "stats_wants_web",
        wants_open_web(
            "bring me up some info and more importantly general statistics about social anxiety"
        ),
    )
    check(
        "friends_are_personal",
        personal_memory_question(
            "I've been really bad at making friends, I go to purdue, im a mechanical engineer"
        ),
    )
    check("rebuild_personal", personal_memory_question("did I ever rebuild a car?"))
    check("focus_personal", personal_memory_question("how do I focus or take breaks?"))
    check("focus_not_web", not wants_open_web("how do I focus or take breaks?"))
    check("home_personal", personal_memory_question("what's running low at home?"))
    check("sam_personal", personal_memory_question("what about Sam and lunch?"))
    school_tokens = {t.casefold() for t in extract_query_tokens("where do I go to school?")}
    check("school_expands", "purdue" in school_tokens or "education" in school_tokens)

    dump = (
        "The gym is anonymous and paying for the bodybuilding club is a non-starter."
        "You're a mechanical engineering student at Purdue University, and you've noticed "
        "that while you're good at engaging with people in structured settings like group "
        "projects, you struggle to form deeper connections. Your interests span mechanical "
        "engineering, robotics, aerospace, CAD, FEA, and microcontrollers."
    )
    check("detect_dump", looks_like_profile_dump(dump))
    stripped = strip_profile_dump(dump)
    check("strip_dump", "you're a mechanical engineering student" not in stripped.casefold())
    check("strip_keeps_answer", "bodybuilding club" in stripped.casefold())

    therapy = (
        "I understand how you feel. It sounds like you're dealing with some social "
        "challenges. It's totally normal to feel that way. If you ever need advice or "
        "someone to talk to about these feelings, I'm here."
    )
    check("detect_therapy", looks_like_empty_therapy(therapy))

    tokens = extract_query_tokens(
        "Yeah but the kids in engineering are weird. I go to the gym. Purdue bodybuilding club"
    )
    tok_l = {t.casefold() for t in tokens}
    check("tokens_gym", "gym" in tok_l)
    check("tokens_purdue", "purdue" in tok_l)

    mixed = (
        "education: Hayden is a mechanical engineering student at Purdue (marker: ZXQ-EDU-PURDUE27).\n"
        "experience: Hayden rebuilt a BMW 540i (marker: ZXQ-BMW540I-REBUILD).\n"
        "gym: Hayden lifts at the campus gym but feels anonymous (marker: ZXQ-GYM-ANT).\n"
        "friends: Hayden is good at talking once a group project forces it (marker: ZXQ-FORCED-TALK).\n"
    )
    kept = compact_digest(mixed, ["gym", "friends", "Purdue", "engineering"])
    check("compact_keeps_gym", "ZXQ-GYM-ANT" in kept)
    check("compact_drops_bmw", "ZXQ-BMW540I-REBUILD" not in kept)

    mem = host_fallback_memory(
        "Yeah but the engineering kids are weird",
        "Don't join the paid club then — use the structured hour you already like.",
        "",
        recent_turns=[
            ("how have people solved social anxiety without drugs?", "CBT and exposure."),
            (
                "I go to Purdue, ME, labs are colleagues, OBHR330 is the fun class",
                "OBHR330 is already a forced-social slot that works.",
            ),
        ],
    )
    check("fallback_thread", "OBHR330" in mem or "Purdue" in mem)

    glued = (
        'The question is a bit vague, but I can help you explore equations in different '
        'contexts. Could you please clarify what you mean by "its main equations"? '
        "Are you referring to equations in mathematics, physics, engineering, or another "
        "field? Or are you asking about equations in a specific context or subject?"
        "The main equations of general relativity are derived from Einstein's field equations."
    )
    check("clarifier_detect", reply_looks_like_clarifier(glued))
    stripped = strip_leading_clarifier(glued)
    check("clarifier_strip_glued", stripped.startswith("The main equations of general relativity"))
    check("clarifier_strip_drops_ask", "a bit vague" not in stripped.casefold())

    check("wants_videos_pull_up", wants_videos("Pull up some videos of it pls"))
    check("wants_videos_please", wants_videos("Pull up some videos please"))
    check("do_it_confirm", is_action_confirm("Do it"))
    check("ok_is_ack", is_short_ack("Ok"))
    check("ok_not_confirm", not is_action_confirm("Ok"))
    topic = topic_for_search(
        "Standing request: What is the calabi yau manifold?\nContext: x\nLast answer: y",
        [("What is the calabi yau manifold?", "A Ricci-flat Kähler manifold.")],
    )
    check("search_topic_strips_what_is", "calabi yau" in topic.casefold())
    check("search_topic_not_action", "video" not in topic.casefold())
    named = named_search_topic(
        "Bring up some videos explaining general relativity. An MIT lecture is prefferred pls"
    )
    check("named_video_topic_gr", "general relativity" in named.casefold())
    check("named_video_topic_mit", "mit" in named.casefold())
    check("named_video_not_endo", "endometriosis" not in named.casefold())
    check(
        "vague_videos_no_named",
        named_search_topic("Pull up some videos of it pls") == "",
    )

    return {"ok": not fails, "fails": fails}


def _prefetch_on_sandbox(sandbox: Path) -> dict[str, Any]:
    from ainet.tools.ops import DatabaseTools

    db = DatabaseTools(sandbox)
    payload = prefetch_personal_context(
        db,
        "Yeah but the kids in engineering classes are weird. I go to the gym. "
        "I'm not gonna pay for the Purdue bodybuilding club",
        standing="making friends on campus as a Purdue ME",
        recent_turns=[
            (
                "I go to Purdue, mechanical engineer, OBHR330 is the only fun class",
                "OBHR330 already gives you a structured social slot.",
            )
        ],
    )
    digest = str(payload.get("digest") or "")
    issues: list[str] = []
    if "ZXQ-GYM-ANT" not in digest and "gym" not in digest.casefold():
        issues.append("missing_gym_context")
    if "ZXQ-BMW540I-REBUILD" in digest:
        issues.append("injected_unrelated_bmw")
    return {
        "ok": not issues,
        "issues": issues,
        "digest": digest[:800],
        "tokens": payload.get("tokens"),
    }


class TurnTrace:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def hook(self, session: ChatSession) -> None:
        original = session._run_tool_call

        def wrapped(call: dict[str, Any]) -> dict[str, Any]:
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
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
            self.calls.append({"tool": name, "args": args, "result": result})
            return result

        session._run_tool_call = wrapped  # type: ignore[method-assign]

    def reset(self) -> None:
        self.calls = []


def _blob(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except TypeError:
        return str(obj)


def score_turn(ask: OACAsk, reply: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    tools = [str(c.get("tool") or "") for c in calls]
    query_calls = [c for c in calls if c.get("tool") == "query_db"]
    web_calls = [c for c in calls if c.get("tool") in {"web_search", "web_fetch"}]
    tool_blob = " ".join(_blob(c.get("result")) for c in query_calls).casefold()
    reply_l = (reply or "").casefold()
    combined = f"{reply_l}\n{tool_blob}"

    markers_hit = [m for m in ask.must_markers if m.casefold() in combined]
    words_hit = [w for w in ask.must_words if w.casefold() in combined]

    issues: list[str] = []
    used_query = bool(query_calls)
    used_web = bool(web_calls)

    if ask.category == "external":
        ok = True
        if not ask.allow_web and used_web:
            issues.append("unexpected_web")
            ok = False
        # web optional but nice
    else:
        if not used_query:
            issues.append("missing_query_db")
        if used_web and not ask.allow_web:
            issues.append(f"used_web:{[c.get('tool') for c in web_calls]}")
        need_m = max(1, (len(ask.must_markers) + 1) // 2) if ask.must_markers else 0
        need_w = max(1, (len(ask.must_words) + 1) // 2) if ask.must_words else 0
        if ask.must_markers and len(markers_hit) < need_m:
            issues.append(f"missing_markers: want>={need_m} got={markers_hit}")
        if ask.must_words and len(words_hit) < need_w and len(markers_hit) < need_m:
            issues.append(f"missing_words: got={words_hit}")
        ok = not issues

    return {
        "question": ask.question,
        "category": ask.category,
        "ok": ok,
        "issues": issues,
        "tools": tools,
        "query_db_calls": len(query_calls),
        "query_db_args": [c.get("args") for c in query_calls],
        "web_calls": len(web_calls),
        "markers_hit": markers_hit,
        "words_hit": words_hit,
        "reply_preview": (reply or "")[:400],
        "note": ask.note,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stress test OAC personal DB retrieval")
    p.add_argument("--source-db", type=Path, default=None)
    p.add_argument("--sandbox", type=Path, default=None)
    p.add_argument("--apply", action="store_true", help="Required to run live Ollama asks")
    p.add_argument("--keep", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--max-tool-rounds", type=int, default=16)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0, help="Only first N questions (0=all)")
    p.add_argument("--unit-only", action="store_true", help="Routing/prefetch tests, no Ollama")
    p.add_argument("--skip-scenario", action="store_true", help="Skip multi-turn thread test")
    p.add_argument("--skip-asks", action="store_true", help="Skip per-question retrieval asks")
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
    args = build_parser().parse_args(argv)
    source = (args.source_db or default_db_root()).resolve()
    if not source.is_dir():
        print(f"Source db not found: {source}", file=sys.stderr)
        return 2

    if args.sandbox:
        sandbox = args.sandbox.resolve()
    else:
        sandbox = Path(tempfile.mkdtemp(prefix="ainet-oac-stress-")).resolve()

    print(f"Sandbox: {sandbox}")
    copy_sandbox(source, sandbox)
    _empty_knowledge(sandbox)
    ensure_knowledge_files(sandbox)
    seed_knowledge(sandbox)
    print("Seeded unique personal markers into knowledge files")

    print("\n=== UNIT ===")
    unit = run_unit_tests()
    print(json.dumps({"ok": unit["ok"], "fails": unit["fails"]}, indent=2))
    print("\n=== PREFETCH ===")
    prefetch = _prefetch_on_sandbox(sandbox)
    print(json.dumps({k: prefetch[k] for k in ("ok", "issues", "tokens")}, indent=2))
    print((prefetch.get("digest") or "")[:400])

    if not unit["ok"] or not prefetch["ok"]:
        print("Unit/prefetch failed", file=sys.stderr)
        if args.unit_only or not args.apply:
            return 1

    if args.unit_only:
        return 0 if unit["ok"] and prefetch["ok"] else 1

    if not args.apply:
        print("Pass --apply to run live Ollama asks (unit+prefetch already ran)", file=sys.stderr)
        return 0 if unit["ok"] and prefetch["ok"] else 1

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
        return 3

    results: list[dict[str, Any]] = []
    if not args.skip_asks:
        asks = oac_questions()
        if args.limit and args.limit > 0:
            asks = asks[: args.limit]
        print(f"\n=== OAC asks ({len(asks)}) ===")
        for i, ask in enumerate(asks, 1):
            print(f"\n[{i}/{len(asks)}] ({ask.category}) {ask.question}")
            session = ChatSession(get_mode("companion"), config=config)
            trace = TurnTrace()
            trace.hook(session)
            try:
                reply = session.ask(ask.question, stream=False) or ""
            except Exception as exc:
                reply = f"(error: {exc})"
                print(traceback.format_exc()[:800])
            scored = score_turn(ask, reply, trace.calls)
            results.append(scored)
            mark = "OK" if scored["ok"] else "FAIL"
            print(f"  {mark} tools={scored['tools']}")
            print(f"  markers={scored['markers_hit']} words={scored['words_hit']}")
            if scored["issues"]:
                print(f"  issues={scored['issues']}")
            print(f"  reply: {scored['reply_preview'][:220]}")

    personal = [r for r in results if r["category"] != "external"]
    passed = sum(1 for r in personal if r["ok"])
    total = len(personal)
    used_query = sum(1 for r in personal if r["query_db_calls"] > 0)
    used_web_bad = sum(1 for r in personal if r["web_calls"] > 0)
    fails = [r for r in personal if not r["ok"]]
    retrieval_ok = (not personal) or (
        passed == total and used_query == total and used_web_bad == 0
    )

    scenario_results: list[dict[str, Any]] = []
    if not args.skip_scenario:
        print("\n=== SCENARIO (one session, social-anxiety thread) ===")
        session = ChatSession(get_mode("companion"), config=config)
        trace = TurnTrace()
        trace.hook(session)
        for i, turn in enumerate(scenario_turns(), 1):
            print(f"\n[S{i}/{len(scenario_turns())}] {turn.question[:100]}")
            trace.reset()
            try:
                reply = session.ask(turn.question, stream=False) or ""
            except Exception as exc:
                reply = f"(error: {exc})"
                print(traceback.format_exc()[:800])
            scored = score_scenario_turn(turn, reply, trace.calls)
            scenario_results.append(scored)
            mark = "OK" if scored["ok"] else "FAIL"
            print(f"  {mark} tools={scored['tools']} words={scored['words_hit']}")
            if scored["issues"]:
                print(f"  issues={scored['issues']}")
            print(f"  reply: {scored['reply_preview'][:280]}")
            injected = (getattr(session, "_injected_context", "") or "")[:180]
            if injected:
                print(f"  injected: {injected}")

    scenario_fails = [r for r in scenario_results if not r["ok"]]
    scenario_ok = (not scenario_results) or not scenario_fails

    summary = {
        "unit_ok": unit["ok"],
        "prefetch_ok": prefetch["ok"],
        "unit_fails": unit["fails"],
        "prefetch_issues": prefetch.get("issues"),
        "accuracy": round(passed / total, 3) if total else 1.0,
        "passed": passed,
        "total": total,
        "query_db_rate": round(used_query / total, 3) if total else 1.0,
        "bad_web_rate": round(used_web_bad / total, 3) if total else 0.0,
        "fail_count": len(fails),
        "scenario_passed": sum(1 for r in scenario_results if r["ok"]),
        "scenario_total": len(scenario_results),
        "ok": unit["ok"] and prefetch["ok"] and retrieval_ok and scenario_ok,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n=== FAILURES ===")
    print(json.dumps({"asks": fails, "scenario": scenario_fails}, indent=2, ensure_ascii=True)[:16000])

    report = {
        "timestamp": _utc_now(),
        "sandbox": str(sandbox),
        "model": config.model,
        "summary": summary,
        "results": results,
        "scenario": scenario_results,
        "prefetch": prefetch,
        "unit": unit,
    }
    out = args.report or (sandbox / "oac-stress-report.json")
    _write_json(out, report)
    print(f"\nReport: {out}")
    print(f"Sandbox: {sandbox}")
    return 0 if summary["ok"] else 1

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
        return 3

    results: list[dict[str, Any]] = []
    print("\n=== OAC asks ===")
    for i, ask in enumerate(asks, 1):
        print(f"\n[{i}/{len(asks)}] ({ask.category}) {ask.question}")
        # Fresh session per question so standing memory does not leak web habits.
        session = ChatSession(get_mode("companion"), config=config)
        trace = TurnTrace()
        trace.hook(session)
        try:
            reply = session.ask(ask.question, stream=False) or ""
        except Exception as exc:
            reply = f"(error: {exc})"
            print(traceback.format_exc()[:800])
        scored = score_turn(ask, reply, trace.calls)
        results.append(scored)
        mark = "OK" if scored["ok"] else "FAIL"
        print(f"  {mark} tools={scored['tools']}")
        print(f"  markers={scored['markers_hit']} words={scored['words_hit']}")
        if scored["issues"]:
            print(f"  issues={scored['issues']}")
        print(f"  reply: {scored['reply_preview'][:220]}")

    personal = [r for r in results if r["category"] != "external"]
    passed = sum(1 for r in personal if r["ok"])
    total = len(personal)
    used_query = sum(1 for r in personal if r["query_db_calls"] > 0)
    used_web_bad = sum(1 for r in personal if r["web_calls"] > 0)
    fails = [r for r in personal if not r["ok"]]

    summary = {
        "accuracy": round(passed / total, 3) if total else 0.0,
        "passed": passed,
        "total": total,
        "query_db_rate": round(used_query / total, 3) if total else 0.0,
        "bad_web_rate": round(used_web_bad / total, 3) if total else 0.0,
        "fail_count": len(fails),
        "ok": passed == total and used_query == total and used_web_bad == 0,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\n=== FAILURES ===")
    print(json.dumps(fails, indent=2, ensure_ascii=True)[:12000])

    report = {
        "timestamp": _utc_now(),
        "sandbox": str(sandbox),
        "model": config.model,
        "summary": summary,
        "results": results,
    }
    out = args.report or (sandbox / "oac-stress-report.json")
    _write_json(out, report)
    print(f"\nReport: {out}")
    print(f"Sandbox: {sandbox}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
