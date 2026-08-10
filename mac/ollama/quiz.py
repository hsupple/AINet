"""Quiz flow: candidate ranking, active state, scoring (host-side for OAC)."""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ainet.tools.fsutil import atomic_write_text
from ainet.tools.ops import DatabaseTools
from ollama.research_sessions import (
    SCORES_PATH,
    ensure_research_scaffold,
    get_session,
    list_sessions,
    session_path,
)


# Suggestion throttle (prompt + tool heuristic)
MIN_TURNS_BETWEEN_SUGGEST = 8
MIN_SECONDS_BETWEEN_SUGGEST = 15 * 60
SUGGEST_CHANCE = 0.22

# Ranking weights
RECENCY_WEIGHT = 0.45
WEAKNESS_WEIGHT = 0.40
WRONG_BOOST = 0.15


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _runtime_root(db: DatabaseTools) -> Path:
    root = db.paths.root / "runtime" / "oac"
    root.mkdir(parents=True, exist_ok=True)
    return root


def active_quiz_path(db: DatabaseTools) -> Path:
    return _runtime_root(db) / "quiz_active.json"


def quiz_meta_path(db: DatabaseTools) -> Path:
    return _runtime_root(db) / "quiz_meta.json"


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_json_file(path: Path, data: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def load_scores(db: DatabaseTools) -> dict[str, Any]:
    ensure_research_scaffold(db)
    data = db.read_json(SCORES_PATH)["data"]
    if not isinstance(data, dict):
        data = {"sessions": {}, "topics": {}, "last_updated": ""}
    data.setdefault("sessions", {})
    data.setdefault("topics", {})
    return data


def save_scores(db: DatabaseTools, scores: dict[str, Any], *, summary: str) -> None:
    scores["last_updated"] = _utc_now()
    db.write_json(SCORES_PATH, scores, create=True, summary=summary)


def _score_entry(scores: dict[str, Any], *, session_id: str = "", topic_slug: str = "") -> dict[str, Any]:
    if session_id:
        bucket = scores.setdefault("sessions", {})
        entry = bucket.get(session_id)
        if not isinstance(entry, dict):
            entry = {
                "memory_score": 0.5,
                "attempts": 0,
                "correct": 0,
                "wrong": 0,
                "last_quiz_at": "",
                "history": [],
            }
            bucket[session_id] = entry
        return entry
    if topic_slug:
        bucket = scores.setdefault("topics", {})
        entry = bucket.get(topic_slug)
        if not isinstance(entry, dict):
            entry = {
                "memory_score": 0.5,
                "attempts": 0,
                "correct": 0,
                "wrong": 0,
                "last_quiz_at": "",
                "history": [],
            }
            bucket[topic_slug] = entry
        return entry
    return {
        "memory_score": 0.5,
        "attempts": 0,
        "correct": 0,
        "wrong": 0,
        "last_quiz_at": "",
        "history": [],
    }


def _recency_score(iso_ts: str | None, *, now: datetime | None = None) -> float:
    """1.0 = very recent, ~0 = old/unknown. Half-life ~7 days."""
    dt = _parse_iso(iso_ts)
    if not dt:
        return 0.15
    now = now or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age_hours = max(0.0, (now - dt).total_seconds() / 3600.0)
    # exponential decay; ~0.5 at 7d, ~0.1 at ~23d
    return max(0.05, min(1.0, 0.5 ** (age_hours / (7 * 24))))


def rank_quiz_candidates(
    db: DatabaseTools,
    *,
    limit: int = 12,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Rank sessions/topics by recency + low memory score + prior wrongs."""
    ensure_research_scaffold(db)
    scores = load_scores(db)
    now = now or datetime.now(timezone.utc)
    ranked: list[dict[str, Any]] = []

    for session in list_sessions(db):
        if not isinstance(session, dict):
            continue
        sid = str(session.get("id") or "")
        if not sid:
            continue
        details = session.get("details_covered") or []
        if not isinstance(details, list) or not details:
            continue
        sess_score = _score_entry(scores, session_id=sid)
        topic_slug = str(session.get("topic_slug") or session.get("related_topic") or "")
        memory = float(sess_score.get("memory_score", session.get("memory_score", 0.5)))
        wrong = int(sess_score.get("wrong") or 0)
        if topic_slug:
            topic_bucket = scores.get("topics") or {}
            topic_score = topic_bucket.get(topic_slug)
            # Only blend topic scores that have real quiz history (avoid default 0.5 floor).
            if isinstance(topic_score, dict) and int(topic_score.get("attempts") or 0) > 0:
                memory = min(memory, float(topic_score.get("memory_score", memory)))
                wrong += int(topic_score.get("wrong") or 0)
        ts = str(session.get("ended_at") or session.get("started_at") or "")
        recency = _recency_score(ts, now=now)
        weakness = 1.0 - max(0.0, min(1.0, memory))
        wrong_factor = min(1.0, wrong / 5.0)
        priority = (
            RECENCY_WEIGHT * recency
            + WEAKNESS_WEIGHT * weakness
            + WRONG_BOOST * wrong_factor
        )
        ranked.append(
            {
                "session_id": sid,
                "subject": session.get("subject") or session.get("title"),
                "topic_slug": topic_slug,
                "path": session_path(sid),
                "started_at": session.get("started_at"),
                "ended_at": session.get("ended_at"),
                "status": session.get("status"),
                "memory_score": memory,
                "wrong_count": wrong,
                "recency": round(recency, 4),
                "priority": round(priority, 4),
                "detail_count": len(details),
                "sample_details": details[:5],
            }
        )

    ranked.sort(key=lambda x: float(x["priority"]), reverse=True)
    return ranked[: max(1, limit)]


def list_quiz_candidates(db: DatabaseTools, *, limit: int = 12) -> dict[str, Any]:
    candidates = rank_quiz_candidates(db, limit=limit)
    return {"ok": True, "count": len(candidates), "candidates": candidates}


def should_suggest_quiz(
    db: DatabaseTools,
    *,
    turn_count: int | None = None,
    force_roll: bool | None = None,
) -> dict[str, Any]:
    """Lightweight anti-spam heuristic for OAC quiz suggestions."""
    active = _load_json_file(active_quiz_path(db))
    if active.get("status") == "active":
        return {
            "ok": True,
            "suggest": False,
            "reason": "quiz already active — continue asking/grading instead",
        }

    meta = _load_json_file(quiz_meta_path(db))
    now = datetime.now(timezone.utc)
    last_suggest = _parse_iso(str(meta.get("last_suggest_at") or ""))
    turns_since = int(meta.get("turns_since_suggest", 999))
    if turn_count is not None:
        last_turn = int(meta.get("last_turn_count", 0))
        if turn_count >= last_turn:
            turns_since = turn_count - int(meta.get("suggest_turn_count", -999))
        meta["last_turn_count"] = turn_count

    if turns_since < MIN_TURNS_BETWEEN_SUGGEST:
        _save_json_file(quiz_meta_path(db), meta)
        return {
            "ok": True,
            "suggest": False,
            "reason": f"only {turns_since} turns since last suggest (min {MIN_TURNS_BETWEEN_SUGGEST})",
            "turns_since_suggest": turns_since,
        }

    if last_suggest is not None:
        if last_suggest.tzinfo is None:
            last_suggest = last_suggest.replace(tzinfo=timezone.utc)
        elapsed = (now - last_suggest).total_seconds()
        if elapsed < MIN_SECONDS_BETWEEN_SUGGEST:
            _save_json_file(quiz_meta_path(db), meta)
            return {
                "ok": True,
                "suggest": False,
                "reason": f"suggested {int(elapsed)}s ago (min {MIN_SECONDS_BETWEEN_SUGGEST}s)",
                "seconds_since_suggest": int(elapsed),
            }

    candidates = rank_quiz_candidates(db, limit=3)
    if not candidates:
        return {
            "ok": True,
            "suggest": False,
            "reason": "no research sessions with details_covered yet",
        }

    roll = SUGGEST_CHANCE if force_roll is None else (1.0 if force_roll else 0.0)
    if force_roll is None:
        roll_ok = random.random() < SUGGEST_CHANCE
    else:
        roll_ok = bool(force_roll)

    if not roll_ok:
        # Still bump turn counter so we don't re-check every message aggressively
        meta["turns_since_suggest"] = turns_since + 1
        _save_json_file(quiz_meta_path(db), meta)
        return {
            "ok": True,
            "suggest": False,
            "reason": "throttle roll declined (occasional suggestions only)",
            "chance": SUGGEST_CHANCE,
        }

    meta["last_suggest_at"] = _utc_now()
    meta["suggest_turn_count"] = turn_count if turn_count is not None else meta.get("last_turn_count", 0)
    meta["turns_since_suggest"] = 0
    _save_json_file(quiz_meta_path(db), meta)
    return {
        "ok": True,
        "suggest": True,
        "reason": "idle gap / enough turns + research material available",
        "top_candidates": candidates[:3],
        "chance": roll,
    }


def _seed_questions_from_candidates(
    db: DatabaseTools,
    *,
    count: int,
    session_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if session_ids:
        candidates = []
        for sid in session_ids:
            session = get_session(db, sid)
            if not session:
                continue
            candidates.append(
                {
                    "session_id": sid,
                    "subject": session.get("subject") or session.get("title"),
                    "topic_slug": session.get("topic_slug") or "",
                    "sample_details": session.get("details_covered") or [],
                }
            )
    else:
        candidates = rank_quiz_candidates(db, limit=max(count * 2, 6))

    questions: list[dict[str, Any]] = []
    for cand in candidates:
        details = cand.get("sample_details") or []
        if not isinstance(details, list):
            continue
        # Prefer full session details when only sample present
        sid = str(cand.get("session_id") or "")
        session = get_session(db, sid) if sid else None
        if session and isinstance(session.get("details_covered"), list):
            details = session["details_covered"]
        for detail in details:
            if len(questions) >= count:
                break
            if isinstance(detail, dict):
                prompt = str(
                    detail.get("question")
                    or detail.get("text")
                    or detail.get("point")
                    or ""
                ).strip()
                expected = str(
                    detail.get("answer") or detail.get("text") or ""
                ).strip()
                kind = str(detail.get("kind") or "point")
            else:
                prompt = str(detail).strip()
                expected = prompt
                kind = "point"
            if not prompt:
                continue
            has_explicit_q = isinstance(detail, dict) and bool(detail.get("question"))
            # For mechanism/point facts, ask "What do you remember about …?"
            if kind != "qa" and not has_explicit_q:
                ask = f"What do you remember about: {prompt}?"
                expected_ans = expected or prompt
            else:
                ask = prompt if has_explicit_q else prompt
                expected_ans = expected
            questions.append(
                {
                    "id": f"q-{uuid.uuid4().hex[:8]}",
                    "session_id": sid,
                    "topic_slug": cand.get("topic_slug") or "",
                    "subject": cand.get("subject") or "",
                    "prompt": ask,
                    "expected_answer": expected_ans,
                    "source_detail": detail if isinstance(detail, dict) else {"text": str(detail)},
                    "status": "pending",
                    "user_answer": "",
                    "correct": None,
                    "asked_at": "",
                    "answered_at": "",
                }
            )
        if len(questions) >= count:
            break
    return questions


def start_quiz(
    db: DatabaseTools,
    *,
    questions: list[dict[str, Any]] | None = None,
    count: int = 5,
    session_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Start an active quiz; OAC may pass drafted questions or let host seed them."""
    ensure_research_scaffold(db)
    active = _load_json_file(active_quiz_path(db))
    if active.get("status") == "active":
        return {
            "ok": False,
            "error": "A quiz is already active. Finish it or call get_quiz_status.",
            "quiz": _public_quiz(active),
        }

    if questions:
        built: list[dict[str, Any]] = []
        for raw in questions[: max(1, count)]:
            if not isinstance(raw, dict):
                continue
            prompt = str(raw.get("prompt") or raw.get("question") or "").strip()
            if not prompt:
                continue
            built.append(
                {
                    "id": str(raw.get("id") or f"q-{uuid.uuid4().hex[:8]}"),
                    "session_id": str(raw.get("session_id") or ""),
                    "topic_slug": str(raw.get("topic_slug") or ""),
                    "subject": str(raw.get("subject") or ""),
                    "prompt": prompt,
                    "expected_answer": str(raw.get("expected_answer") or raw.get("answer") or ""),
                    "source_detail": raw.get("source_detail"),
                    "status": "pending",
                    "user_answer": "",
                    "correct": None,
                    "asked_at": "",
                    "answered_at": "",
                }
            )
        if not built:
            return {"ok": False, "error": "No usable questions provided"}
        qlist = built
    else:
        qlist = _seed_questions_from_candidates(db, count=max(1, min(count, 12)), session_ids=session_ids)
        if not qlist:
            return {
                "ok": False,
                "error": "No quiz material — research sessions need details_covered first",
            }

    # Mark first question asked
    qlist[0]["status"] = "asked"
    qlist[0]["asked_at"] = _utc_now()

    quiz = {
        "id": f"quiz-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}",
        "status": "active",
        "started_at": _utc_now(),
        "ended_at": "",
        "questions": qlist,
        "current_index": 0,
        "answered": 0,
        "correct_count": 0,
        "wrong_count": 0,
    }
    _save_json_file(active_quiz_path(db), quiz)
    return {
        "ok": True,
        "quiz_id": quiz["id"],
        "total": len(qlist),
        "current_question": _public_question(qlist[0], reveal_answer=False),
        "instruction": (
            "Ask the current question conversationally. When Hayden answers, grade it, "
            "then call record_quiz_answer with correct=true/false."
        ),
    }


def get_quiz_status(db: DatabaseTools, *, reveal_answer: bool = False) -> dict[str, Any]:
    quiz = _load_json_file(active_quiz_path(db))
    if not quiz:
        return {"ok": True, "status": "idle", "quiz": None}
    return {"ok": True, "status": quiz.get("status", "idle"), "quiz": _public_quiz(quiz, reveal_answer=reveal_answer)}


def record_quiz_answer(
    db: DatabaseTools,
    *,
    user_answer: str,
    correct: bool,
    brief_correction: str = "",
    question_id: str | None = None,
) -> dict[str, Any]:
    """Record graded answer, update Scores.json, advance to next question."""
    path = active_quiz_path(db)
    quiz = _load_json_file(path)
    if quiz.get("status") != "active":
        return {"ok": False, "error": "No active quiz"}

    questions = quiz.get("questions") or []
    if not isinstance(questions, list) or not questions:
        return {"ok": False, "error": "Active quiz has no questions"}

    idx = int(quiz.get("current_index") or 0)
    if question_id:
        for i, q in enumerate(questions):
            if isinstance(q, dict) and q.get("id") == question_id:
                idx = i
                break

    if idx < 0 or idx >= len(questions):
        return {"ok": False, "error": "Invalid question index"}

    question = questions[idx]
    if not isinstance(question, dict):
        return {"ok": False, "error": "Corrupt question entry"}

    now = _utc_now()
    question["user_answer"] = user_answer
    question["correct"] = bool(correct)
    question["status"] = "answered"
    question["answered_at"] = now
    if brief_correction:
        question["brief_correction"] = brief_correction

    quiz["answered"] = int(quiz.get("answered") or 0) + 1
    if correct:
        quiz["correct_count"] = int(quiz.get("correct_count") or 0) + 1
    else:
        quiz["wrong_count"] = int(quiz.get("wrong_count") or 0) + 1

    _apply_score(db, question, correct=bool(correct), ts=now)

    next_q = None
    next_idx = idx + 1
    if next_idx < len(questions):
        quiz["current_index"] = next_idx
        nxt = questions[next_idx]
        if isinstance(nxt, dict):
            nxt["status"] = "asked"
            nxt["asked_at"] = now
            next_q = _public_question(nxt, reveal_answer=False)
        done = False
    else:
        quiz["status"] = "completed"
        quiz["ended_at"] = now
        quiz["current_index"] = idx
        done = True

    _save_json_file(path, quiz)
    return {
        "ok": True,
        "correct": bool(correct),
        "done": done,
        "answered": quiz.get("answered"),
        "correct_count": quiz.get("correct_count"),
        "wrong_count": quiz.get("wrong_count"),
        "total": len(questions),
        "next_question": next_q,
        "expected_answer": question.get("expected_answer") if not correct else None,
        "instruction": (
            "Quiz complete — summarize briefly."
            if done
            else "Ask the next_question. Grade the next answer the same way."
        ),
    }


def _apply_score(db: DatabaseTools, question: dict[str, Any], *, correct: bool, ts: str) -> None:
    scores = load_scores(db)
    sid = str(question.get("session_id") or "")
    topic = str(question.get("topic_slug") or "")
    targets: list[dict[str, Any]] = []
    if sid:
        targets.append(_score_entry(scores, session_id=sid))
    if topic:
        targets.append(_score_entry(scores, topic_slug=topic))
    if not targets:
        return

    for entry in targets:
        entry["attempts"] = int(entry.get("attempts") or 0) + 1
        if correct:
            entry["correct"] = int(entry.get("correct") or 0) + 1
        else:
            entry["wrong"] = int(entry.get("wrong") or 0) + 1
        # EMA toward 1.0 on correct, 0.0 on wrong
        prev = float(entry.get("memory_score", 0.5))
        target = 1.0 if correct else 0.0
        entry["memory_score"] = round(prev * 0.7 + target * 0.3, 4)
        entry["last_quiz_at"] = ts
        history = list(entry.get("history") or [])
        history.append(
            {
                "ts": ts,
                "correct": correct,
                "question_id": question.get("id"),
                "prompt": (question.get("prompt") or "")[:180],
            }
        )
        entry["history"] = history[-40:]

    save_scores(db, scores, summary="Record quiz answer scores")

    # Mirror onto session entity
    if sid:
        session = get_session(db, sid)
        if session:
            sess_entry = _score_entry(scores, session_id=sid)
            session["memory_score"] = sess_entry.get("memory_score", 0.5)
            session["last_quiz_at"] = ts
            hist = list(session.get("score_history") or [])
            hist.append({"ts": ts, "correct": correct, "memory_score": session["memory_score"]})
            session["score_history"] = hist[-40:]
            db.write_json(
                session_path(sid),
                session,
                create=False,
                summary=f"Update memory_score after quiz for {sid}",
            )
            # refresh index entry lightly via research_sessions
            from ollama.research_sessions import _index_session

            _index_session(db, session)


def _public_question(q: dict[str, Any], *, reveal_answer: bool) -> dict[str, Any]:
    out = {
        "id": q.get("id"),
        "session_id": q.get("session_id"),
        "topic_slug": q.get("topic_slug"),
        "subject": q.get("subject"),
        "prompt": q.get("prompt"),
        "status": q.get("status"),
    }
    if reveal_answer or q.get("status") == "answered":
        out["expected_answer"] = q.get("expected_answer")
        out["user_answer"] = q.get("user_answer")
        out["correct"] = q.get("correct")
        if q.get("brief_correction"):
            out["brief_correction"] = q.get("brief_correction")
    return out


def _public_quiz(quiz: dict[str, Any], *, reveal_answer: bool = False) -> dict[str, Any]:
    questions = quiz.get("questions") or []
    idx = int(quiz.get("current_index") or 0)
    current = None
    if isinstance(questions, list) and 0 <= idx < len(questions) and isinstance(questions[idx], dict):
        current = _public_question(questions[idx], reveal_answer=reveal_answer)
    return {
        "id": quiz.get("id"),
        "status": quiz.get("status"),
        "started_at": quiz.get("started_at"),
        "ended_at": quiz.get("ended_at"),
        "current_index": idx,
        "total": len(questions) if isinstance(questions, list) else 0,
        "answered": quiz.get("answered"),
        "correct_count": quiz.get("correct_count"),
        "wrong_count": quiz.get("wrong_count"),
        "current_question": current,
    }
