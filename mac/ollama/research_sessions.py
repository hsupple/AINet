"""Research session entities under Hayden/Research/Sessions/."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from ainet.defaults import load_default
from ainet.tools.ops import DatabaseTools
from ollama.topics import slugify_topic, topic_root


SESSIONS_DIR = "Hayden/Research/Sessions"
INDEX_PATH = "Hayden/Research/Index.json"
SCORES_PATH = "Hayden/Research/Scores.json"
RESEARCH_READ = "Hayden/Research/Read.json"

_SESSION_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


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


def make_session_id(subject_or_slug: str = "", *, when: datetime | None = None) -> str:
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    base = slugify_topic(subject_or_slug) if subject_or_slug else "Session"
    base = _SESSION_ID_RE.sub("-", base).strip("-")[:48] or "Session"
    return f"{base}-{stamp}-{uuid.uuid4().hex[:6]}"


def session_path(session_id: str) -> str:
    return f"{SESSIONS_DIR}/{session_id}.json"


def ensure_research_scaffold(db: DatabaseTools) -> None:
    """Ensure Sessions/, Index sessions list, and Scores.json exist."""
    db.create_folder(SESSIONS_DIR, summary="Research conversation sessions")
    _ensure_index(db)
    _ensure_scores(db)
    if db.paths.resolve(RESEARCH_READ).exists():
        read = db.read_json(RESEARCH_READ)["data"]
        if isinstance(read, dict):
            ctx = read.get("important_context") or []
            if isinstance(ctx, list) and "Sessions/" not in ctx:
                ctx = list(ctx) + ["Sessions/", "Scores.json"]
                db.patch_json(
                    RESEARCH_READ,
                    {"important_context": ctx[:12]},
                    summary="Point Research Read at Sessions + Scores",
                )


def _ensure_index(db: DatabaseTools) -> dict[str, Any]:
    if db.paths.resolve(INDEX_PATH).exists():
        index = db.read_json(INDEX_PATH)["data"]
        if not isinstance(index, dict):
            index = {"topics": [], "sessions": [], "last_updated": ""}
    else:
        index = {"topics": [], "sessions": [], "last_updated": ""}
    index.setdefault("topics", [])
    index.setdefault("sessions", [])
    index.setdefault("last_updated", "")
    if not db.paths.resolve(INDEX_PATH).exists():
        db.write_json(INDEX_PATH, index, create=True, summary="Create Research Index")
    return index


def _ensure_scores(db: DatabaseTools) -> dict[str, Any]:
    if db.paths.resolve(SCORES_PATH).exists():
        data = db.read_json(SCORES_PATH)["data"]
        if isinstance(data, dict):
            data.setdefault("sessions", {})
            data.setdefault("topics", {})
            data.setdefault("last_updated", "")
            return data
    scores = load_default("ResearchScores.json")
    if not isinstance(scores, dict):
        scores = {"sessions": {}, "topics": {}, "last_updated": ""}
    db.write_json(SCORES_PATH, scores, create=True, summary="Seed Research Scores")
    return scores


def list_sessions(db: DatabaseTools, *, status: str | None = None) -> list[dict[str, Any]]:
    ensure_research_scaffold(db)
    root = db.paths.resolve(SESSIONS_DIR)
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*.json")):
        rel = f"{SESSIONS_DIR}/{path.name}"
        try:
            data = db.read_json(rel)["data"]
        except (OSError, ValueError, KeyError):
            continue
        if not isinstance(data, dict):
            continue
        if status and data.get("status") != status:
            continue
        out.append(data)
    return out


def get_session(db: DatabaseTools, session_id: str) -> dict[str, Any] | None:
    path = session_path(session_id)
    if not db.paths.resolve(path).exists():
        return None
    data = db.read_json(path)["data"]
    return data if isinstance(data, dict) else None


def upsert_research_session(
    db: DatabaseTools,
    *,
    session_id: str | None = None,
    subject: str = "",
    title: str = "",
    topic_slug: str = "",
    topic_path: str = "",
    details_covered: list[Any] | None = None,
    append_details: bool = True,
    length_turns: int | None = None,
    started_at: str | None = None,
    related_topic: str = "",
    source_session_ids: list[str] | None = None,
    changelog_entry_ids: list[str] | None = None,
    notes: str = "",
    status: str | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    """Create or merge a research session entity and refresh Index."""
    ensure_research_scaffold(db)
    subject = (subject or title or topic_slug or "Research session").strip()
    title = (title or subject).strip()
    if not session_id:
        session_id = make_session_id(topic_slug or subject)
    path = session_path(session_id)
    existing = get_session(db, session_id)
    if existing is None:
        data = load_default("ResearchSession.json")
        if not isinstance(data, dict):
            data = {}
        data.update(
            {
                "id": session_id,
                "subject": subject,
                "title": title,
                "topic_slug": topic_slug or "",
                "topic_path": topic_path
                or (topic_root(topic_slug) if topic_slug else ""),
                "started_at": started_at or _utc_now(),
                "ended_at": "",
                "status": status or "open",
                "duration_seconds": None,
                "length_turns": int(length_turns or 0),
                "details_covered": [],
                "related_topic": related_topic or topic_slug or "",
                "last_quiz_at": "",
                "memory_score": 0.5,
                "score_history": [],
                "source_session_ids": [],
                "changelog_entry_ids": [],
                "notes": notes or "",
            }
        )
        created = True
    else:
        data = dict(existing)
        created = False
        if subject:
            data["subject"] = subject
        if title:
            data["title"] = title
        if topic_slug:
            data["topic_slug"] = topic_slug
            data["topic_path"] = topic_path or topic_root(topic_slug)
        elif topic_path:
            data["topic_path"] = topic_path
        if related_topic:
            data["related_topic"] = related_topic
        if started_at and not data.get("started_at"):
            data["started_at"] = started_at
        if length_turns is not None:
            data["length_turns"] = int(length_turns)
        if notes:
            data["notes"] = notes
        if status:
            data["status"] = status

    if details_covered:
        normalized = [_normalize_detail(d) for d in details_covered]
        if append_details:
            existing_details = list(data.get("details_covered") or [])
            seen = {_detail_key(d) for d in existing_details if isinstance(d, dict)}
            for item in normalized:
                key = _detail_key(item)
                if key not in seen:
                    existing_details.append(item)
                    seen.add(key)
            data["details_covered"] = existing_details
        else:
            data["details_covered"] = normalized

    if source_session_ids:
        merged = list(data.get("source_session_ids") or [])
        for sid in source_session_ids:
            if sid and sid not in merged:
                merged.append(sid)
        data["source_session_ids"] = merged

    if changelog_entry_ids:
        merged = list(data.get("changelog_entry_ids") or [])
        for eid in changelog_entry_ids:
            if eid and eid not in merged:
                merged.append(eid)
        data["changelog_entry_ids"] = merged

    # Pull durable score if present
    scores = _ensure_scores(db)
    sess_score = (scores.get("sessions") or {}).get(session_id)
    if isinstance(sess_score, dict) and sess_score.get("memory_score") is not None:
        data["memory_score"] = float(sess_score["memory_score"])
        if sess_score.get("last_quiz_at"):
            data["last_quiz_at"] = sess_score["last_quiz_at"]

    db.write_json(
        path,
        data,
        create=created,
        summary=summary or f"{'Create' if created else 'Update'} research session {session_id}",
    )
    _index_session(db, data)
    return {"ok": True, "created": created, "path": path, "session": data}


def complete_research_session(
    db: DatabaseTools,
    session_id: str,
    *,
    ended_at: str | None = None,
    details_covered: list[Any] | None = None,
    length_turns: int | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    """Mark a research session complete (ended_at + duration)."""
    ensure_research_scaffold(db)
    existing = get_session(db, session_id)
    if existing is None:
        return {"ok": False, "error": f"Unknown research session: {session_id}"}

    if details_covered:
        upsert_research_session(
            db,
            session_id=session_id,
            details_covered=details_covered,
            append_details=True,
            length_turns=length_turns,
            summary="Append details before completing session",
        )
        existing = get_session(db, session_id) or existing
    elif length_turns is not None:
        existing["length_turns"] = int(length_turns)

    end = ended_at or _utc_now()
    start = _parse_iso(str(existing.get("started_at") or ""))
    end_dt = _parse_iso(end)
    duration = None
    if start and end_dt:
        duration = max(0, int((end_dt - start).total_seconds()))

    existing["ended_at"] = end
    existing["status"] = "complete"
    existing["duration_seconds"] = duration

    path = session_path(session_id)
    db.write_json(
        path,
        existing,
        create=False,
        summary=summary or f"Complete research session {session_id}",
    )
    _index_session(db, existing)
    return {"ok": True, "path": path, "session": existing}


def _normalize_detail(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        text = str(raw.get("text") or raw.get("point") or raw.get("q") or raw.get("title") or "")
        kind = str(raw.get("kind") or raw.get("type") or "point")
        tags = raw.get("tags") if isinstance(raw.get("tags"), list) else []
        out: dict[str, Any] = {"kind": kind, "text": text.strip()}
        if tags:
            out["tags"] = [str(t) for t in tags]
        if raw.get("answer") is not None:
            out["answer"] = str(raw["answer"])
        if raw.get("question") is not None:
            out["question"] = str(raw["question"])
            if not out["text"]:
                out["text"] = str(raw["question"])
        return out
    return {"kind": "point", "text": str(raw).strip()}


def _detail_key(detail: dict[str, Any]) -> str:
    return f"{detail.get('kind')}|{(detail.get('text') or '').strip().lower()}"


def _index_session(db: DatabaseTools, session: dict[str, Any]) -> None:
    index = _ensure_index(db)
    sessions = index.setdefault("sessions", [])
    entry = {
        "id": session.get("id"),
        "subject": session.get("subject") or session.get("title"),
        "path": session_path(str(session.get("id"))),
        "topic_slug": session.get("topic_slug") or session.get("related_topic") or "",
        "started_at": session.get("started_at") or "",
        "ended_at": session.get("ended_at") or "",
        "status": session.get("status") or "open",
        "memory_score": session.get("memory_score", 0.5),
        "length_turns": session.get("length_turns", 0),
        "duration_seconds": session.get("duration_seconds"),
    }
    sid = entry["id"]
    replaced = False
    for i, item in enumerate(sessions):
        if isinstance(item, dict) and item.get("id") == sid:
            sessions[i] = entry
            replaced = True
            break
    if not replaced:
        sessions.append(entry)
    index["last_updated"] = _utc_now()
    db.write_json(INDEX_PATH, index, create=True, summary=f"Index research session {sid}")
