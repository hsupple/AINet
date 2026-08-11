"""Helpers for Hayden/Research topic threads."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from ainet.defaults import load_default
from ainet.tools.ops import DatabaseTools


_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify_topic(title: str) -> str:
    parts = [p for p in _SLUG_RE.split(title.strip()) if p]
    slug = "-".join(parts)[:64].strip("-")
    return slug or "Topic"


def topic_root(slug: str) -> str:
    return f"Hayden/Research/Topics/{slug}"


def ensure_topic(db: DatabaseTools, title: str) -> dict[str, Any]:
    """Create or open a research topic COP-like folder. Returns paths + slug."""
    from ollama.research_sessions import ensure_research_scaffold

    ensure_research_scaffold(db)
    slug = slugify_topic(title)
    root = topic_root(slug)
    db.create_folder(root, summary=f"Research topic: {title}")
    read_path = f"{root}/Read.json"
    notes_path = f"{root}/Notes.json"
    history_path = f"{root}/History.json"

    if not db.paths.resolve(read_path).exists():
        read = load_default("Read.json")
        read["summary"] = f"Research thread: {title}"
        read["state"] = "active"
        read["important_context"] = [f"Topic: {title}"]
        read["active_items"] = ["Establish what we already covered", "List open questions"]
        db.write_json(read_path, read, create=True, summary=f"Seed research Read for {title}")
    if not db.paths.resolve(notes_path).exists():
        db.write_json(
            notes_path,
            {
                "title": title,
                "key_claims": [],
                "mechanisms": [],
                "timeline": [],
                "open_questions": [],
                "resources": [],
                "last_updated": "",
            },
            create=True,
            summary=f"Seed research Notes for {title}",
        )
    if not db.paths.resolve(history_path).exists():
        db.write_json(
            history_path,
            load_default("History.json"),
            create=True,
            summary=f"Seed research History for {title}",
        )

    # Index
    index_path = "Hayden/Research/Index.json"
    if db.paths.resolve(index_path).exists():
        index = db.read_json(index_path)["data"]
    else:
        index = {"topics": [], "sessions": [], "last_updated": ""}
    if not isinstance(index, dict):
        index = {"topics": [], "sessions": [], "last_updated": ""}
    index.setdefault("sessions", [])
    topics = index.setdefault("topics", [])
    if not any(t.get("slug") == slug for t in topics if isinstance(t, dict)):
        topics.append({"slug": slug, "title": title, "path": root})
        db.write_json(index_path, index, create=True, summary=f"Index research topic {slug}")

    return {
        "ok": True,
        "slug": slug,
        "title": title,
        "path": root,
        "read": read_path,
        "notes": notes_path,
        "history": history_path,
    }


def load_topic_context(db: DatabaseTools, slug: str, *, lean: bool = True) -> str:
    """Build continuity stub for a topic.

    lean=True (default): path + Read summary/active_items only — model must tool-call for Notes.
    lean=False: include compact Notes key fields (still not a full dump).
    """
    root = topic_root(slug)
    read_path = f"{root}/Read.json"
    lines = [f"Active topic path: {root}", "Fetch Notes.json via read_json only if needed."]

    read_file = db.paths.resolve(read_path)
    if read_file.exists():
        read = db.read_json(read_path)["data"]
        if isinstance(read, dict):
            if read.get("summary"):
                lines.append(f"summary: {read['summary']}")
            state = read.get("state")
            if state:
                lines.append(f"state: {state}")
            active = read.get("active_items") or []
            if isinstance(active, list) and active:
                lines.append("active: " + "; ".join(str(x) for x in active[:5]))
            open_q = []
            # common place people stash questions on Read
            for key in ("uncertainties",):
                vals = read.get(key) or []
                if isinstance(vals, list):
                    open_q.extend(vals[:3])
            if open_q:
                lines.append("open: " + "; ".join(_short(x) for x in open_q))

    if not lean:
        notes_path = f"{root}/Notes.json"
        if db.paths.resolve(notes_path).exists():
            notes = db.read_json(notes_path)["data"]
            if isinstance(notes, dict):
                for key in ("key_claims", "open_questions", "mechanisms"):
                    vals = notes.get(key) or []
                    if isinstance(vals, list) and vals:
                        lines.append(f"{key}: " + "; ".join(_short(x) for x in vals[:4]))

    return "\n".join(lines)


def latest_open_research_subject(db: DatabaseTools) -> str | None:
    index_path = "Hayden/Research/Index.json"
    if not db.paths.resolve(index_path).exists():
        return None
    index = db.read_json(index_path)["data"]
    sessions = index.get("sessions") if isinstance(index, dict) else None
    if not isinstance(sessions, list) or not sessions:
        return None
    for row in reversed(sessions):
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "open") not in {"open", ""}:
            continue
        title = str(row.get("subject") or row.get("title") or "").strip()
        low = title.lower()
        if not title or low in {"research thread", "research"} or "mentioned" in low:
            continue
        return title
    return None


def record_topic_filing(
    db: DatabaseTools,
    slug: str,
    title: str,
    *,
    user: str,
    assistant: str,
    entry_ids: list[str],
) -> None:
    """Write lasting claims into Notes.json and an event into History.json."""
    ensure_topic(db, title)
    root = topic_root(slug)
    now = _utc_now()
    notes_path = f"{root}/Notes.json"
    hist_path = f"{root}/History.json"
    if db.paths.resolve(notes_path).exists():
        notes = db.read_json(notes_path)["data"]
        if isinstance(notes, dict):
            claims = notes.get("key_claims") if isinstance(notes.get("key_claims"), list) else []
            mechs = notes.get("mechanisms") if isinstance(notes.get("mechanisms"), list) else []
            if user and user not in claims:
                claims.append(user)
            clip = assistant.strip()
            if len(clip) > 500:
                clip = clip[:500] + "…"
            if clip and clip not in mechs:
                mechs.append(clip)
            notes["key_claims"] = claims[:40]
            notes["mechanisms"] = mechs[:40]
            notes["last_updated"] = now
            db.write_json(notes_path, notes, summary=f"File topic notes for {title}")
    if db.paths.resolve(hist_path).exists():
        hist = db.read_json(hist_path)["data"]
        if isinstance(hist, dict):
            events = hist.get("events") if isinstance(hist.get("events"), list) else []
            events.append(
                {
                    "id": (entry_ids[0][:12] if entry_ids else now[-12:]),
                    "timestamp": now,
                    "type": "filed_turn",
                    "content": user,
                    "source": ",".join(entry_ids),
                    "importance": 0.6,
                    "confidence": 0.8,
                    "tags": ["research"],
                    "related_entities": list(entry_ids),
                }
            )
            hist["events"] = events[-80:]
            db.write_json(hist_path, hist, summary=f"File topic history for {title}")


def record_personal_filing(
    db: DatabaseTools,
    domain: str,
    *,
    user: str,
    assistant: str,
    entry_ids: list[str],
) -> str:
    """Identity / Psychology / Habits — Notes + History, same shape as research topics."""
    folder = {
        "identity": "Hayden/Identity",
        "personality": "Hayden/Identity",
        "voice": "Hayden/Identity",
        "psychology": "Hayden/Psychology",
        "habits": "Hayden/Habits",
    }.get(domain.lower().strip(), "")
    if not folder:
        return ""
    now = _utc_now()
    notes_path = f"{folder}/Notes.json"
    hist_path = f"{folder}/History.json"
    if not db.paths.resolve(notes_path).exists():
        db.write_json(
            notes_path,
            {
                "title": folder.rsplit("/", 1)[-1],
                "key_claims": [],
                "evidence": [],
                "last_updated": "",
            },
            create=True,
            summary=f"Seed {folder} Notes",
        )
    if not db.paths.resolve(hist_path).exists():
        db.write_json(
            hist_path,
            load_default("History.json"),
            create=True,
            summary=f"Seed {folder} History",
        )
    notes = db.read_json(notes_path)["data"]
    if isinstance(notes, dict):
        evidence = notes.get("evidence") if isinstance(notes.get("evidence"), list) else []
        claims = notes.get("key_claims") if isinstance(notes.get("key_claims"), list) else []
        blob = user.strip()
        if blob and blob not in claims:
            claims.append(blob)
        evidence.append(
            {
                "text": blob,
                "assistant_clip": (assistant[:240] + "…") if len(assistant) > 240 else assistant,
                "entry_ids": list(entry_ids),
                "at": now,
            }
        )
        notes["key_claims"] = claims[:60]
        notes["evidence"] = evidence[-80:]
        notes["last_updated"] = now
        db.write_json(notes_path, notes, summary=f"File personal evidence into {folder}")
    hist = db.read_json(hist_path)["data"]
    if isinstance(hist, dict):
        events = hist.get("events") if isinstance(hist.get("events"), list) else []
        events.append(
            {
                "id": (entry_ids[0][:12] if entry_ids else now[-12:]),
                "timestamp": now,
                "type": "filed_turn",
                "content": user,
                "source": ",".join(entry_ids),
                "importance": 0.5,
                "confidence": 0.7,
                "tags": [domain],
                "related_entities": list(entry_ids),
            }
        )
        hist["events"] = events[-80:]
        db.write_json(hist_path, hist, summary=f"File personal history into {folder}")
    return notes_path


def _short(value: Any, limit: int = 120) -> str:
    if isinstance(value, dict):
        text = str(value.get("text") or value.get("title") or value)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"

