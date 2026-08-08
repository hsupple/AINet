"""Helpers for Hayden/Research topic threads."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ainet.defaults import load_default
from ainet.tools.ops import DatabaseTools


_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def slugify_topic(title: str) -> str:
    parts = [p for p in _SLUG_RE.split(title.strip()) if p]
    slug = "-".join(parts)[:64].strip("-")
    return slug or "Topic"


def topic_root(slug: str) -> str:
    return f"Hayden/Research/Topics/{slug}"


def ensure_topic(db: DatabaseTools, title: str) -> dict[str, Any]:
    """Create or open a research topic COP-like folder. Returns paths + slug."""
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
        index = {"topics": [], "last_updated": ""}
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


def _short(value: Any, limit: int = 120) -> str:
    if isinstance(value, dict):
        text = str(value.get("text") or value.get("title") or value)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"

