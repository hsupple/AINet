"""Phase 1 filing — AI writes a note; host stores evidence in History.json."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ainet.tools import changelog
from ainet.tools.ops import DatabaseTools
from ainet.defaults import load_default_for_path
from ollama.dest_resolver import resolve_dest

_TOPIC_FILES = ("Notes.json", "History.json", "Read.json", "Schedule.json")
_QUESTIONS_FILES = ("Notes.json", "History.json")
_INBOX_PATHS = ("Hayden/Inbox", "Inbox")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _entry_user_text(entry: dict[str, Any]) -> str:
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    return str(details.get("user_text") or entry.get("summary") or "").strip()


def ensure_topic_folder(db: DatabaseTools, folder: str) -> dict[str, Any]:
    """Ensure folder exists with Notes, History, Read, Schedule."""
    folder = folder.replace("\\", "/").strip("/")
    created: list[str] = []
    seed_files = _QUESTIONS_FILES if folder == "Questions" or folder.startswith("Questions/") else _TOPIC_FILES
    if not db.paths.resolve(folder).exists():
        db.create_folder(folder, summary=f"Create topic folder {folder}")
        created.append(folder)
    for name in seed_files:
        path = f"{folder}/{name}"
        if not db.paths.resolve(path).exists():
            db.create_json(path, load_default_for_path(path), summary=f"Seed {name} for {folder}")
            created.append(path)
    return {"ok": True, "folder": folder, "created": created}


def _collect_ids(entry_id: str, entry_ids: list[Any] | None) -> list[str]:
    seen: list[str] = []
    for raw in list(entry_ids or []) + [entry_id]:
        eid = str(raw or "").strip()
        if eid and eid not in seen:
            seen.append(eid)
    return seen


def _note_has_id(note: Any, eid: str) -> bool:
    if not isinstance(note, dict):
        return False
    if str(note.get("id") or "") == eid:
        return True
    ids = note.get("ids")
    if isinstance(ids, list) and eid in {str(x) for x in ids}:
        return True
    return False


def file_note(
    db: DatabaseTools,
    *,
    entry_id: str = "",
    entry_ids: list[Any] | None = None,
    dest: str = "",
    text: str = "",
    summary: str | None = None,
) -> dict[str, Any]:
    """File changelog turn(s): AI note → Notes.json; raw message + id → History.json."""
    ids = _collect_ids(entry_id, entry_ids)
    note_text = str(text or "").strip()
    dest_raw = str(dest or "").strip()

    if not ids:
        return {"ok": False, "error": "entry_id or entry_ids is required"}
    if not dest_raw:
        return {"ok": False, "error": "dest is required (e.g. Values, Memories, Questions, discard)"}
    if dest_raw.lower() not in {"discard", "drop", "ephemeral"} and not note_text:
        return {"ok": False, "error": "text is required — write a short note about the turn"}

    resolved: list[tuple[str, dict[str, Any]]] = []
    missing: list[str] = []
    for eid in ids:
        entry = changelog.get_entry(db.paths, eid)
        if not entry:
            missing.append(eid)
        else:
            resolved.append((eid, entry))
    if not resolved:
        return {"ok": False, "error": f"Unknown entry id: {ids[0]}", "entry_ids": ids}

    primary_id, primary_entry = resolved[0]
    user_text = _entry_user_text(primary_entry)
    folder = resolve_dest(db, dest_raw, user_text=user_text)
    filed_ids = [eid for eid, _entry in resolved]

    if folder == "discard":
        return {
            "ok": True,
            "action": "discard",
            "entry_id": primary_id,
            "entry_ids": filed_ids,
            "dest": dest_raw,
            "missing": missing or None,
        }

    if not folder:
        return {
            "ok": False,
            "error": f"Unknown dest: {dest_raw!r}. Pick a folder from the folders list or create one with create_folder first.",
            "entry_id": primary_id,
            "entry_ids": filed_ids,
        }
    if folder in _INBOX_PATHS or folder.startswith("Hayden/Inbox/"):
        return {
            "ok": False,
            "error": "dest=Hayden/Inbox is blocked for SOI filing. Pick a real long-term folder.",
            "entry_id": primary_id,
            "entry_ids": filed_ids,
        }

    ensure_topic_folder(db, folder)
    now = _utc_now()
    dest_label = dest_raw.strip()

    notes_path = f"{folder}/Notes.json"
    notes_doc = db.read_json(notes_path)["data"]
    if not isinstance(notes_doc, dict):
        notes_doc = {"notes": []}
    notes = notes_doc.setdefault("notes", [])
    if not isinstance(notes, list):
        notes = []
        notes_doc["notes"] = notes
    already = any(_note_has_id(n, eid) for n in notes for eid in filed_ids)
    if not already:
        row: dict[str, Any] = {
            "text": note_text,
            "id": primary_id,
            "dest": dest_label,
            "filed_at": now,
        }
        if len(filed_ids) > 1:
            row["ids"] = filed_ids
        notes.append(row)
    notes_doc["last_updated"] = now
    db.write_json(
        notes_path,
        notes_doc,
        summary=summary or f"Note for {primary_id} → {dest_label}",
    )

    hist_path = f"{folder}/History.json"
    hist_doc = db.read_json(hist_path)["data"]
    if not isinstance(hist_doc, dict):
        hist_doc = {"events": []}
    events = hist_doc.setdefault("events", [])
    if not isinstance(events, list):
        events = []
        hist_doc["events"] = events
    existing = {str(ev.get("id") or "") for ev in events if isinstance(ev, dict)}
    for eid, entry in resolved:
        if eid in existing:
            continue
        events.append(
            {
                "id": eid,
                "timestamp": str(entry.get("ts") or now),
                "type": "oac_turn",
                "content": _entry_user_text(entry),
                "source": "changelog",
                "importance": 0.5,
                "confidence": 1.0,
                "tags": [dest_label],
                "related_entities": [x for x in filed_ids if x != eid],
            }
        )
    db.write_json(hist_path, hist_doc, summary=f"History evidence for {primary_id}")

    if not (folder == "Questions" or folder.startswith("Questions/")):
        db.mark_read_stale(folder, f"New note filed ({dest_label})", source_path=notes_path)

    return {
        "ok": True,
        "action": "note",
        "entry_id": primary_id,
        "entry_ids": filed_ids,
        "dest": dest_label,
        "folder": folder,
        "notes_path": notes_path,
        "history_path": hist_path,
        "note_preview": note_text[:200],
        "missing": missing or None,
    }
