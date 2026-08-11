"""Personal-domain filing helpers (Identity / Psychology / Habits)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ainet.defaults import load_default
from ainet.tools.ops import DatabaseTools


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def record_personal_filing(
    db: DatabaseTools,
    domain: str,
    *,
    user: str,
    assistant: str,
    entry_ids: list[str],
) -> str:
    """Identity / Psychology / Habits — Notes + History."""
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
