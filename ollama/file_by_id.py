"""SOI files by Changelog/Inbox id — host copies the stored text. Do not retype turns."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from ainet.tools import changelog
from ainet.tools.ops import DatabaseTools
from ollama.content_filing import (
    content_kind,
    cop_name_in_text,
)
from ollama.topics import record_personal_filing


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _entry_text(entry: dict[str, Any]) -> tuple[str, str]:
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    user = str(details.get("user_text") or entry.get("summary") or "").strip()
    assistant = str(details.get("assistant_text") or "").strip()
    return user, assistant


_NAMED_DEST_KINDS = {
    "identity": {"identity", "voice", "psychology"},
    "personality": {"identity", "voice"},
    "voice": {"voice", "identity"},
    "psychology": {"psychology", "identity"},
    "habits": {"habits"},
}


def _qualify_dest(dest: str) -> str:
    dest = dest.replace("\\", "/").strip()
    if dest.startswith(("Hayden/", "School/", "Work/", "Household/", "runtime/")):
        return dest
    if dest.startswith(("Preferences/", "Habits/", "Inbox/", "Relationships/", "Body/", "Psychology/", "Desires/", "Values/", "Identity/", "Memories/")):
        return f"Hayden/{dest}"
    return dest


def _append_note(db: DatabaseTools, path: str, note: str, summary: str) -> bool:
    if not note:
        return False
    if not db.paths.resolve(path).exists():
        return False
    data = db.read_json(path)["data"]
    if not isinstance(data, dict):
        return False
    field = None
    for candidate in (
        "notes",
        "likes",
        "environments",
        "routines_i_like",
        "favorites",
        "items",
        "goals",
        "active",
        "values",
        "triggers",
        "rituals",
        "plans",
        "next_steps",
        "low",
        "what_i_know_about_them",
    ):
        if isinstance(data.get(candidate), list):
            field = candidate
            break
    if field is None:
        data["notes"] = []
        field = "notes"
    bucket = data[field]
    if note not in bucket:
        bucket.append(note)
    data[field] = bucket
    data["last_updated"] = _utc_now()
    db.write_json(path, data, summary=summary)
    return True


def file_by_id(
    db: DatabaseTools,
    *,
    entry_id: str = "",
    entry_ids: list[Any] | None = None,
    inbox_id: str = "",
    dest: str = "",
    subject: str = "",
    topic_slug: str = "",
    summary: str | None = None,
) -> dict[str, Any]:
    """File stored Changelog/Inbox content by id. SOI must not paste turn bodies."""
    dest_raw = str(dest or "").strip()
    if not dest_raw:
        return {"ok": False, "error": "dest is required (discard | named dest | leaf path)"}

    ids = [str(x).strip() for x in (entry_ids or []) if str(x).strip()]
    if entry_id and str(entry_id).strip() not in ids:
        ids.insert(0, str(entry_id).strip())

    inbox = str(inbox_id or "").strip()
    if inbox:
        return _file_inbox(db, inbox, dest_raw)

    if not ids:
        return {"ok": False, "error": "entry_id or entry_ids required"}

    dest_norm = dest_raw.lower().strip()
    first = changelog.get_entry(db.paths, ids[0])
    user0, asst0 = _entry_text(first or {})
    kind0 = content_kind(user0, asst0)
    if dest_norm in {"research", "session"} or dest_raw.replace("\\", "/").startswith(
        "Hayden/Research"
    ):
        return {
            "ok": False,
            "error": (
                "Research is removed. File into School/, Work/, Household/, or a Hayden leaf "
                "(dest=psychology|identity|habits|voice or a .json path)."
            ),
            "entry_ids": ids,
        }
    if dest_norm in {"discard", "ephemeral", "drop"}:
        return {
            "ok": True,
            "action": "discard",
            "entry_ids": ids,
            "note": "Host will archive these to Masterlog.json and drop them from the Changelog queue.",
        }

    if dest_norm in {"identity", "personality", "voice", "psychology", "habits"}:
        texts = [changelog.get_entry(db.paths, eid) for eid in ids]
        user_bits: list[str] = []
        asst_bits: list[str] = []
        for entry in texts:
            if not entry:
                continue
            u, a = _entry_text(entry)
            if u:
                user_bits.append(u)
            if a:
                asst_bits.append(a)
        path = record_personal_filing(
            db,
            dest_norm,
            user=" | ".join(user_bits),
            assistant=" ".join(asst_bits)[:900],
            entry_ids=ids,
        )
        if dest_norm in {"voice", "identity", "personality"}:
            _append_voice_evidence(db, " ".join(user_bits))
        return {
            "ok": True,
            "action": dest_norm,
            "entry_ids": ids,
            "filed_to": path or f"Hayden/{dest_norm.title()}/Notes.json",
        }

    path = _qualify_dest(dest_raw)
    path_norm = path.replace("\\", "/")
    if path_norm.startswith("Hayden/Identity") and kind0 not in {"identity", "voice"}:
        return {
            "ok": False,
            "error": (
                f"{path_norm} refused — this turn is {kind0}, not identity. "
                "Inspect domain_snapshot and file in the matching Folderrules domain."
            ),
            "entry_ids": ids,
        }
    if path_norm.startswith("Hayden/Research"):
        return {
            "ok": False,
            "error": "Research is removed. Use a Folderrules leaf under School/, Work/, Household/, or Hayden/.",
            "entry_ids": ids,
        }
    if path_norm.startswith(("School/", "Work/", "Household/")) and kind0 in {
        "psychology",
        "identity",
        "voice",
        "habits",
    }:
        return {
            "ok": False,
            "error": (
                f"{path_norm} refused — this turn is {kind0}. "
                "file_by_id dest=psychology|identity|habits|voice, not a School/Work COP."
            ),
            "entry_ids": ids,
        }
    if ("/Courses/" in path_norm or "/Projects/" in path_norm) and not cop_name_in_text(
        path_norm, user0
    ):
        return {
            "ok": False,
            "error": (
                f"{path_norm} refused — that COP name is not in user_text. "
                "Do not invent courses or projects."
            ),
            "entry_ids": ids,
        }
    if "inbox" in path_norm.lower():
        return {
            "ok": False,
            "error": (
                "Inbox is not a filing dest for Changelog turns. "
                "Use Folderrules (School/, Work/, Hayden/ leaves) or create_cop."
            ),
            "entry_ids": ids,
        }
    if not path.endswith(".json"):
        leaf = None
        for name in ("Plan.json", "Notes.json", "History.json"):
            candidate = f"{path_norm.rstrip('/')}/{name}"
            if db.paths.resolve(candidate).exists():
                leaf = candidate
                break
        if not leaf:
            return {
                "ok": False,
                "error": f"dest must be discard, a named dest, or a .json leaf path (got {dest_raw})",
            }
        path = leaf
        path_norm = path

    filed: list[str] = []
    for eid in ids:
        entry = changelog.get_entry(db.paths, eid)
        if not entry:
            continue
        user, _asst = _entry_text(entry)
        if not user:
            continue
        if not db.paths.resolve(path).exists():
            # Person dossiers: create a thin file so the id can land.
            if "/People/" in path:
                name = path.rsplit("/", 1)[-1].removesuffix(".json")
                db.create_json(
                    path,
                    {
                        "name": name,
                        "aliases": [],
                        "how_we_met": user,
                        "relationship_type": "acquaintance",
                        "status": "active",
                        "closeness": 0.3,
                        "how_i_feel": "",
                        "how_i_act_around_them": "",
                        "what_they_know_about_me": [],
                        "what_i_know_about_them": [user],
                        "shared_history": [],
                        "boundaries": [],
                        "secrets_involving_them": [],
                        "triggers_around_them": [],
                        "tags": [],
                        "related_paths": [],
                        "last_updated": _utc_now(),
                    },
                    summary=f"Create person dossier from entry {eid}",
                )
                filed.append(eid)
                continue
            return {"ok": False, "error": f"Leaf does not exist: {path}", "entry_id": eid}
        _append_note(db, path, user, summary or f"File oac_turn {eid} into {path}")
        filed.append(eid)

    return {"ok": True, "action": "leaf", "entry_ids": filed, "filed_to": path}


def _file_inbox(db: DatabaseTools, inbox_id: str, dest: str) -> dict[str, Any]:
    path = "Hayden/Inbox/Captures.json"
    if not db.paths.resolve(path).exists():
        return {"ok": False, "error": "Inbox Captures.json missing"}
    data = db.read_json(path)["data"]
    if not isinstance(data, dict):
        return {"ok": False, "error": "Captures.json must be an object"}
    captures = data.get("captures") if isinstance(data.get("captures"), list) else []
    dest_norm = dest.lower().strip()
    dest_path = "" if dest_norm in {"discard", "ephemeral", "drop"} else _qualify_dest(dest)
    found = False
    for cap in captures:
        if not isinstance(cap, dict) or str(cap.get("id") or "") != inbox_id:
            continue
        found = True
        text = str(cap.get("text") or "").strip()
        if dest_norm in {"discard", "ephemeral", "drop"}:
            cap["status"] = "discarded"
            cap["filed_to"] = ""
        else:
            if dest_path.endswith(".json") and db.paths.resolve(dest_path).exists() and text:
                _append_note(db, dest_path, text, summary=f"File inbox {inbox_id} into {dest_path}")
            cap["status"] = "filed"
            cap["filed_to"] = dest_path
        break
    if not found:
        return {"ok": False, "error": f"Unknown inbox id: {inbox_id}"}
    data["captures"] = captures
    data["last_updated"] = _utc_now()
    db.write_json(path, data, summary=f"Inbox {inbox_id} → {dest}")
    return {"ok": True, "action": "inbox", "inbox_id": inbox_id, "filed_to": dest_path or "discarded"}


def _append_voice_evidence(db: DatabaseTools, user_text: str) -> None:
    path = "Hayden/Identity/Voice.json"
    if not user_text or not db.paths.resolve(path).exists():
        return
    data = db.read_json(path)["data"]
    if not isinstance(data, dict):
        return
    said = data.get("things_i_say_a_lot") if isinstance(data.get("things_i_say_a_lot"), list) else []
    clip = user_text if len(user_text) <= 80 else user_text[:80] + "…"
    if clip and clip not in said:
        said.append(clip)
    data["things_i_say_a_lot"] = said[-24:]
    low = user_text.lower()
    if re.search(r"\b(fuck|shit|ass|bitch|damn|crap|retard)\b", low):
        data["swearing"] = "present in prompts — see things_i_say_a_lot"
    if not str(data.get("tone") or "").strip():
        data["tone"] = "casual spoken; files from how Hayden actually types"
    data["last_updated"] = _utc_now()
    db.write_json(path, data, summary="File voice evidence from prompt")
