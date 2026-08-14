"""Build the simplified filing payload for SOI test (phase 1 + phase 2)."""

from __future__ import annotations

import json
import re
from typing import Any

from ainet.tools.ops import DatabaseTools

_CREATE_UNDER = ("Hayden", "Household", "Projects", "Questions")
_HIDDEN_CHILDREN: dict[str, set[str]] = {
    # Inbox is not a long-term filing target (file_note blocks it).
    "Hayden": {"Inbox", "School", "Work"},
    # Questions/Research is not the Deep Research vault, but dest=Research is forbidden.
    "Questions": {"Research"},
}


def _child_folder_names(db: DatabaseTools, path: str) -> list[str]:
    if not db.paths.resolve(path).is_dir():
        return []
    try:
        listing = db.list_dir(path)
    except Exception:
        return []
    names: list[str] = []
    blocked = _HIDDEN_CHILDREN.get(path, set())
    for child in listing.get("children") or []:
        if not isinstance(child, dict) or child.get("type") != "dir":
            continue
        name = str(child.get("name") or "").strip()
        if name and name not in blocked:
            names.append(name)
    return sorted(names, key=str.lower)


def build_test_filing_payload(
    db: DatabaseTools,
    batch_changelog: list[dict[str, Any]],
    batch_inbox: list[dict[str, Any]],
    *,
    entry_for_soi,
    inbox_for_soi,
) -> dict[str, Any]:
    folders: dict[str, list[str]] = {}
    for domain in _CREATE_UNDER:
        folders[domain] = _child_folder_names(db, domain)

    return {
        "create_under": list(_CREATE_UNDER),
        "folders": folders,
        "changelog_entries": [entry_for_soi(e) for e in batch_changelog],
        "inbox_unfiled": [inbox_for_soi(c) for c in batch_inbox],
    }


def _group_changelog(entries: list[Any]) -> list[dict[str, Any]]:
    """Group turns by session so the model can resolve pronouns across a thread."""
    groups: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        sid = str(entry.get("session_id") or "").strip() or "(no session)"
        if sid not in index:
            index[sid] = len(groups)
            groups.append({"session_id": sid, "entries": []})
        groups[index[sid]]["entries"].append(
            {
                "id": entry.get("id"),
                "ts": entry.get("ts"),
                "user_text": entry.get("user_text"),
            }
        )
    return groups


def format_test_user_message(prompt: str, payload: dict[str, Any]) -> str:
    """User message: short header, then labeled batch sections (rules live in system)."""
    lines: list[str] = []
    header = (prompt or "").strip()
    if header:
        lines.append(header)
        lines.append("")

    create_under = payload.get("create_under") or []
    lines.append("create_under: " + ", ".join(str(x) for x in create_under))
    lines.append("Questions is the only create_under root you may file into directly.")
    lines.append("")

    folders = payload.get("folders") or {}
    lines.append("folders:")
    for domain in _CREATE_UNDER:
        kids = folders.get(domain) or []
        if kids:
            label = ", ".join(kids)
        elif domain == "Questions":
            label = "(none - dest=Questions)"
        else:
            label = "(none)"
        lines.append(f"  {domain}: {label}")
    lines.append("")

    lines.append("changelog_entries (grouped by session, chronological within each):")
    entries = payload.get("changelog_entries") or []
    lines.append(json.dumps(_group_changelog(entries), ensure_ascii=False, indent=2))

    inbox = payload.get("inbox_unfiled") or []
    if inbox:
        lines.append("")
        lines.append("inbox_unfiled:")
        lines.append(json.dumps(inbox, ensure_ascii=False, indent=2))

    return "\n".join(lines)


# ---- Phase 2: read refresh payload ----------------------------------------

_RETRIEVAL_ACTION = re.compile(
    r"\b(ask(?:ed)?|inquir(?:e|ed|y|ies)|request(?:ed)?|find|view|read|list|"
    r"retrieve|check|look(?:ed|ing)?|search(?:ed|ing)?|confirm(?:ed)?|"
    r"confirmation|acknowledg(?:e|ed|ment)?|repeat(?:ed)?)\b",
    re.I,
)
_MEMORY_SUBJECT = re.compile(
    r"\b(preferences?|likes?|dislikes?|questions?|memory|stored information|"
    r"database|folders?|read\.json|notes?\.json|history\.json)\b",
    re.I,
)


def _is_retrieval_noise(text: str) -> bool:
    """True when a source only records asking the AI to retrieve stored data."""
    value = str(text or "").strip()
    return bool(
        value
        and _RETRIEVAL_ACTION.search(value)
        and _MEMORY_SUBJECT.search(value)
    )


def _read_folder_jsons(db: DatabaseTools, folder: str, last_updated: str) -> dict[str, Any]:
    """Read every JSON in a folder. History.json is filtered to entries after last_updated."""
    folder_path = db.paths.resolve(folder)
    if not folder_path.is_dir():
        return {}
    files: dict[str, Any] = {}
    for child in sorted(folder_path.iterdir()):
        if not child.is_file() or child.suffix != ".json":
            continue
        try:
            text = child.read_text(encoding="utf-8")
            parsed = json.loads(text)
        except (json.JSONDecodeError, OSError):
            continue

        if child.name == "History.json" and isinstance(parsed, dict):
            events = parsed.get("events")
            if isinstance(events, list):
                events = [
                    e for e in events
                    if isinstance(e, dict)
                    and not _is_retrieval_noise(str(e.get("content") or ""))
                ]
                if last_updated:
                    events = [
                        e for e in events
                        if (e.get("timestamp") or "") > last_updated
                    ]
                parsed = {"events": events}
            files[child.name] = parsed
        elif child.name == "Notes.json" and isinstance(parsed, dict):
            notes = parsed.get("notes")
            if isinstance(notes, list):
                parsed = {
                    **parsed,
                    "notes": [
                        note for note in notes
                        if not isinstance(note, dict)
                        or not _is_retrieval_noise(str(note.get("text") or ""))
                    ],
                }
            files[child.name] = parsed
        elif child.name == "Read.json":
            # Rebuild from canonical sibling sources. Feeding an old bad digest
            # back to a small model makes its mistakes self-perpetuating.
            continue
        else:
            # Keep specialty leaves even when empty so Phase 2 can patch them
            # (e.g. Health.json) without inventing new filenames.
            files[child.name] = parsed
    return files


_PROTECTED_LEAF_NAMES = frozenset(
    {
        "Read.json",
        "Notes.json",
        "History.json",
        "Schedule.json",
        "Spotify.json",
        "Captures.json",
    }
)


def patchable_leaf_names(files: dict[str, Any]) -> list[str]:
    """Specialty JSON basenames Phase 2 may patch_json (not Notes/History/Read)."""
    return sorted(
        name
        for name in files
        if name.endswith(".json") and name not in _PROTECTED_LEAF_NAMES
    )


def assert_phase2_patch_path(folder: str, path: str) -> str:
    """Normalize and validate a Phase 2 patch_json path under folder.

    Returns the normalized relative path. Raises ValueError on reject.
    """
    from ainet.tools.paths import normalize_relpath

    folder_n = normalize_relpath(folder)
    path_n = normalize_relpath(path)
    if folder_n == ".":
        raise ValueError("Phase 2 cannot patch at the database root.")
    if path_n != folder_n and not path_n.startswith(folder_n + "/"):
        raise ValueError(f"Phase 2 may only write under {folder_n}/ (got {path_n}).")
    # Direct children only — no nested Music/Spotify.json from Preferences.
    rel = path_n[len(folder_n) + 1 :] if path_n.startswith(folder_n + "/") else ""
    if not rel or "/" in rel:
        raise ValueError(
            f"Phase 2 patch_json targets must be direct files in {folder_n}/ (got {path_n})."
        )
    if rel in _PROTECTED_LEAF_NAMES:
        raise ValueError(
            f"Phase 2 cannot patch_json {rel}. "
            "Use refresh_read for Read.json; leave Notes/History/Schedule alone."
        )
    if not rel.endswith(".json"):
        raise ValueError(f"Phase 2 patch_json requires a .json file (got {path_n}).")
    return path_n


def assert_phase2_read_path(folder: str, read_path: str) -> str:
    """Normalize and validate refresh_read path for the current Phase 2 folder."""
    from ainet.tools.paths import normalize_relpath

    folder_n = normalize_relpath(folder)
    path_n = normalize_relpath(read_path)
    expected = f"{folder_n}/Read.json"
    # Allow bare folder path; refresh_read resolves it.
    if path_n in {folder_n, expected}:
        return expected
    raise ValueError(f"Phase 2 refresh_read must target {expected} (got {path_n}).")


def build_read_refresh_folders(db: DatabaseTools) -> list[dict[str, Any]]:
    """Find all folders with stale Read.json and build per-folder payloads."""
    stale_result = db.list_stale_reads()
    stale_paths: list[str] = stale_result.get("paths") or []

    folders: list[dict[str, Any]] = []
    seen: set[str] = set()

    for read_path in stale_paths:
        folder = read_path.rsplit("/", 1)[0] if "/" in read_path else "."
        if folder in seen:
            continue
        seen.add(folder)

        # Pre-read Read.json to get last_updated cutoff for History filtering
        try:
            read_data = db.read_json(f"{folder}/Read.json")["data"]
        except (OSError, ValueError, KeyError):
            read_data = {}
        last_updated = read_data.get("last_updated") or "" if isinstance(read_data, dict) else ""

        files = _read_folder_jsons(db, folder, last_updated)
        if not files:
            continue

        history = files.get("History.json", {})
        new_events = history.get("events", []) if isinstance(history, dict) else []
        new_count = len(new_events) if isinstance(new_events, list) else 0

        notes = files.get("Notes.json", {})
        notes_list = notes.get("notes") if isinstance(notes, dict) else []
        if not isinstance(notes_list, list):
            notes_list = []

        patchable = patchable_leaf_names(files)
        folders.append({
            "folder": folder,
            "files": files,
            "patchable_leaves": patchable,
            "new_entries_since_last_refresh": new_count,
            "notes_count": len(notes_list),
        })

    return folders


def format_read_refresh_message(prompt: str, folder_payload: dict[str, Any]) -> str:
    """User message for a single folder's phase 2 compaction."""
    lines = [prompt.strip(), ""]
    folder = folder_payload["folder"]
    lines.append(f"folder: {folder}")
    patchable = folder_payload.get("patchable_leaves") or patchable_leaf_names(
        folder_payload.get("files") or {}
    )
    lines.append(
        "patchable_leaves (use patch_json on these paths only when facts belong): "
        + (", ".join(f"{folder}/{name}" for name in patchable) if patchable else "(none)")
    )
    lines.append("")
    lines.append("files:")
    lines.append(json.dumps(folder_payload["files"], ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("FINAL TRIAGE BEFORE refresh_read:")
    lines.append(
        "- DROP any source item whose only meaning is that Hayden asked to find, "
        "read, list, view, retrieve, check, or discuss stored information."
    )
    lines.append(
        "- DROP acknowledgments, instructions to the AI, database/file status, "
        "counts, and missing-data speculation."
    )
    lines.append(
        "- KEEP only concrete facts, tastes, experiences, interests, plans, or "
        "meaningful unanswered personal questions supported by evidence."
    )
    lines.append(
        "- IF a kept fact belongs in a listed patchable leaf -> patch_json that leaf "
        "first (set last_updated), then incorporate it into the Read digest."
    )
    lines.append(
        "- active_items are real ongoing actions or plans, not information-retrieval requests."
    )
    lines.append(
        "- If a list has no qualifying evidence, send an empty array. Then call refresh_read once."
    )
    return "\n".join(lines)
