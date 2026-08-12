"""Build the simplified filing payload for SOI test (phase 1 + phase 2)."""

from __future__ import annotations

import json
from typing import Any

from ainet.tools.ops import DatabaseTools

_CREATE_UNDER = ("Hayden", "Household", "Projects", "Questions")
_HIDDEN_CHILDREN: dict[str, set[str]] = {
    # Inbox is not a long-term filing target (file_note blocks it).
    "Hayden": {"Inbox", "School", "Work"},
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


def format_test_user_message(prompt: str, payload: dict[str, Any]) -> str:
    """User message: prompt text, then labeled sections."""
    lines = [prompt.strip(), ""]

    create_under = payload.get("create_under") or []
    lines.append("create_under: " + ", ".join(str(x) for x in create_under))
    lines.append("")

    folders = payload.get("folders") or {}
    lines.append("folders:")
    for domain in _CREATE_UNDER:
        kids = folders.get(domain) or []
        lines.append(f"  {domain}: " + (", ".join(kids) if kids else "(none)"))
    lines.append("")

    lines.append("changelog_entries:")
    import json

    entries = payload.get("changelog_entries") or []
    lines.append(json.dumps(entries, ensure_ascii=False, indent=2))

    inbox = payload.get("inbox_unfiled") or []
    if inbox:
        lines.append("")
        lines.append("inbox_unfiled:")
        lines.append(json.dumps(inbox, ensure_ascii=False, indent=2))

    return "\n".join(lines)


# ---- Phase 2: read refresh payload ----------------------------------------

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
            if isinstance(events, list) and last_updated:
                new_events = [
                    e for e in events
                    if isinstance(e, dict) and (e.get("timestamp") or "") > last_updated
                ]
                parsed = {"events": new_events}
            files[child.name] = parsed
        else:
            files[child.name] = parsed
    return files


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

        folders.append({
            "folder": folder,
            "files": files,
            "new_entries_since_last_refresh": new_count,
            "notes_count": len(notes_list),
        })

    return folders


def format_read_refresh_message(prompt: str, folder_payload: dict[str, Any]) -> str:
    """User message for a single folder's phase 2 compaction."""
    lines = [prompt.strip(), ""]
    lines.append(f"folder: {folder_payload['folder']}")
    lines.append(f"new_entries_since_last_refresh: {folder_payload['new_entries_since_last_refresh']}")
    lines.append(f"notes_count: {folder_payload['notes_count']}")
    lines.append("")
    lines.append("files:")
    lines.append(json.dumps(folder_payload["files"], ensure_ascii=False, indent=2))
    return "\n".join(lines)
