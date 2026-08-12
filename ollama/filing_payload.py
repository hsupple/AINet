"""Build the simplified filing payload for SOI test."""

from __future__ import annotations

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
