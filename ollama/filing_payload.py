"""Build the filing payload for SOI (log_item dests + existing labels)."""

from __future__ import annotations

import json
from typing import Any

from ainet.logstore import dest_names, list_existing_labels
from ainet.tools.ops import DatabaseTools
from ainet.tools.project import list_projects


def build_test_filing_payload(
    db: DatabaseTools,
    batch_changelog: list[dict[str, Any]],
    batch_inbox: list[dict[str, Any]] | None = None,
    *,
    entry_for_soi,
    inbox_for_soi=None,
) -> dict[str, Any]:
    _ = batch_inbox, inbox_for_soi
    projects = list_projects(db).get("projects") or []
    return {
        "dests": dest_names(),
        "projects": [str(p.get("name") or "") for p in projects if isinstance(p, dict)],
        "labels": list_existing_labels(db),
        "changelog_entries": [entry_for_soi(e) for e in batch_changelog],
    }


def _group_changelog(entries: list[Any]) -> list[dict[str, Any]]:
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
    lines: list[str] = []
    header = (prompt or "").strip()
    if header:
        lines.append(header)
        lines.append("")

    dests = payload.get("dests") or []
    lines.append("dests: " + ", ".join(str(x) for x in dests))
    projects = payload.get("projects") or []
    lines.append("projects: " + (", ".join(str(x) for x in projects) if projects else "(none)"))
    lines.append("")

    labels = payload.get("labels") or {}
    lines.append("existing keys (reuse when the same person, trait, or topic):")
    if isinstance(labels, dict) and labels:
        for bucket, rows in labels.items():
            if rows:
                lines.append(f"  {bucket}: " + ", ".join(str(x) for x in rows))
    else:
        lines.append("  (none yet)")
    lines.append("")

    lines.append("changelog_entries (grouped by session, chronological within each):")
    entries = payload.get("changelog_entries") or []
    lines.append(json.dumps(_group_changelog(entries), ensure_ascii=False, indent=2))
    return "\n".join(lines)
