"""Append-only changelog helpers + SOI status marking (host-owned)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ainet.tools.fsutil import atomic_write_text
from ainet.tools.paths import DbPaths


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load(paths: DbPaths) -> dict[str, Any]:
    changelog_path = paths.resolve("Changelog.json", must_exist=True)
    with changelog_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "entries" not in data:
        raise ValueError("Changelog.json must be an object with an 'entries' array.")
    if not isinstance(data["entries"], list):
        raise ValueError("Changelog.json 'entries' must be a list.")
    return data


def _save(paths: DbPaths, data: dict[str, Any]) -> None:
    changelog_path = paths.resolve("Changelog.json", must_exist=True)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    atomic_write_text(changelog_path, text)


def append_entry(
    paths: DbPaths,
    *,
    action: str,
    path: str,
    summary: str,
    details: dict[str, Any] | None = None,
    actor: str = "ai",
    soi_status: str | None = None,
    entry_id: str | None = None,
) -> dict[str, Any]:
    data = _load(paths)
    entry: dict[str, Any] = {
        "id": entry_id or uuid.uuid4().hex[:16],
        "ts": _utc_now(),
        "actor": actor,
        "action": action,
        "path": path,
        "summary": summary,
    }
    if details:
        entry["details"] = details
    if soi_status is not None:
        entry["soi_status"] = soi_status
    elif actor == "oac" or action == "oac_turn":
        entry["soi_status"] = "pending"

    data["entries"].append(entry)
    _save(paths, data)
    return entry


def pending_oac_entries(paths: DbPaths) -> list[dict[str, Any]]:
    """OAC handoff rows still waiting for SOI (pending / missing status)."""
    data = _load(paths)
    pending: list[dict[str, Any]] = []
    for i, entry in enumerate(data["entries"]):
        if not isinstance(entry, dict):
            continue
        if entry.get("action") != "oac_turn" and entry.get("actor") != "oac":
            continue
        status = entry.get("soi_status", "pending")
        if status != "pending":
            continue
        pending.append({"index": i, **entry})
    return pending


def mark_soi_status(
    paths: DbPaths,
    *,
    entry_ids: list[str],
    status: str,
) -> int:
    """Mark changelog entries filed|discarded. Returns count updated."""
    if status not in {"pending", "filed", "discarded"}:
        raise ValueError(f"Invalid soi_status: {status}")
    wanted = set(entry_ids)
    if not wanted:
        return 0
    data = _load(paths)
    updated = 0
    for entry in data["entries"]:
        if not isinstance(entry, dict):
            continue
        eid = entry.get("id")
        if eid in wanted:
            entry["soi_status"] = status
            entry["soi_processed_at"] = _utc_now()
            updated += 1
    if updated:
        _save(paths, data)
    return updated


def ensure_changelog_file(root: Path) -> None:
    path = Path(root) / "Changelog.json"
    if path.exists():
        return
    atomic_write_text(path, json.dumps({"version": 1, "entries": []}, indent=2) + "\n")
