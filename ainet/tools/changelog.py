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

    # Changelog.json is the pending OAC→SOI queue only.
    # Tool activity is permanent history on Masterlog.json.
    if actor == "oac" or action == "oac_turn":
        data["entries"].append(entry)
        _save(paths, data)
        return entry
    append_masterlog_entries(paths, [entry])
    return entry


def _load_masterlog(paths: DbPaths) -> dict[str, Any]:
    ensure_masterlog_file(paths.root)
    master_path = paths.resolve("Masterlog.json", must_exist=True)
    with master_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "entries" not in data:
        raise ValueError("Masterlog.json must be an object with an 'entries' array.")
    if not isinstance(data["entries"], list):
        raise ValueError("Masterlog.json 'entries' must be a list.")
    return data


def _save_masterlog(paths: DbPaths, data: dict[str, Any]) -> None:
    master_path = paths.resolve("Masterlog.json", must_exist=True)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    atomic_write_text(master_path, text)


def append_masterlog_entries(paths: DbPaths, entries: list[dict[str, Any]]) -> int:
    """Append unique entries to Masterlog.json. Never deletes. Returns count added."""
    if not entries:
        return 0
    data = _load_masterlog(paths)
    existing = {
        str(e.get("id") or "")
        for e in data["entries"]
        if isinstance(e, dict) and e.get("id")
    }
    added = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        eid = str(entry.get("id") or "")
        if not eid:
            continue
        if eid in existing:
            for i, row in enumerate(data["entries"]):
                if isinstance(row, dict) and str(row.get("id") or "") == eid:
                    data["entries"][i] = entry
                    added += 1
                    break
            continue
        data["entries"].append(entry)
        existing.add(eid)
        added += 1
    if added:
        data["last_updated"] = _utc_now()
        _save_masterlog(paths, data)
    return added


def get_entry(paths: DbPaths, entry_id: str) -> dict[str, Any] | None:
    """Return a Changelog or Masterlog entry by id, or None."""
    wanted = str(entry_id or "").strip()
    if not wanted:
        return None
    data = _load(paths)
    for i, entry in enumerate(data["entries"]):
        if isinstance(entry, dict) and str(entry.get("id") or "") == wanted:
            return {"index": i, **entry}
    master = _load_masterlog(paths)
    for entry in master["entries"]:
        if isinstance(entry, dict) and str(entry.get("id") or "") == wanted:
            return {"index": -1, "archived": True, **entry}
    return None


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
    dest_by_id: dict[str, str] | None = None,
) -> int:
    """Mark entries filed|discarded, copy them to Masterlog, remove from Changelog queue.

    Masterlog is append-only and never deleted. Changelog is the pending oac_turn queue.
    """
    if status not in {"pending", "filed", "discarded"}:
        raise ValueError(f"Invalid soi_status: {status}")
    wanted = {str(x) for x in entry_ids if x}
    if not wanted:
        return 0
    dest_by_id = dest_by_id or {}
    data = _load(paths)
    now = _utc_now()

    if status == "pending":
        updated = 0
        for entry in data["entries"]:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("id") or "") in wanted:
                entry["soi_status"] = "pending"
                entry.pop("soi_processed_at", None)
                updated += 1
        if updated:
            _save(paths, data)
        return updated

    keep: list[Any] = []
    archived: list[dict[str, Any]] = []
    updated = 0
    for entry in data["entries"]:
        if not isinstance(entry, dict):
            keep.append(entry)
            continue
        eid = str(entry.get("id") or "")
        if eid not in wanted:
            keep.append(entry)
            continue
        row = dict(entry)
        row["soi_status"] = status
        row["soi_processed_at"] = now
        row["archived_at"] = now
        if eid in dest_by_id:
            row["filed_to"] = dest_by_id[eid]
        archived.append(row)
        updated += 1
    if updated:
        append_masterlog_entries(paths, archived)
        data["entries"] = keep
        _save(paths, data)
    return updated


def migrate_resolved_to_masterlog(paths: DbPaths) -> int:
    """Move already filed/discarded oac_turns from Changelog into Masterlog."""
    data = _load(paths)
    keep: list[Any] = []
    moved: list[dict[str, Any]] = []
    for entry in data["entries"]:
        if not isinstance(entry, dict):
            keep.append(entry)
            continue
        is_turn = entry.get("action") == "oac_turn" or entry.get("actor") == "oac"
        status = entry.get("soi_status")
        if is_turn and status not in {"filed", "discarded"}:
            keep.append(entry)
            continue
        row = dict(entry)
        row.setdefault("archived_at", row.get("soi_processed_at") or _utc_now())
        moved.append(row)
    if not moved:
        return 0
    append_masterlog_entries(paths, moved)
    data["entries"] = keep
    _save(paths, data)
    return len(moved)


def ensure_changelog_file(root: Path) -> None:
    path = Path(root) / "Changelog.json"
    if path.exists():
        return
    atomic_write_text(path, json.dumps({"version": 1, "entries": []}, indent=2) + "\n")


def ensure_masterlog_file(root: Path) -> None:
    path = Path(root) / "Masterlog.json"
    if not path.exists():
        atomic_write_text(
            path,
            json.dumps({"version": 1, "entries": [], "last_updated": ""}, indent=2) + "\n",
        )
