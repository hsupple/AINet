"""Append-only changelog helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ainet.tools.fsutil import atomic_write_text
from ainet.tools.paths import DbPaths


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_entry(
    paths: DbPaths,
    *,
    action: str,
    path: str,
    summary: str,
    details: dict[str, Any] | None = None,
    actor: str = "ai",
) -> dict[str, Any]:
    changelog_path = paths.resolve("Changelog.json", must_exist=True)
    with changelog_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict) or "entries" not in data:
        raise ValueError("Changelog.json must be an object with an 'entries' array.")

    entry: dict[str, Any] = {
        "ts": _utc_now(),
        "actor": actor,
        "action": action,
        "path": path,
        "summary": summary,
    }
    if details:
        entry["details"] = details

    entries = data.setdefault("entries", [])
    if not isinstance(entries, list):
        raise ValueError("Changelog.json 'entries' must be a list.")
    entries.append(entry)

    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    atomic_write_text(changelog_path, text)
    return entry
