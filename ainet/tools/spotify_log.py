"""Host-owned Spotify tool log under Hayden/Preferences/Music/Spotify.json.

AI may read this file. Only the host appends — never AI write tools.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ainet.tools.fsutil import atomic_write_text
from ainet.tools.ops import DatabaseTools
from ainet.tools.paths import DbPaths

LOG_PATH = "Hayden/Preferences/Music/Spotify.json"
_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def empty_log() -> dict[str, Any]:
    return {
        "description": (
            "Host log of every Spotify tool call. AI may read; only the host appends."
        ),
        "last_updated": "",
        "entries": [],
    }


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def ensure_log_file(root: Path) -> Path:
    path = Path(root) / "Hayden" / "Preferences" / "Music" / "Spotify.json"
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(empty_log(), indent=2, ensure_ascii=False) + "\n")
    return path


def _load(paths: DbPaths) -> dict[str, Any]:
    ensure_log_file(paths.root)
    target = paths.resolve(LOG_PATH, must_exist=True)
    with target.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        data = empty_log()
    entries = data.get("entries")
    if not isinstance(entries, list):
        data["entries"] = []
    return data


def _save(paths: DbPaths, data: dict[str, Any]) -> None:
    target = paths.resolve(LOG_PATH)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    atomic_write_text(target, text)


def append_spotify_use(
    db: DatabaseTools,
    *,
    ask: str = "",
    args: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    session_id: str = "",
) -> dict[str, Any]:
    """Append one Spotify tool invocation. Never raises to the chat path."""
    entry = {
        "id": uuid.uuid4().hex[:16],
        "ts": _utc_now(),
        "ask": str(ask or "").strip(),
        "session_id": str(session_id or "").strip(),
        "tool": "spotify",
        "args": _json_safe(args or {}),
        "result": _json_safe(result if isinstance(result, dict) else {"result": result}),
    }
    try:
        with _LOCK:
            data = _load(db.paths)
            entries = data["entries"]
            entries.append(entry)
            data["entries"] = entries
            data["last_updated"] = entry["ts"]
            _save(db.paths, data)
    except Exception:
        return {"ok": False, "logged": False}
    return {"ok": True, "logged": True, "id": entry["id"], "path": LOG_PATH}
