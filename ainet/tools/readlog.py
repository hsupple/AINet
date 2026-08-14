"""Per-folder Read.json freshness + size limits (hot index only)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from ainet.tools.fsutil import atomic_write_text
from ainet.tools.paths import DbPaths, normalize_relpath

# Cap retained consumed entries so Reads stay lean.
_MAX_CONSUMED = 40

# Hard practical caps — Read is a digest/index, never a dump.
READ_LIMITS: dict[str, int] = {
    "summary_chars": 400,
    "state_chars": 160,
    "item_chars": 180,
    "important_context": 12,
    "active_items": 10,
    "recent_changes": 8,
    "known_facts": 12,
    "uncertainties": 8,
    "log_summary_chars": 160,
    "read_changelog_pending": 60,
    "max_bytes": 12288,
}

ARRAY_FIELDS = (
    "important_context",
    "active_items",
    "recent_changes",
    "known_facts",
    "uncertainties",
)

SKIP_STALE_BASENAMES = frozenset(
    {
        "read.json",
        "changelog.json",
        "folderrules.json",
        "rules.txt",
        "calendar.json",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def is_read_json_path(path: str) -> bool:
    return Path(normalize_relpath(path)).name.casefold() == "read.json"


def should_skip_stale_mark(path: str) -> bool:
    """Mutations that must not auto-bump a folder Read."""
    norm = normalize_relpath(path)
    parts = PurePosixPath(norm).parts
    if not parts:
        return True
    if parts[0].casefold() in {"runtime", "chats"}:
        return True
    name = parts[-1].casefold()
    if name in SKIP_STALE_BASENAMES:
        return True
    return False


def find_nearest_read_path(paths: DbPaths, path: str) -> str | None:
    """Walk up from path's folder looking for Read.json. Returns db-relative path."""
    norm = normalize_relpath(path)
    parts = list(PurePosixPath(norm).parts)
    if not parts:
        return None
    # Start at containing folder when path is a file; at the folder itself when it is a dir.
    abs_path = paths.resolve(norm)
    if abs_path.exists() and abs_path.is_dir():
        start_parts = parts
    elif parts[-1].endswith(".json") or "." in parts[-1]:
        start_parts = parts[:-1]
    else:
        start_parts = parts

    for i in range(len(start_parts), 0, -1):
        folder = "/".join(start_parts[:i])
        candidate = f"{folder}/Read.json"
        if paths.resolve(candidate).is_file():
            return candidate
    # Root-level Read.json (rare)
    if paths.resolve("Read.json").is_file():
        return "Read.json"
    return None


def ensure_read_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure needs_update + read_changelog exist without wiping content."""
    if "needs_update" not in data:
        data["needs_update"] = False
    if "read_changelog" not in data or not isinstance(data.get("read_changelog"), list):
        data["read_changelog"] = []
    return data


def pending_changelog_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    entries = data.get("read_changelog") or []
    if not isinstance(entries, list):
        return []
    out: list[dict[str, Any]] = []
    for e in entries:
        if isinstance(e, dict) and e.get("status", "pending") == "pending":
            out.append(e)
    return out


def read_needs_refresh(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("needs_update") is True:
        return True
    return bool(pending_changelog_entries(data))


def _clip_str(value: Any, limit: int) -> tuple[str, bool]:
    text = value if isinstance(value, str) else ("" if value is None else str(value))
    if len(text) <= limit:
        return text, False
    return text[: max(0, limit - 1)].rstrip() + "…", True


def _trim_list_items(items: list[Any], *, max_items: int, item_chars: int) -> tuple[list[Any], list[str]]:
    notes: list[str] = []
    out: list[Any] = []
    if len(items) > max_items:
        notes.append(f"truncated list to {max_items} (had {len(items)})")
        items = items[:max_items]
    for item in items:
        if isinstance(item, str):
            clipped, changed = _clip_str(item, item_chars)
            out.append(clipped)
            if changed:
                notes.append("clipped long string item")
        elif isinstance(item, dict):
            copy = dict(item)
            if "text" in copy:
                clipped, changed = _clip_str(copy.get("text"), item_chars)
                copy["text"] = clipped
                if changed:
                    notes.append("clipped long fact text")
            out.append(copy)
        else:
            out.append(item)
    return out, notes


def enforce_read_limits(
    data: dict[str, Any],
    *,
    auto_trim: bool = True,
    note_trim_in_log: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """Keep Read.json a short hot index. Auto-trim by default; raise if still oversized."""
    ensure_read_fields(data)
    notes: list[str] = []
    limits = READ_LIMITS

    if "summary" in data or data.get("summary") is not None:
        clipped, changed = _clip_str(data.get("summary", ""), limits["summary_chars"])
        data["summary"] = clipped
        if changed:
            notes.append(f"summary clipped to {limits['summary_chars']} chars")

    if "state" in data or data.get("state") is not None:
        clipped, changed = _clip_str(data.get("state", ""), limits["state_chars"])
        data["state"] = clipped
        if changed:
            notes.append(f"state clipped to {limits['state_chars']} chars")

    for field in ARRAY_FIELDS:
        raw = data.get(field)
        if raw is None:
            data[field] = []
            continue
        if not isinstance(raw, list):
            raise ValueError(f"Read.json field '{field}' must be an array")
        trimmed, field_notes = _trim_list_items(
            raw,
            max_items=limits[field],
            item_chars=limits["item_chars"],
        )
        data[field] = trimmed
        for n in field_notes:
            notes.append(f"{field}: {n}")

    # Keep read_changelog lean (pending cap + truncate summaries)
    log = data.get("read_changelog")
    if isinstance(log, list):
        pending = [e for e in log if isinstance(e, dict) and e.get("status", "pending") == "pending"]
        consumed = [e for e in log if isinstance(e, dict) and e.get("status") == "consumed"]
        other = [e for e in log if not isinstance(e, dict)]
        if len(pending) > limits["read_changelog_pending"]:
            drop = len(pending) - limits["read_changelog_pending"]
            pending = pending[-limits["read_changelog_pending"] :]
            notes.append(f"read_changelog: dropped {drop} oldest pending entries")
        for e in pending + consumed:
            if isinstance(e, dict) and "summary" in e:
                clipped, changed = _clip_str(e.get("summary"), limits["log_summary_chars"])
                e["summary"] = clipped
                if changed:
                    notes.append("read_changelog: clipped entry summary")
        data["read_changelog"] = pending + consumed[-_MAX_CONSUMED:] + other

    if notes and note_trim_in_log:
        # Record trim as already-consumed so it does not re-trigger refresh.
        data["read_changelog"].append(
            {
                "id": uuid.uuid4().hex[:12],
                "at": _utc_now(),
                "summary": "auto-trimmed oversized Read fields: " + "; ".join(dict.fromkeys(notes))[:200],
                "source_path": "",
                "status": "consumed",
                "consumed_at": _utc_now(),
            }
        )
        # Re-cap consumed after note
        pending = [e for e in data["read_changelog"] if isinstance(e, dict) and e.get("status") == "pending"]
        consumed = [e for e in data["read_changelog"] if isinstance(e, dict) and e.get("status") == "consumed"]
        other = [e for e in data["read_changelog"] if not isinstance(e, dict)]
        data["read_changelog"] = pending + consumed[-_MAX_CONSUMED:] + other

    encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
    if len(encoded) > limits["max_bytes"]:
        if not auto_trim:
            raise ValueError(
                f"Read.json exceeds max_bytes ({limits['max_bytes']}); "
                "compress into leaf files / History and keep only pointers"
            )
        # Aggressive fallback: drop known_facts/uncertainties bodies to pointers-only hint
        for field in ("known_facts", "uncertainties", "recent_changes"):
            if isinstance(data.get(field), list) and data[field]:
                data[field] = data[field][: max(1, limits[field] // 2)]
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        if len(encoded) > limits["max_bytes"]:
            raise ValueError(
                f"Read.json still exceeds max_bytes ({limits['max_bytes']}) after trim; "
                "move detail into sibling leaf files or History and leave path pointers"
            )
        notes.append("aggressive size trim applied")

    if not auto_trim and notes:
        raise ValueError("Read.json exceeds size limits: " + "; ".join(notes))

    return data, list(dict.fromkeys(notes))


def prepare_read_payload(data: Any) -> tuple[dict[str, Any], list[str]]:
    """Validate + trim a Read.json object before write."""
    if not isinstance(data, dict):
        raise ValueError("Read.json must be a JSON object")
    ensure_read_fields(data)
    return enforce_read_limits(data, auto_trim=True, note_trim_in_log=True)


def append_read_log_entry(
    data: dict[str, Any],
    *,
    summary: str,
    source_path: str = "",
    entry_id: str | None = None,
) -> dict[str, Any]:
    ensure_read_fields(data)
    clipped, _ = _clip_str(
        (summary or "content changed").strip() or "content changed",
        READ_LIMITS["log_summary_chars"],
    )
    entry = {
        "id": entry_id or uuid.uuid4().hex[:12],
        "at": _utc_now(),
        "summary": clipped,
        "source_path": source_path or "",
        "status": "pending",
    }
    data["read_changelog"].append(entry)
    # Cap pending growth
    pending = [e for e in data["read_changelog"] if isinstance(e, dict) and e.get("status", "pending") == "pending"]
    consumed = [e for e in data["read_changelog"] if isinstance(e, dict) and e.get("status") == "consumed"]
    other = [e for e in data["read_changelog"] if not isinstance(e, dict)]
    if len(pending) > READ_LIMITS["read_changelog_pending"]:
        pending = pending[-READ_LIMITS["read_changelog_pending"] :]
    data["read_changelog"] = pending + consumed[-_MAX_CONSUMED:] + other
    data["needs_update"] = True
    return entry


def consume_read_log(data: dict[str, Any]) -> int:
    """Mark pending changelog entries consumed; set needs_update=false. Returns count."""
    ensure_read_fields(data)
    count = 0
    for e in data["read_changelog"]:
        if isinstance(e, dict) and e.get("status", "pending") == "pending":
            e["status"] = "consumed"
            e["consumed_at"] = _utc_now()
            count += 1
    data["needs_update"] = False
    # Trim old consumed entries (keep pending + newest consumed)
    pending = [e for e in data["read_changelog"] if isinstance(e, dict) and e.get("status") == "pending"]
    consumed = [
        e for e in data["read_changelog"] if isinstance(e, dict) and e.get("status") == "consumed"
    ]
    other = [e for e in data["read_changelog"] if not isinstance(e, dict)]
    data["read_changelog"] = pending + consumed[-_MAX_CONSUMED:] + other
    return count


def load_read_doc(paths: DbPaths, read_path: str) -> dict[str, Any]:
    target = paths.resolve(read_path, must_exist=True)
    with target.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Read.json must be a JSON object: {read_path}")
    return ensure_read_fields(data)


def save_read_doc(paths: DbPaths, read_path: str, data: dict[str, Any]) -> None:
    """Persist Read metadata writes (stale/refresh) without re-running content trim notes."""
    ensure_read_fields(data)
    # Still enforce changelog caps only
    pending = [e for e in data["read_changelog"] if isinstance(e, dict) and e.get("status", "pending") == "pending"]
    consumed = [e for e in data["read_changelog"] if isinstance(e, dict) and e.get("status") == "consumed"]
    other = [e for e in data["read_changelog"] if not isinstance(e, dict)]
    if len(pending) > READ_LIMITS["read_changelog_pending"]:
        pending = pending[-READ_LIMITS["read_changelog_pending"] :]
    data["read_changelog"] = pending + consumed[-_MAX_CONSUMED:] + other
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if len(text.encode("utf-8")) > READ_LIMITS["max_bytes"] * 2:
        # Metadata-only saves shouldn't blow up; drop oldest consumed
        data["read_changelog"] = pending + consumed[-10:] + other
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    atomic_write_text(paths.resolve(read_path), text)


def migrate_read_file(path: Path) -> bool:
    """Add needs_update/read_changelog to an on-disk Read.json. Returns True if changed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(data, dict):
        return False
    before = json.dumps(data, sort_keys=True)
    ensure_read_fields(data)
    after = json.dumps(data, sort_keys=True)
    if before == after:
        return False
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def migrate_all_reads(db_root: Path | str) -> list[str]:
    root = Path(db_root)
    updated: list[str] = []
    for path in sorted(root.rglob("Read.json")):
        if "runtime" in path.parts or "Chats" in path.parts:
            continue
        if migrate_read_file(path):
            try:
                updated.append(path.relative_to(root).as_posix())
            except ValueError:
                updated.append(str(path))
    return updated
