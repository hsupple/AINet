"""Keyed knowledge store.

Each subject is a JSON key whose value is an observation list:
  "Jake": [{"time": "...", "text": "..."}]

SOI log_item creates the key if missing and appends. Host computes count/strength on query.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

MAX_ENTRIES = 80
MAX_TEXT_CHARS = 240
MAX_QUERY_KEYS = 16
MAX_QUERY_ENTRIES = 24

HAYDEN_FILE = "hayden.json"
HAYDEN_ARRAYS = (
    "characteristics",
    "preferences",
    "habits",
    "values",
    "desires",
    "body",
    "psychology",
)

ROOT_FILES = (
    HAYDEN_FILE,
    "people.json",
    "questions.json",
    "household.json",
    "memories.json",
    "secrets.json",
)

RESERVED_KEYS = frozenset({"version", "name", "summary", "items"})
KEEP_META_KEYS = frozenset({"version", "name", "summary"})

# dest alias -> (path, map key). Empty map key means the document root is the map.
_DEST_MAP: dict[str, tuple[str, str]] = {
    "hayden": (HAYDEN_FILE, "characteristics"),
    "characteristics": (HAYDEN_FILE, "characteristics"),
    "characteristic": (HAYDEN_FILE, "characteristics"),
    "identity": (HAYDEN_FILE, "characteristics"),
    "personality": (HAYDEN_FILE, "characteristics"),
    "voice": (HAYDEN_FILE, "characteristics"),
    "sides": (HAYDEN_FILE, "characteristics"),
    "core": (HAYDEN_FILE, "characteristics"),
    "interests": (HAYDEN_FILE, "characteristics"),
    "interest": (HAYDEN_FILE, "characteristics"),
    "education": (HAYDEN_FILE, "characteristics"),
    "experience": (HAYDEN_FILE, "characteristics"),
    "software_skills": (HAYDEN_FILE, "characteristics"),
    "ai_experience": (HAYDEN_FILE, "characteristics"),
    "questioning_style": (HAYDEN_FILE, "characteristics"),
    "curiosity": (HAYDEN_FILE, "characteristics"),
    "preferences": (HAYDEN_FILE, "preferences"),
    "preference": (HAYDEN_FILE, "preferences"),
    "likes": (HAYDEN_FILE, "preferences"),
    "dislikes": (HAYDEN_FILE, "preferences"),
    "food": (HAYDEN_FILE, "preferences"),
    "media": (HAYDEN_FILE, "preferences"),
    "music": (HAYDEN_FILE, "preferences"),
    "habits": (HAYDEN_FILE, "habits"),
    "habit": (HAYDEN_FILE, "habits"),
    "routines": (HAYDEN_FILE, "habits"),
    "vices": (HAYDEN_FILE, "habits"),
    "values": (HAYDEN_FILE, "values"),
    "value": (HAYDEN_FILE, "values"),
    "principles": (HAYDEN_FILE, "values"),
    "desires": (HAYDEN_FILE, "desires"),
    "desire": (HAYDEN_FILE, "desires"),
    "wants": (HAYDEN_FILE, "desires"),
    "goals": (HAYDEN_FILE, "desires"),
    "body": (HAYDEN_FILE, "body"),
    "health": (HAYDEN_FILE, "body"),
    "psychology": (HAYDEN_FILE, "psychology"),
    "feelings": (HAYDEN_FILE, "psychology"),
    "anxiety": (HAYDEN_FILE, "psychology"),
    "coping": (HAYDEN_FILE, "psychology"),
    "triggers": (HAYDEN_FILE, "psychology"),
    "people": ("people.json", ""),
    "person": ("people.json", ""),
    "relationships": ("people.json", ""),
    "relationship": ("people.json", ""),
    "questions": ("questions.json", ""),
    "question": ("questions.json", ""),
    "science": ("questions.json", ""),
    "household": ("household.json", ""),
    "home": ("household.json", ""),
    "pantry": ("household.json", ""),
    "maintenance": ("household.json", ""),
    "memories": ("memories.json", ""),
    "memory": ("memories.json", ""),
    "history": ("memories.json", ""),
    "secrets": ("secrets.json", ""),
    "secret": ("secrets.json", ""),
    "private": ("secrets.json", ""),
}

_DISCARD = frozenset(
    {
        "discard",
        "drop",
        "ephemeral",
        "planner",
        "plan",
        "plans",
        "schedule",
        "todo",
        "todos",
        "planner.json",
    }
)
_FILE_HINT = (
    "Use query_db to look up stored facts by name, words, and dates. "
    "hayden.json — who Hayden is. people.json — people keyed by name. "
    "questions.json — topics. household.json — home. "
    "memories.json — life events. secrets.json — private. "
    "Projects/<Name>/project.json — named projects. "
    "Near-term schedule belongs on Calendar.json later, not a planner file."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def empty_hayden() -> dict[str, Any]:
    doc: dict[str, Any] = {"version": 1}
    for key in HAYDEN_ARRAYS:
        doc[key] = {}
    return doc


def empty_log() -> dict[str, Any]:
    return {}


def empty_project(name: str = "") -> dict[str, Any]:
    return {"version": 1, "name": name}


def knowledge_file_names() -> tuple[str, ...]:
    return ROOT_FILES


def dest_names() -> list[str]:
    return [
        "hayden",
        "preferences",
        "habits",
        "values",
        "desires",
        "body",
        "psychology",
        "people",
        "questions",
        "household",
        "memories",
        "secrets",
        "discard",
    ]


def _parse_ts(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        text = text + "T00:00:00+00:00"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_bound(value: str, *, end: bool = False) -> datetime | None:
    text = (value or "").strip()
    date_only = len(text) == 10 and text[4] == "-" and text[7] == "-"
    parsed = _parse_ts(text)
    if parsed is not None and date_only and end:
        return parsed.replace(hour=23, minute=59, second=59)
    return parsed


# --- decay -----------------------------------------------------------------
# Linear remaining = 1 - age/lifetime. Lifetime interpolates low→high with count.
# 1 mention uses low_days; entrenched_count+ mentions use high_days.
# Observations below floor are deleted (except secrets). Tune in Folderrules.json.

ENTRENCHED_COUNT = 5
DECAY_FLOOR = 0.05
_RETIRED_FILES = ("planner.json",)


@dataclass(frozen=True)
class DecayProfile:
    low_days: float
    high_days: float
    floor: float = DECAY_FLOOR
    delete: bool = True


_DEFAULT_DECAY: dict[str, DecayProfile] = {
    "characteristics": DecayProfile(31, 365),
    "values": DecayProfile(31, 365),
    "preferences": DecayProfile(21, 180),
    "habits": DecayProfile(14, 120),
    "desires": DecayProfile(14, 90),
    "body": DecayProfile(10, 60),
    "psychology": DecayProfile(14, 180),
    "household": DecayProfile(7, 45),
    "people": DecayProfile(21, 365),
    "questions": DecayProfile(14, 90),
    "memories": DecayProfile(60, 730),
    "secrets": DecayProfile(180, 730, delete=False),
    "project": DecayProfile(21, 180),
}


def _decay_config(db: Any | None) -> dict[str, Any]:
    try:
        raw = (db.permissions.rules.raw or {}).get("decay")  # type: ignore[union-attr]
    except Exception:
        raw = None
    return raw if isinstance(raw, dict) else {}


def _profile_key(section: str) -> str:
    text = str(section or "").strip()
    if text.casefold().startswith("project:"):
        return "project"
    if text.casefold() == "hayden":
        return "characteristics"
    return text


def decay_profile(section: str, db: Any | None = None) -> DecayProfile:
    key = _profile_key(section)
    base = _DEFAULT_DECAY.get(key) or DecayProfile(21, 180)
    cfg = _decay_config(db)
    floor = float(cfg.get("floor") if cfg.get("floor") is not None else base.floor)
    row = cfg.get("profiles") if isinstance(cfg.get("profiles"), dict) else {}
    override = row.get(key) if isinstance(row.get(key), dict) else {}
    low = float(override.get("low_days") if override.get("low_days") is not None else base.low_days)
    high = float(override.get("high_days") if override.get("high_days") is not None else base.high_days)
    if "floor" in override and override.get("floor") is not None:
        floor = float(override["floor"])
    delete = base.delete
    if "delete" in override:
        delete = bool(override["delete"])
    elif key == "secrets" and "delete" in cfg:
        delete = bool(cfg.get("delete"))
    return DecayProfile(
        low_days=max(1.0, low),
        high_days=max(low, high),
        floor=max(0.0, min(1.0, floor)),
        delete=delete,
    )


def resistance_days(profile: DecayProfile, count: int, db: Any | None = None) -> float:
    n = ENTRENCHED_COUNT
    cfg = _decay_config(db)
    if cfg.get("entrenched_count") is not None:
        try:
            n = max(2, int(cfg["entrenched_count"]))
        except (TypeError, ValueError):
            n = ENTRENCHED_COUNT
    n_obs = max(1, int(count or 0))
    if n_obs <= 1:
        t = 0.0
    else:
        t = min(1.0, (n_obs - 1) / (n - 1))
    return profile.low_days + t * (profile.high_days - profile.low_days)


def _age_days(when: str, now: datetime) -> float:
    parsed = _parse_ts(when)
    if parsed is None:
        return 0.0
    return max(0.0, (now - parsed).total_seconds() / 86400.0)


def observation_remaining(
    entry: Any,
    count: int,
    profile: DecayProfile,
    now: datetime,
    db: Any | None = None,
) -> float:
    life = resistance_days(profile, count, db)
    if life <= 0:
        return 0.0
    remaining = 1.0 - (_age_days(_entry_time(entry), now) / life)
    return max(0.0, remaining)


def compute_strength(
    times: list[str],
    count: int,
    *,
    now: datetime | None = None,
    section: str = "",
    db: Any | None = None,
    entries: list[Any] | None = None,
) -> float:
    stamp = now or datetime.now(timezone.utc)
    profile = decay_profile(section, db) if section else DecayProfile(21, 180)
    if entries is not None:
        n = len(entries)
        return round(sum(observation_remaining(e, n, profile, stamp, db) for e in entries), 3)
    fake = [{"time": t} for t in times]
    n = int(count or len(fake))
    return round(sum(observation_remaining(e, n, profile, stamp, db) for e in fake), 3)


def prune_mapping(
    mapping: dict[str, Any],
    profile: DecayProfile,
    now: datetime,
    db: Any | None = None,
) -> tuple[int, int]:
    """Drop observations/keys below the floor. Returns (dropped_keys, dropped_entries)."""
    dropped_keys = 0
    dropped_entries = 0
    if not profile.delete:
        return 0, 0
    for key in list(mapping.keys()):
        if _is_reserved(str(key)):
            continue
        raw = mapping.get(key)
        if not isinstance(raw, list):
            continue
        count = len(raw)
        kept: list[Any] = []
        for entry in raw:
            if observation_remaining(entry, count, profile, now, db) >= profile.floor:
                kept.append(entry)
            else:
                dropped_entries += 1
        strength = sum(observation_remaining(e, count, profile, now, db) for e in kept)
        if not kept or strength < profile.floor:
            del mapping[key]
            dropped_keys += 1
            dropped_entries += len(kept)
        elif len(kept) != count:
            mapping[key] = kept
    return dropped_keys, dropped_entries


def _prune_document(doc: dict[str, Any], path: str, now: datetime, db: Any | None = None) -> tuple[int, int]:
    name = Path(str(path).replace("\\", "/")).name.casefold()
    dropped_keys = 0
    dropped_entries = 0
    if name == HAYDEN_FILE:
        for section in HAYDEN_ARRAYS:
            mapping = _coerce_map(doc.get(section))
            dk, de = prune_mapping(mapping, decay_profile(section, db), now, db)
            doc[section] = mapping
            dropped_keys += dk
            dropped_entries += de
        return dropped_keys, dropped_entries
    section = "project" if name == "project.json" else Path(path).stem
    mapping = _map_from_doc(doc, "")
    dk, de = prune_mapping(mapping, decay_profile(section, db), now, db)
    dropped_keys += dk
    dropped_entries += de
    return dropped_keys, dropped_entries


def decay_knowledge(db: Any, *, now: datetime | None = None) -> dict[str, Any]:
    """Persist prune of dead observations across knowledge files. Host-only."""
    stamp = now or datetime.now(timezone.utc)
    dropped_keys = 0
    dropped_entries = 0
    files: list[str] = []
    root = db.paths.root
    leftover = root / "planner.json"
    if leftover.is_file():
        leftover.unlink()
        files.append("planner.json")
    for path in [HAYDEN_FILE, *[n for n in ROOT_FILES if n != HAYDEN_FILE]]:
        if not db.paths.resolve(path).exists():
            continue
        doc = _load_doc(db, path)
        dk, de = _prune_document(doc, path, stamp, db)
        if dk or de:
            db.write_json(path, doc, summary=f"decay prune {path}")
            dropped_keys += dk
            dropped_entries += de
            files.append(path)
    for project in _project_names(db):
        path = f"Projects/{project}/project.json"
        if not db.paths.resolve(path).exists():
            continue
        doc = _load_doc(db, path)
        dk, de = _prune_document(doc, path, stamp, db)
        if dk or de:
            db.write_json(path, doc, summary=f"decay prune {path}")
            dropped_keys += dk
            dropped_entries += de
            files.append(path)
    return {
        "ok": True,
        "dropped_keys": dropped_keys,
        "dropped_entries": dropped_entries,
        "files": files,
    }


def normalize_label(label: str) -> str:
    return " ".join((label or "").strip().split())


def _label_key(label: str) -> str:
    return normalize_label(label).casefold()


def _is_reserved(key: str) -> bool:
    return str(key or "").casefold() in RESERVED_KEYS


def _entry_time(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("time") or "")
    return ""


def _entry_text(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("text") or entry.get("reason") or "")
    return str(entry or "")


def _times_from_entries(entries: list[Any]) -> list[str]:
    return [_entry_time(e) for e in entries if _entry_time(e)]


def _summarize_key(
    name: str,
    entries: list[Any],
    *,
    section: str = "",
    db: Any | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    stamp = now or datetime.now(timezone.utc)
    times = _times_from_entries(entries)
    count = len(entries) if isinstance(entries, list) else 0
    profile = decay_profile(section, db)
    strength = compute_strength([], count, now=stamp, section=section, db=db, entries=entries)
    return {
        "name": name,
        "count": count,
        "strength": strength,
        "resistance_days": round(resistance_days(profile, count, db), 1),
        "first": times[0] if times else "",
        "last": times[-1] if times else "",
    }


def _merge_legacy_list(out: dict[str, Any], rows: list[Any]) -> None:
    for row in rows:
        if not isinstance(row, dict):
            continue
        name, entries = _legacy_item_to_entries(row)
        if not name:
            continue
        existing = out.setdefault(name, [])
        if isinstance(existing, list):
            existing.extend(entries)


def _coerce_map(value: Any) -> dict[str, Any]:
    """Accept a name→entries map, or migrate legacy items[] of {label, times, reasons}."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, raw in value.items():
            if str(key).casefold() == "items" and isinstance(raw, list):
                _merge_legacy_list(out, raw)
                continue
            if _is_reserved(str(key)):
                continue
            if isinstance(raw, list):
                out[str(key)] = list(raw)
            elif isinstance(raw, dict) and ("label" in raw or "name" in raw):
                name, entries = _legacy_item_to_entries(raw)
                if name:
                    existing = out.setdefault(name, [])
                    if isinstance(existing, list):
                        existing.extend(entries)
        return out
    if isinstance(value, list):
        out: dict[str, Any] = {}
        _merge_legacy_list(out, value)
        return out
    return {}


def _legacy_item_to_entries(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    name = normalize_label(str(row.get("label") or row.get("name") or ""))
    if not name:
        return "", []
    times = row.get("times") if isinstance(row.get("times"), list) else []
    reasons = row.get("reasons") if isinstance(row.get("reasons"), list) else []
    entries: list[dict[str, Any]] = []
    n = max(len(times), len(reasons), 1 if (times or reasons) else 0)
    for i in range(n):
        when = str(times[i]) if i < len(times) else (str(times[-1]) if times else "")
        text = str(reasons[i]) if i < len(reasons) else (str(reasons[-1]) if reasons else "")
        if when or text:
            entries.append({"time": when, "text": text})
    return name, entries


def _find_key(mapping: dict[str, Any], label: str) -> str | None:
    want = _label_key(label)
    if not want:
        return None
    for key in mapping:
        if _is_reserved(str(key)):
            continue
        if _label_key(str(key)) == want:
            return str(key)
    return None


def append_observation(
    mapping: dict[str, Any],
    *,
    label: str,
    text: str,
    when: str,
    extra: dict[str, Any] | None = None,
) -> tuple[str, list[Any], bool]:
    """Create mapping[label] = [] if needed, then append one observation. Returns (key, list, created)."""
    clean = normalize_label(label)
    if not clean:
        raise ValueError("label is required")
    if _is_reserved(clean):
        raise ValueError(f"label {clean!r} is reserved")
    reason_text = " ".join((text or "").strip().split())[:MAX_TEXT_CHARS]
    if not reason_text:
        raise ValueError("reason is required")
    created = False
    key = _find_key(mapping, clean)
    if key is None:
        key = clean
        mapping[key] = []
        created = True
    bucket = mapping.get(key)
    if not isinstance(bucket, list):
        bucket = []
        mapping[key] = bucket
    last = bucket[-1] if bucket else None
    if (
        isinstance(last, dict)
        and _entry_text(last).casefold() == reason_text.casefold()
        and _entry_time(last) == when
    ):
        return key, bucket, created
    row: dict[str, Any] = {"time": when, "text": reason_text}
    if extra:
        for field, value in extra.items():
            if value is None or value == "":
                continue
            row[field] = value
    bucket.append(row)
    mapping[key] = bucket[-MAX_ENTRIES:]
    return key, mapping[key], created


def empty_doc_for(path: str, *, project_name: str = "") -> dict[str, Any]:
    name = Path(str(path).replace("\\", "/")).name.casefold()
    if name == HAYDEN_FILE:
        return empty_hayden()
    if name == "project.json":
        return empty_project(project_name)
    return empty_log()


def ensure_knowledge_files(root: Path) -> None:
    from ainet.tools.fsutil import atomic_write_text
    import json

    root.mkdir(parents=True, exist_ok=True)
    for name in ROOT_FILES:
        path = root / name
        if path.exists():
            continue
        payload = empty_hayden() if name == HAYDEN_FILE else empty_log()
        atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    projects = root / "Projects"
    projects.mkdir(parents=True, exist_ok=True)
    for name in _RETIRED_FILES:
        leftover = root / name
        if leftover.is_file():
            leftover.unlink()


def _load_doc(db: Any, path: str) -> dict[str, Any]:
    if not db.paths.resolve(path).exists():
        name = PurePosixPath(path).name
        project = ""
        parts = PurePosixPath(path).parts
        if len(parts) >= 2 and parts[0].casefold() == "projects":
            project = parts[1]
        doc = empty_doc_for(path, project_name=project)
        db.write_json(path, doc, create=True, summary=f"Seed {name}")
        return doc
    data = db.read_json(path)["data"]
    if isinstance(data, dict):
        return data
    return empty_doc_for(path)


def _project_names(db: Any) -> list[str]:
    root = db.paths.root / "Projects"
    if not root.is_dir():
        return []
    names: list[str] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if child.is_dir() and not child.name.startswith("."):
            names.append(child.name)
    return names


def resolve_dest(db: Any, dest: str) -> tuple[str, str] | str | None:
    """Return (path, map_key), 'discard', or None. Empty map_key = document root."""
    raw = (dest or "").strip()
    if not raw:
        return None
    key = raw.replace("\\", "/").strip("/").casefold()
    if key in _DISCARD:
        return "discard"
    if key in _DEST_MAP:
        return _DEST_MAP[key]

    if key.endswith(".json") or "/" in key:
        path = raw.replace("\\", "/").strip("/")
        name = Path(path).name.casefold()
        if name == HAYDEN_FILE:
            return (HAYDEN_FILE, "characteristics")
        if name in {n.casefold() for n in ROOT_FILES}:
            return (Path(path).name, "")
        if name == "project.json":
            return (path, "")

    for project in _project_names(db):
        if project.casefold() == key or key == f"projects/{project.casefold()}":
            return (f"Projects/{project}/project.json", "")
    return None


def _map_from_doc(doc: dict[str, Any], map_key: str) -> dict[str, Any]:
    """Return the live map dict (same object stored on doc) so appends persist."""
    if map_key:
        coerced = _coerce_map(doc.get(map_key))
        doc[map_key] = coerced
        return coerced
    coerced = _coerce_map(doc)
    reserved = {k: doc[k] for k in list(doc) if str(k).casefold() in KEEP_META_KEYS}
    doc.clear()
    doc.update(reserved)
    doc.update(coerced)
    return doc


def list_existing_labels(db: Any) -> dict[str, list[str]]:
    """Name index so SOI reuses existing keys instead of inventing synonyms."""
    out: dict[str, list[str]] = {}

    def add(bucket: str, mapping: dict[str, Any]) -> None:
        rows: list[tuple[str, int, str]] = []
        for name, entries in mapping.items():
            if _is_reserved(name) or not isinstance(entries, list):
                continue
            times = _times_from_entries(entries)
            last = times[-1] if times else ""
            rows.append((str(name), len(entries), last))
        rows.sort(key=lambda r: r[2], reverse=True)
        labeled = [f"{name} ({count})" if count else name for name, count, _last in rows[:40]]
        if labeled:
            out[bucket] = labeled

    hayden = _load_doc(db, HAYDEN_FILE)
    for key in HAYDEN_ARRAYS:
        add(key, _coerce_map(hayden.get(key)))
    for name in ROOT_FILES:
        if name == HAYDEN_FILE:
            continue
        doc = _load_doc(db, name)
        add(Path(name).stem, _coerce_map(doc))
    for project in _project_names(db):
        path = f"Projects/{project}/project.json"
        if db.paths.resolve(path).exists():
            doc = _load_doc(db, path)
            add(f"project:{project}", _coerce_map(doc))
    return out


def log_item(
    db: Any,
    *,
    dest: str,
    label: str,
    reason: str,
    entry_id: str = "",
    entry_ids: list[Any] | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    """SOI filing: create key if needed, append one observation."""
    from ainet.tools import changelog

    ids: list[str] = []
    for raw in list(entry_ids or []) + [entry_id]:
        eid = str(raw or "").strip()
        if eid and eid not in ids:
            ids.append(eid)
    dest_raw = str(dest or "").strip()
    if not dest_raw:
        return {"ok": False, "error": "dest is required"}

    resolved = resolve_dest(db, dest_raw)
    if resolved == "discard":
        return {
            "ok": True,
            "action": "discard",
            "dest": dest_raw,
            "entry_ids": ids,
        }
    if not resolved:
        return {
            "ok": False,
            "error": (
                f"Unknown dest: {dest_raw!r}. Use one of: {', '.join(dest_names())} "
                "or a project name."
            ),
            "entry_ids": ids,
        }

    path, map_key = resolved
    label_clean = normalize_label(label)

    if dest_raw.lower() not in _DISCARD and not label_clean:
        return {
            "ok": False,
            "error": "label is required — person name, trait, or topic key",
            "entry_ids": ids,
        }
    if dest_raw.lower() not in _DISCARD and not str(reason or "").strip():
        return {"ok": False, "error": "reason is required — what to append", "entry_ids": ids}

    if ids:
        missing = [eid for eid in ids if not changelog.get_entry(db.paths, eid)]
        if len(missing) == len(ids):
            return {"ok": False, "error": f"Unknown entry id: {ids[0]}", "entry_ids": ids}

    when = utc_now()
    if ids:
        entry = changelog.get_entry(db.paths, ids[0])
        if entry and entry.get("ts"):
            when = str(entry.get("ts"))

    stamp = datetime.now(timezone.utc)
    doc = _load_doc(db, path)
    _prune_document(doc, path, stamp, db)
    mapping = _map_from_doc(doc, map_key)
    key, bucket, created = append_observation(
        mapping,
        label=label,
        text=reason,
        when=when,
    )
    if map_key:
        doc[map_key] = mapping
    written = db.write_json(
        path,
        doc,
        summary=summary or f"{'create' if created else 'append'} {key} → {path}",
    )
    section = map_key or Path(path).stem
    return {
        "ok": True,
        "action": "create" if created else "append",
        "dest": dest_raw,
        "path": path,
        "name": key,
        "created_key": created,
        "count": len(bucket),
        "strength": compute_strength([], len(bucket), now=stamp, section=section, db=db, entries=bucket),
        "resistance_days": round(resistance_days(decay_profile(section, db), len(bucket), db), 1),
        "entry_id": ids[0] if ids else "",
        "entry_ids": ids,
        "write": written,
    }


def _iter_query_targets(db: Any, dest: str, file: str) -> list[tuple[str, str, dict[str, Any]]]:
    """List (path, section, mapping) to search."""
    targets: list[tuple[str, str, dict[str, Any]]] = []
    dest_raw = (dest or file or "").strip()
    dest_key = dest_raw.replace("\\", "/").strip("/").casefold()
    if dest_key in {"hayden", "hayden.json"}:
        hayden = _load_doc(db, HAYDEN_FILE)
        for section in HAYDEN_ARRAYS:
            targets.append((HAYDEN_FILE, section, _coerce_map(hayden.get(section))))
        return targets
    if dest_raw:
        resolved = resolve_dest(db, dest_raw)
        if not resolved or resolved == "discard":
            return []
        path, map_key = resolved
        doc = _load_doc(db, path)
        mapping = _map_from_doc(doc, map_key)
        section = map_key or Path(path).stem
        targets.append((path, section, mapping))
        return targets

    hayden = _load_doc(db, HAYDEN_FILE)
    for section in HAYDEN_ARRAYS:
        targets.append((HAYDEN_FILE, section, _coerce_map(hayden.get(section))))
    for name in ROOT_FILES:
        if name == HAYDEN_FILE:
            continue
        if name == "secrets.json":
            continue
        doc = _load_doc(db, name)
        targets.append((name, Path(name).stem, _coerce_map(doc)))
    for project in _project_names(db):
        path = f"Projects/{project}/project.json"
        if db.paths.resolve(path).exists():
            doc = _load_doc(db, path)
            targets.append((path, f"project:{project}", _coerce_map(doc)))
    return targets


def _name_matches(key: str, name: str) -> bool:
    if not name:
        return True
    want = _label_key(name)
    have = _label_key(key)
    return have == want or want in have or have in want


def _words_match(name: str, entries: list[Any], q: str) -> list[Any]:
    tokens = [t for t in (q or "").casefold().split() if t]
    if not tokens:
        return list(entries)
    name_l = name.casefold()
    kept: list[Any] = []
    for entry in entries:
        blob = f"{name_l} {_entry_text(entry).casefold()}"
        if all(token in blob for token in tokens):
            kept.append(entry)
    if all(token in name_l for token in tokens):
        return list(entries) if not kept else kept
    return kept


def _date_in_range(when: str, after: datetime | None, before: datetime | None) -> bool:
    parsed = _parse_ts(when)
    if parsed is None:
        return not after and not before
    if after and parsed < after:
        return False
    if before and parsed > before:
        return False
    return True


def query_db(
    db: Any,
    *,
    dest: str = "",
    file: str = "",
    name: str = "",
    q: str = "",
    after: str = "",
    before: str = "",
    since_days: int | None = None,
    keys_only: bool = False,
    include_secrets: bool = False,
    limit: int = MAX_QUERY_KEYS,
) -> dict[str, Any]:
    """OAC retrieval: filter knowledge maps by name, words, and dates."""
    after_dt = _parse_bound(after, end=False)
    before_dt = _parse_bound(before, end=True)
    if since_days is not None:
        try:
            days = int(since_days)
        except (TypeError, ValueError):
            days = 0
        if days > 0:
            after_dt = datetime.now(timezone.utc) - timedelta(days=days)

    dest_raw = (dest or file or "").strip()
    if dest_raw.casefold() in {"secrets", "secrets.json"}:
        include_secrets = True
    targets = _iter_query_targets(db, dest_raw, "")
    if include_secrets and not dest_raw:
        doc = _load_doc(db, "secrets.json")
        targets.append(("secrets.json", "secrets", _coerce_map(doc)))

    stamp = datetime.now(timezone.utc)
    cap = max(1, min(int(limit or MAX_QUERY_KEYS), 40))
    matches: list[dict[str, Any]] = []
    for path, section, mapping in targets:
        profile = decay_profile(section, db)
        for key, raw_entries in mapping.items():
            if _is_reserved(str(key)) or not isinstance(raw_entries, list):
                continue
            if not _name_matches(str(key), name):
                continue
            count = len(raw_entries)
            live = list(raw_entries)
            if profile.delete:
                live = [
                    e
                    for e in raw_entries
                    if observation_remaining(e, count, profile, stamp, db) >= profile.floor
                ]
                live_strength = sum(
                    observation_remaining(e, count, profile, stamp, db) for e in live
                )
                if not live or live_strength < profile.floor:
                    continue
            entries = live
            if after_dt or before_dt:
                entries = [
                    e
                    for e in entries
                    if _date_in_range(_entry_time(e), after_dt, before_dt)
                ]
            if q:
                word_hits = _words_match(str(key), entries, q)
                tokens = [t for t in q.casefold().split() if t]
                name_hit = bool(tokens) and all(tok in str(key).casefold() for tok in tokens)
                if word_hits:
                    entries = word_hits
                elif name_hit:
                    pass
                else:
                    continue
            if (after_dt or before_dt) and not entries:
                continue
            summary = _summarize_key(
                str(key),
                live if not (q or after_dt or before_dt) else entries,
                section=section,
                db=db,
                now=stamp,
            )
            row: dict[str, Any] = {
                "file": path,
                "section": section,
                **summary,
            }
            if not keys_only:
                row["entries"] = entries[-MAX_QUERY_ENTRIES:]
            matches.append(row)

    matches.sort(key=lambda r: float(r.get("strength") or 0), reverse=True)
    capped = matches[:cap]
    digest = _observation_digest(capped)
    out: dict[str, Any] = {
        "ok": True,
        "count": min(len(matches), cap),
        "total": len(matches),
        "matches": capped,
        "hint": (
            "Each match is a named key with decayed strength. "
            "Stale observations below the floor are omitted. "
            "Narrow with name, q, after, before, or since_days. "
            "match.file is a db-relative path (e.g. hayden.json), not a URL — never web_fetch it."
        ),
    }
    if digest:
        out["digest"] = digest
        out["hint"] = (
            f"{out['hint']} Answer Hayden NOW in plain speech from digest and entries[].text. "
            "You are AI1, not Hayden — speak about him in second person (you/your)."
        )
    return out


def _observation_digest(matches: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in matches:
        if not isinstance(row, dict):
            continue
        key = str(row.get("name") or "").strip()
        entries = row.get("entries")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"{key}: {text}" if key else text)
    return "\n".join(lines)


def root_listing_hint() -> str:
    return _FILE_HINT
