"""User Projects under db/Projects/ — create, list, resolve paths."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from ainet.defaults import load_default_for_path
from ainet.tools.ops import DatabaseTools
from ainet.tools.paths import normalize_relpath

PROJECTS_ROOT = "Projects"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}$")

# Seeded into every new project (JSON via defaults + folders).
_PROJECT_FILES = (
    "Read.json",
    "History.json",
    "Notes.json",
    "Plan.json",
    "Profile.json",
)
_PROJECT_DIRS = (
    "Files",
    "History",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slug_name(name: str) -> str:
    text = re.sub(r"\s+", " ", (name or "").strip())
    text = text.strip("/\\")
    if not text:
        raise ValueError("project name is required")
    # Allow "Projects/Foo" or just "Foo"
    if text.replace("\\", "/").startswith(f"{PROJECTS_ROOT}/"):
        text = text.replace("\\", "/").split("/", 1)[1]
    if "/" in text or "\\" in text:
        raise ValueError("project name must be a single folder name (no slashes)")
    if not _NAME_RE.match(text):
        raise ValueError(
            "project name must start with alphanumeric and use letters, numbers, space, _.- only"
        )
    return text


def project_path(name: str) -> str:
    return f"{PROJECTS_ROOT}/{_slug_name(name)}"


def user_project_name_from_path(path: str) -> str | None:
    """If path is Projects/<Name>/..., return <Name>. Ignores Work/Projects and Hayden/Projects."""
    raw = (path or "").replace("\\", "/").strip().strip("/")
    if not raw:
        return None
    parts = [p for p in raw.split("/") if p]
    if len(parts) >= 2 and parts[0].casefold() == PROJECTS_ROOT.casefold() and parts[1]:
        try:
            return _slug_name(parts[1])
        except ValueError:
            return None
    return None


def ensure_projects_root(db: DatabaseTools) -> None:
    """Host-side container mkdir — AI cannot create top-level db/ entries."""
    root = db.paths.root / PROJECTS_ROOT
    if root.is_dir():
        return
    if root.exists():
        raise ValueError(f"{PROJECTS_ROOT} exists but is not a directory")
    root.mkdir(parents=True, exist_ok=True)


def list_projects(db: DatabaseTools) -> dict[str, Any]:
    root = db.paths.resolve(PROJECTS_ROOT)
    if not root.is_dir():
        return {"ok": True, "root": PROJECTS_ROOT, "count": 0, "projects": []}
    rows: list[dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        rel = f"{PROJECTS_ROOT}/{child.name}"
        summary = ""
        read_path = child / "Read.json"
        if read_path.is_file():
            try:
                data = db.read_json(f"{rel}/Read.json").get("data") or {}
                if isinstance(data, dict):
                    summary = str(data.get("summary") or data.get("state") or "")[:200]
            except Exception:
                pass
        rows.append({"name": child.name, "path": rel, "summary": summary})
    return {"ok": True, "root": PROJECTS_ROOT, "count": len(rows), "projects": rows}


def create_project(
    db: DatabaseTools,
    *,
    name: str,
    summary: str = "",
) -> dict[str, Any]:
    """Create Projects/<Name>/ with Read/History/Notes/Plan/Profile + Files/History/."""
    folder = project_path(name)
    target = db.paths.resolve(folder)
    created: list[str] = []
    existed = target.exists()
    if existed and not target.is_dir():
        raise ValueError(f"A file already exists at {folder}")

    ensure_projects_root(db)

    if not target.exists():
        db.create_folder(folder, summary=summary or f"Create project {folder}")
        created.append(folder)

    for dirname in _PROJECT_DIRS:
        path = f"{folder}/{dirname}"
        if not db.paths.resolve(path).exists():
            db.create_folder(path, summary=f"Project folder {dirname}")
            created.append(path)

    now = _utc_now()
    for filename in _PROJECT_FILES:
        path = f"{folder}/{filename}"
        if db.paths.resolve(path).exists():
            continue
        data = load_default_for_path(path)
        if isinstance(data, dict) and filename == "Read.json":
            data = dict(data)
            data["summary"] = (summary or f"Project {PurePosixPath(folder).name}").strip()[:400]
            data["state"] = "active"
            data["last_updated"] = now
            data["needs_update"] = False
            data["read_changelog"] = []
        if isinstance(data, dict) and filename == "Profile.json":
            data = dict(data)
            data.setdefault("name", PurePosixPath(folder).name)
            if summary:
                data["summary"] = summary[:400]
        db.write_json(path, data, create=True, summary=f"Seed {filename} for {folder}")
        created.append(path)

    return {
        "ok": True,
        "created": not existed,
        "name": PurePosixPath(folder).name,
        "path": folder,
        "seeded": created,
        "hint": (
            f"Project directory ready at {folder} (not a generic folder). "
            f"Call open_project(name={PurePosixPath(folder).name!r}) to focus this chat on it."
        ),
    }


def find_project(db: DatabaseTools, name_or_path: str) -> str | None:
    """Return Projects/<Name> path or None."""
    raw = (name_or_path or "").strip().replace("\\", "/")
    if not raw:
        return None
    if raw.casefold() == PROJECTS_ROOT.casefold():
        return None
    try:
        if raw.startswith(f"{PROJECTS_ROOT}/") or raw.startswith(f"{PROJECTS_ROOT.casefold()}/"):
            parts = PurePosixPath(normalize_relpath(raw)).parts
            if len(parts) >= 2:
                candidate = f"{PROJECTS_ROOT}/{parts[1]}"
                if db.paths.resolve(candidate).is_dir():
                    return candidate
        name = _slug_name(raw)
    except ValueError:
        return None
    candidate = f"{PROJECTS_ROOT}/{name}"
    if db.paths.resolve(candidate).is_dir():
        return candidate
    # Case-insensitive match
    root = db.paths.resolve(PROJECTS_ROOT)
    if root.is_dir():
        want = name.casefold()
        for child in root.iterdir():
            if child.is_dir() and child.name.casefold() == want:
                return f"{PROJECTS_ROOT}/{child.name}"
    return None


def resolve_under_project(project_root: str, path: str, db: DatabaseTools) -> str:
    """Map a user/AI path to a db-relative path under project_root.

    Accepts:
      - bare filename (searches project tree)
      - relative path like Notes.json or Files/ideas.md
      - full Projects/Name/... path
      - '.' for the project root itself
    """
    root = normalize_relpath(project_root)
    raw = (path or "").strip().replace("\\", "/")
    if not raw or raw in {".", "./"}:
        return root

    # Full path already under this project
    try:
        norm = normalize_relpath(raw)
    except Exception as exc:
        raise ValueError(str(exc)) from exc

    if norm == root or norm.startswith(root + "/"):
        return norm

    parts = PurePosixPath(norm).parts
    top = parts[0] if parts else ""
    root_name = PurePosixPath(root).name

    # Bare name matching the project folder → the project root (for rename)
    if "/" not in norm and top.casefold() == root_name.casefold():
        return root

    # Absolute-ish escape: references another top-level db domain / other project
    if top:
        top_abs = db.paths.resolve(top)
        if top_abs.is_dir() and top_abs.parent.resolve() == db.paths.root.resolve():
            raise ValueError(f"Path '{path}' is outside the focused project {root}")

    # Relative to project
    rel = norm.lstrip("./")
    candidate = normalize_relpath(f"{root}/{rel}")
    if candidate != root and not candidate.startswith(root + "/"):
        raise ValueError(f"Path escapes project root: {path}")

    # If exact exists, use it
    if db.paths.resolve(candidate).exists():
        return candidate

    # Bare filename search within project
    name = PurePosixPath(rel).name
    if "/" not in rel.strip("/"):
        base = db.paths.resolve(root)
        if base.is_dir():
            matches = [p for p in base.rglob(name) if p.is_file() and "runtime" not in p.parts]
            if len(matches) == 1:
                return db.paths.relative_of(matches[0])
            if len(matches) > 1:
                opts = [db.paths.relative_of(p) for p in matches[:8]]
                raise ValueError(f"Multiple files named {name!r}: {', '.join(opts)}")
    return candidate


def resolve_project_rename_dest(project_root: str, dest: str) -> str | None:
    """If dest is a sibling rename of the project folder, return Projects/<NewName>."""
    root = normalize_relpath(project_root)
    raw = (dest or "").strip().replace("\\", "/")
    if not raw:
        return None
    try:
        norm = normalize_relpath(raw)
    except Exception:
        return None
    parts = PurePosixPath(norm).parts
    # Projects/NewName
    if len(parts) == 2 and parts[0].casefold() == PROJECTS_ROOT.casefold():
        if normalize_relpath(f"{PROJECTS_ROOT}/{parts[1]}") != root:
            return f"{PROJECTS_ROOT}/{parts[1]}"
        return root
    # Bare NewName (sibling under Projects/)
    if len(parts) == 1 and parts[0].casefold() != PurePosixPath(root).name.casefold():
        return f"{PROJECTS_ROOT}/{parts[0]}"
    return None


def ensure_path_in_project(project_root: str, path: str) -> str:
    """Normalize and assert path stays inside project_root."""
    root = normalize_relpath(project_root)
    norm = normalize_relpath(path)
    if norm != root and not norm.startswith(root + "/"):
        raise ValueError(f"Blocked: '{path}' is outside focused project {root}")
    return norm
