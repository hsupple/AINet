"""Map dest labels/paths to db/ folder paths using the live folder tree."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from ainet.tools.ops import DatabaseTools
from ainet.tools.paths import normalize_relpath

_DISCARD = frozenset({"discard", "drop", "ephemeral"})
_DOMAINS = ("Hayden", "Household", "Projects", "Questions")
_BLOCKED_TOP_LEVEL = frozenset({"School", "Work"})
_ROOT_DESTS = frozenset({"Questions"})
_RESEARCH_DESTS = frozenset({"research", "questions/research"})


def build_file_structure(db: DatabaseTools, *, max_depth: int = 6) -> dict[str, Any]:
    """Full folder trees for the four main domains (included in every filing batch)."""
    out: dict[str, Any] = {}
    for domain in _DOMAINS:
        if not db.paths.resolve(domain).exists():
            out[domain] = {"path": domain, "type": "dir", "children": []}
            continue
        try:
            out[domain] = (db.tree(domain, max_depth=max_depth).get("tree") or {})
        except Exception:
            out[domain] = {"path": domain, "type": "dir", "children": [], "error": True}
    return out


def _list_folder_paths(db: DatabaseTools) -> list[str]:
    paths: list[str] = []
    root = db.paths.root
    for domain in _DOMAINS:
        base = root / domain
        if not base.is_dir():
            continue
        for node in sorted(base.rglob("*")):
            if not node.is_dir():
                continue
            if "runtime" in node.parts or "__pycache__" in node.parts:
                continue
            try:
                rel = node.relative_to(root).as_posix()
            except ValueError:
                continue
            if rel.lower() == "questions/research":
                continue
            paths.append(rel)
    return paths


def _folders_named(db: DatabaseTools, name: str) -> list[str]:
    want = name.casefold()
    return [p for p in _list_folder_paths(db) if PurePosixPath(p).name.casefold() == want]


def _qualify_path(raw: str) -> str:
    text = raw.replace("\\", "/").strip("/")
    if not text:
        return ""
    if text.split("/", 1)[0] in _DOMAINS:
        return normalize_relpath(text)
    return normalize_relpath(f"Hayden/{text}")


def resolve_dest(db: DatabaseTools, dest: str, *, user_text: str = "") -> str | None:
    """Return folder path relative to db/, 'discard', or None."""
    _ = user_text
    raw = (dest or "").strip()
    if not raw:
        return None
    if raw.lower() in _DISCARD:
        return "discard"

    normalized = raw.replace("\\", "/").strip("/").lower()
    if normalized in _RESEARCH_DESTS:
        return "Questions"

    # Questions is the one create_under root where filing at the root is allowed.
    if raw in _ROOT_DESTS:
        return raw

    # School/Work are not valid top-level filing targets here.
    if raw in _BLOCKED_TOP_LEVEL:
        return None

    # dest=Projects means Hayden/Projects (informal notes), not the top-level COP root.
    if raw == "Projects":
        hayden_projects = "Hayden/Projects"
        if db.paths.resolve(hayden_projects).is_dir():
            return hayden_projects
        return None

    # Other top-level domains are not filing targets — pick a child folder.
    if raw in _DOMAINS:
        return None

    # Path form: Hayden/Values, Projects/AINet, Preferences/Food
    if "/" in raw:
        path = _qualify_path(raw)
        if path.lower() == "questions/research":
            return "Questions"
        target = db.paths.resolve(path)
        if target.is_dir():
            return path
        parent = str(PurePosixPath(path).parent)
        if parent and parent != "." and db.paths.resolve(parent).is_dir():
            return path
        return None

    # Unique folder name anywhere in the tree
    matches = _folders_named(db, raw)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer shortest path (usually top-level hayden folder)
        matches.sort(key=lambda p: (p.count("/"), p))
        return matches[0]

    # Project by name — only when that project folder already exists
    project = f"Projects/{raw}"
    if db.paths.resolve(project).is_dir():
        return project

    # New folder under Hayden if parent domain exists and name is safe
    if raw.replace("_", "").replace("-", "").replace(" ", "").isalnum() and len(raw) <= 64:
        return f"Hayden/{raw}"

    return None


def list_dest_labels(db: DatabaseTools) -> dict[str, list[str]]:
    """Flat folder paths — compact dest picker (also in file_structure trees)."""
    paths = _list_folder_paths(db)
    return {
        "folders": paths,
        "projects": sorted(
            PurePosixPath(p).name
            for p in paths
            if p.startswith("Projects/") and p.count("/") == 1
        ),
    }
