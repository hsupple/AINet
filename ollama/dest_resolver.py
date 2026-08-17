"""Map dest labels to knowledge files."""

from __future__ import annotations

from typing import Any

from ainet.logstore import dest_names, list_existing_labels, resolve_dest as _resolve
from ainet.tools.ops import DatabaseTools


def resolve_dest(db: DatabaseTools, dest: str, *, user_text: str = "") -> str | None:
    """Return knowledge path, 'discard', or None."""
    _ = user_text
    hit = _resolve(db, dest)
    if hit == "discard":
        return "discard"
    if not hit:
        return None
    path, _array = hit
    return path


def list_dest_labels(db: DatabaseTools) -> dict[str, Any]:
    from ainet.logstore import _project_names

    return {
        "dests": dest_names(),
        "projects": _project_names(db),
        "labels": list_existing_labels(db),
    }
