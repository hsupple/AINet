"""SOI filing — log_item is the only write path."""

from __future__ import annotations

from typing import Any

from ainet.logstore import log_item as _log_item
from ainet.tools.ops import DatabaseTools


def file_note(
    db: DatabaseTools,
    *,
    entry_id: str = "",
    entry_ids: list[Any] | None = None,
    dest: str = "",
    text: str = "",
    label: str = "",
    reason: str = "",
    summary: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Back-compat wrapper: file_note(text=) becomes log_item(reason=)."""
    return _log_item(
        db,
        dest=dest,
        label=label or dest,
        reason=str(reason or text or "").strip(),
        entry_id=entry_id,
        entry_ids=entry_ids,
        summary=summary,
    )
