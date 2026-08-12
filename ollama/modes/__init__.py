"""Mode registry."""

from __future__ import annotations

from ollama.modes import companion, conversation, dormant, planner, soi, soi_test
from ollama.modes.base import Mode

_MODES: dict[str, Mode] = {
    companion.MODE.id: companion.MODE,
    conversation.MODE.id: conversation.MODE,
    planner.MODE.id: planner.MODE,
    soi.MODE.id: soi.MODE,
    soi_test.MODE.id: soi_test.MODE,
    "dormant": dormant.MODE,
}

DEFAULT_MODE_ID = companion.MODE.id


def get_mode(mode_id: str) -> Mode:
    key = mode_id.strip().lower()
    if key in {"research", "quiz"}:
        key = companion.MODE.id
    if key not in _MODES:
        known = ", ".join(sorted(set(_MODES)))
        raise KeyError(f"Unknown mode '{mode_id}'. Known modes: {known}")
    return _MODES[key]


def list_modes() -> list[Mode]:
    seen: set[str] = set()
    out: list[Mode] = []
    for mode in (_MODES[k] for k in sorted(_MODES)):
        if mode.id in seen:
            continue
        seen.add(mode.id)
        out.append(mode)
    return out
