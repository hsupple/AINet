"""Mode registry."""

from __future__ import annotations

from ollama.modes import companion, conversation, deep_research, dormant, planner, project, soi, soi_test
from ollama.modes.base import Mode

_MODES: dict[str, Mode] = {
    companion.MODE.id: companion.MODE,
    conversation.MODE.id: conversation.MODE,
    planner.MODE.id: planner.MODE,
    deep_research.MODE.id: deep_research.MODE,
    project.MODE.id: project.MODE,
    soi.MODE.id: soi.MODE,
    soi_test.MODE.id: soi_test.MODE,
    soi_test.MODE_P2.id: soi_test.MODE_P2,
    "dormant": dormant.MODE,
}

DEFAULT_MODE_ID = companion.MODE.id


def get_mode(mode_id: str) -> Mode:
    key = mode_id.strip().lower().replace(" ", "_").replace("-", "_")
    if key in {"research", "deepresearch"}:
        key = deep_research.MODE.id
    if key == "quiz":
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
