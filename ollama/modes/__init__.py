"""Mode registry."""

from __future__ import annotations

from ollama.modes import companion, deep_research, dormant, project, soi, soi_test
from ollama.modes.base import Mode

_MODES: dict[str, Mode] = {
    companion.MODE.id: companion.MODE,
    deep_research.MODE.id: deep_research.MODE,
    project.MODE.id: project.MODE,
    soi.MODE.id: soi.MODE,
    soi_test.MODE.id: soi_test.MODE,
    soi_test.MODE_P2.id: soi_test.MODE_P2,
    "dormant": dormant.MODE,
}

# Old flavor names all resolve to the single OAC personality.
_OAC_ALIASES = {
    "conversation": companion.MODE.id,
    "planner": companion.MODE.id,
    "quiz": companion.MODE.id,
    "oac": companion.MODE.id,
}

DEFAULT_MODE_ID = companion.MODE.id

# Modes shown in the web chat picker (not capability/internal).
UI_MODE_IDS = (companion.MODE.id, deep_research.MODE.id)


def get_mode(mode_id: str) -> Mode:
    key = mode_id.strip().lower().replace(" ", "_").replace("-", "_")
    if key in {"research", "deepresearch"}:
        key = deep_research.MODE.id
    key = _OAC_ALIASES.get(key, key)
    if key not in _MODES:
        known = ", ".join(sorted(set(_MODES) | set(_OAC_ALIASES)))
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


def list_ui_modes(*, project_focused: bool = False) -> list[Mode]:
    ids = list(UI_MODE_IDS)
    if project_focused:
        ids.append(project.MODE.id)
    out: list[Mode] = []
    seen: set[str] = set()
    for mode_id in ids:
        mode = _MODES[mode_id]
        if mode.id in seen:
            continue
        seen.add(mode.id)
        out.append(mode)
    return out
