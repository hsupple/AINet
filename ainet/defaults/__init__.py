"""Default JSON shapes for knowledge files and generic new JSON."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

KNOWLEDGE_DOCUMENTS = (
    "hayden.json",
    "people.json",
    "questions.json",
    "household.json",
    "memories.json",
    "secrets.json",
    "project.json",
)


def defaults_dir() -> Path:
    return Path(__file__).resolve().parent


def list_templates() -> list[str]:
    return sorted(p.name for p in defaults_dir().glob("*.json"))


_LOG_FILES = {
    "people.json",
    "questions.json",
    "household.json",
    "memories.json",
    "secrets.json",
}


def load_default(filename: str) -> dict[str, Any] | list[Any] | Any:
    """Return a deep copy of the template for `filename` (basename).

    Exact match first, else `generic.json`. Strips `_ainet` metadata.
    """
    name = Path(filename).name
    key = name.casefold()
    if key == "hayden.json":
        name = "Hayden.json"
    elif key in _LOG_FILES:
        name = "LogFile.json"
    path = defaults_dir() / name
    if not path.is_file():
        path = defaults_dir() / "generic.json"
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return _strip_meta(data)


def load_default_for_path(relative_path: str) -> dict[str, Any] | list[Any] | Any:
    """Resolve a template from a DB-relative path."""
    rel = str(relative_path).replace("\\", "/").strip("/")
    name = Path(rel).name
    parts = Path(rel).parts
    if name.casefold() == "project.json" or (
        len(parts) >= 2 and parts[0].casefold() == "projects" and name.casefold() == "project.json"
    ):
        data = load_default("project.json")
        if isinstance(data, dict) and len(parts) >= 2:
            data = dict(data)
            data["name"] = parts[1]
        return data
    if name.casefold() == "hayden.json":
        return load_default("Hayden.json")
    return load_default(name)


def _strip_meta(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: deepcopy(v) for k, v in data.items() if k != "_ainet"}
    return deepcopy(data)


def package_template_text(filename: str) -> str:
    name = Path(filename).name
    path = defaults_dir() / name
    if not path.is_file() and name.casefold() == "hayden.json":
        path = defaults_dir() / "Hayden.json"
    if not path.is_file():
        raise FileNotFoundError(f"No default template named {name}")
    return path.read_text(encoding="utf-8")
