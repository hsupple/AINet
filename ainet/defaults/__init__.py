"""Default JSON shapes for standard COP documents.

Edit these files to define how new COP docs start:
  Profile.json, Read.json, Plan.json, History.json
  (+ Decisions.json, Open Questions.json for projects)

`generic.json` is the fallback for any other new *.json name.
The `_ainet` block is documentation only and is stripped when seeding the DB.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

# Core COP document types — edit these.
COP_DOCUMENTS = (
    "Profile.json",
    "Read.json",
    "Plan.json",
    "History.json",
    "Decisions.json",
    "Open Questions.json",
)

# Hayden personal templates
HAYDEN_DOCUMENTS = (
    "Person.json",
    "SecretCategory.json",
    "Sides.json",
    "Captures.json",
    "MilestoneLog.json",
)


def defaults_dir() -> Path:
    return Path(__file__).resolve().parent


def list_templates() -> list[str]:
    return sorted(p.name for p in defaults_dir().glob("*.json"))


def load_default(filename: str) -> dict[str, Any] | list[Any] | Any:
    """Return a deep copy of the template for `filename` (basename).

    Exact match first, else `generic.json`. Strips `_ainet` metadata.
    """
    name = Path(filename).name
    path = defaults_dir() / name
    if not path.is_file():
        path = defaults_dir() / "generic.json"
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return _strip_meta(data)


def load_default_for_path(relative_path: str) -> dict[str, Any] | list[Any] | Any:
    """Resolve a template from a DB-relative path (basename + Hayden conventions)."""
    rel = str(relative_path).replace("\\", "/").strip("/")
    name = Path(rel).name
    parts = Path(rel).parts

    # One file per person
    if (
        len(parts) >= 4
        and parts[0] == "Hayden"
        and parts[1] == "Relationships"
        and parts[2] == "People"
        and name.endswith(".json")
    ):
        return load_default("Person.json")

    # Secrets category vaults (not Read/Index)
    if (
        len(parts) == 3
        and parts[0] == "Hayden"
        and parts[1] == "Secrets"
        and name.endswith(".json")
        and name not in {"Read.json", "Index.json"}
    ):
        return load_default("SecretCategory.json")

    if (
        len(parts) >= 3
        and parts[0] == "Hayden"
        and parts[1] == "Memories"
        and parts[2] == "Milestones"
        and name == "Log.json"
    ):
        return load_default("MilestoneLog.json")

    if (
        len(parts) == 3
        and parts[0] == "Hayden"
        and parts[1] == "Inbox"
        and name == "Captures.json"
    ):
        return load_default("Captures.json")

    return load_default(name)


def _strip_meta(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: deepcopy(v) for k, v in data.items() if k != "_ainet"}
    return deepcopy(data)


def package_template_text(filename: str) -> str:
    name = Path(filename).name
    path = defaults_dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"No default template named {name}")
    return path.read_text(encoding="utf-8")
