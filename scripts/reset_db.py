#!/usr/bin/env python3
"""Reset the local AINet database to an empty skeleton.

Keeps schema (Rules.txt, Folderrules.json) and the domain folder tree.
Clears OAC/SOI queues, runtime, generated people/history, and
empties remaining JSON leaves (named templates when they exist).

Does not touch mac/db. Real db/ is not changed unless you pass --yes.

Examples (from repo root):
  python scripts/reset_db.py
  python scripts/reset_db.py --yes
  python scripts/reset_db.py --yes --no-backup
  python scripts/reset_db.py --yes --queues-only
  python scripts/reset_db.py --db path/to/db --yes
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ainet.defaults import COP_DOCUMENTS, HAYDEN_DOCUMENTS, load_default_for_path
from ainet.tools import changelog as changelog_mod
from ainet.tools.fsutil import atomic_write_text
from ollama.config import default_db_root

KEEP_ROOT_FILES = {"Rules.txt", "Folderrules.json"}
KEEP_NAMES = {".gitkeep"}
TEMPLATE_NAMES = set(COP_DOCUMENTS) | set(HAYDEN_DOCUMENTS)

SECRET_VAULTS = (
    "Hayden/Secrets/Personal.json",
    "Hayden/Secrets/Family.json",
    "Hayden/Secrets/Relational.json",
    "Hayden/Secrets/Sexual.json",
    "Hayden/Secrets/Dangerous.json",
    "Hayden/Secrets/Other.json",
)

GENERATED_FILE_GLOBS = (
    "runtime/oac/sessions/*.json",
    "runtime/oac/current.json",
    "runtime/soi/*.json",
    "runtime/soi/*.jsonl",
    "Hayden/Relationships/People/*.json",
)

GENERATED_DIR_GLOBS = (
    "School/Courses/*",
    "Hayden/Memories/Childhood/*",
    "Hayden/Memories/Formative/*",
    "Hayden/Memories/Recent/*",
    "Hayden/Memories/Wounds/*",
    "Hayden/Memories/Wins/*",
    "Hayden/Preferences/Music/*",
    "**/History/*",
)

SKELETON_DIRS = (
    "runtime",
    "runtime/oac",
    "runtime/oac/sessions",
    "runtime/soi",
    "Hayden",
    "Hayden/Identity",
    "Hayden/Values",
    "Hayden/Preferences",
    "Hayden/Preferences/Music",
    "Hayden/Habits",
    "Hayden/Desires",
    "Hayden/Relationships",
    "Hayden/Relationships/People",
    "Hayden/Secrets",
    "Hayden/Memories",
    "Hayden/Memories/Milestones",
    "Hayden/Memories/Childhood",
    "Hayden/Memories/Formative",
    "Hayden/Memories/Recent",
    "Hayden/Memories/Wounds",
    "Hayden/Memories/Wins",
    "Hayden/Body",
    "Hayden/Psychology",
    "Hayden/Inbox",
    "Hayden/History",
    "School",
    "School/Courses",
    "School/History",
    "Work",
    "Work/Projects",
    "Work/History",
    "Household",
    "Household/Pantry",
    "Household/Maintenance",
    "Household/History",
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _empty_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _empty_value(v) for k, v in value.items() if k != "_ainet"}
    if isinstance(value, list):
        return []
    if isinstance(value, str):
        return ""
    if isinstance(value, bool):
        return False
    if isinstance(value, int) and not isinstance(value, bool):
        return 0
    if isinstance(value, float):
        return 0.0
    return None


def _load_json(path: Path) -> Any | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data


class ResetPlan:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.actions: list[str] = []
        self.removed: set[str] = set()

    def log(self, action: str) -> None:
        self.actions.append(action)

    def apply(self, dry_run: bool, fn, action: str) -> None:
        self.log(action)
        if not dry_run:
            fn()

    def mark_removed(self, rel: str) -> None:
        self.removed.add(rel.replace("\\", "/"))

    def is_removed(self, rel: str) -> bool:
        rel = rel.replace("\\", "/")
        if rel in self.removed:
            return True
        return any(rel.startswith(item.rstrip("/") + "/") for item in self.removed)


def _ensure_skeleton(root: Path, plan: ResetPlan, *, dry_run: bool) -> None:
    for rel in SKELETON_DIRS:
        path = root / rel
        if path.is_dir():
            continue
        plan.apply(dry_run, lambda p=path: p.mkdir(parents=True, exist_ok=True), f"mkdir {rel}")
        keep = path / ".gitkeep"
        if not keep.exists():
            plan.apply(
                dry_run,
                lambda k=keep: k.write_text("", encoding="utf-8"),
                f"gitkeep {rel}/.gitkeep",
            )


def _unlink(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _wipe_children(root: Path, rel_dir: str, plan: ResetPlan, *, dry_run: bool) -> None:
    """Delete every child of a folder except .gitkeep. Uses iterdir (Windows-safe)."""
    folder = root / rel_dir
    if not folder.is_dir():
        return
    for child in list(folder.iterdir()):
        if child.name in KEEP_NAMES:
            continue
        rel = _rel(root, child)
        plan.mark_removed(rel)
        plan.apply(dry_run, lambda p=child: _unlink(p), f"delete {rel}")


def _clear_generated(root: Path, plan: ResetPlan, *, dry_run: bool) -> None:
    for rel in SECRET_VAULTS:
        path = root / rel
        if path.exists():
            plan.mark_removed(rel)
            plan.apply(dry_run, lambda p=path: _unlink(p), f"delete {rel}")

    _wipe_children(root, "Hayden/Relationships/People", plan, dry_run=dry_run)
    _wipe_children(root, "School/Courses", plan, dry_run=dry_run)
    _wipe_children(root, "Hayden/Memories/Childhood", plan, dry_run=dry_run)
    _wipe_children(root, "Hayden/Memories/Formative", plan, dry_run=dry_run)
    _wipe_children(root, "Hayden/Memories/Recent", plan, dry_run=dry_run)
    _wipe_children(root, "Hayden/Memories/Wounds", plan, dry_run=dry_run)
    _wipe_children(root, "Hayden/Memories/Wins", plan, dry_run=dry_run)
    _wipe_children(root, "Hayden/Preferences/Music", plan, dry_run=dry_run)
    _wipe_children(root, "Hayden/History", plan, dry_run=dry_run)
    _wipe_children(root, "School/History", plan, dry_run=dry_run)
    _wipe_children(root, "Work/History", plan, dry_run=dry_run)
    _wipe_children(root, "Household/History", plan, dry_run=dry_run)
    _wipe_children(root, "Work/Projects/AINet/History", plan, dry_run=dry_run)

    projects = root / "Work" / "Projects"
    if projects.is_dir():
        for path in projects.iterdir():
            if path.name in KEEP_NAMES or path.name == "AINet":
                continue
            rel = _rel(root, path)
            plan.mark_removed(rel)
            plan.apply(dry_run, lambda p=path: _unlink(p), f"delete {rel}")

    for pattern in GENERATED_FILE_GLOBS:
        for path in root.glob(pattern):
            if path.name in KEEP_NAMES:
                continue
            rel = _rel(root, path)
            if plan.is_removed(rel):
                continue
            plan.mark_removed(rel)
            plan.apply(dry_run, lambda p=path: _unlink(p), f"delete {rel}")

    for pattern in GENERATED_DIR_GLOBS:
        for path in root.glob(pattern):
            if path.name in KEEP_NAMES:
                continue
            if path.name in {"History.json"}:
                continue
            rel = _rel(root, path)
            if plan.is_removed(rel):
                continue
            plan.mark_removed(rel)
            plan.apply(dry_run, lambda p=path: _unlink(p), f"delete {rel}")


def _reset_queues(root: Path, plan: ResetPlan, *, dry_run: bool) -> None:
    def write_changelog() -> None:
        _write_json(root / "Changelog.json", {"version": 1, "entries": []})

    def write_masterlog() -> None:
        _write_json(
            root / "Masterlog.json",
            {"version": 1, "entries": [], "last_updated": ""},
        )

    def write_calendar() -> None:
        _write_json(
            root / "Calendar.json",
            {"version": 1, "mutable_by": "code_only", "events": []},
        )

    plan.apply(dry_run, write_changelog, "reset Changelog.json")
    plan.apply(dry_run, write_masterlog, "reset Masterlog.json")
    plan.apply(dry_run, write_calendar, "reset Calendar.json")
    if not dry_run:
        changelog_mod.ensure_changelog_file(root)
        changelog_mod.ensure_masterlog_file(root)


def _reset_json_file(root: Path, path: Path, plan: ResetPlan, *, dry_run: bool) -> None:
    rel = _rel(root, path)
    name = path.name
    if name in KEEP_ROOT_FILES or rel in KEEP_ROOT_FILES:
        return
    if rel in {"Changelog.json", "Masterlog.json", "Calendar.json"}:
        return

    def write_template() -> None:
        payload = load_default_for_path(rel)
        _write_json(path, payload)

    def write_empty_shape() -> None:
        data = _load_json(path)
        if not isinstance(data, dict):
            _write_json(path, {})
            return
        _write_json(path, _empty_value(data))

    if (
        name in TEMPLATE_NAMES
        or rel.endswith("/Read.json")
        or rel in {"Hayden/Memories/Milestones/Log.json"}
    ):
        plan.apply(dry_run, write_template, f"template {rel}")
        return
    plan.apply(dry_run, write_empty_shape, f"empty {rel}")


def _reset_content(root: Path, plan: ResetPlan, *, dry_run: bool) -> None:
    for path in sorted(root.rglob("*.json")):
        if "runtime" in path.parts:
            continue
        if plan.is_removed(_rel(root, path)):
            continue
        _reset_json_file(root, path, plan, dry_run=dry_run)


def reset_db(
    root: Path,
    *,
    dry_run: bool,
    queues_only: bool,
    backup: bool,
) -> ResetPlan:
    if not root.exists():
        raise SystemExit(f"DB root does not exist: {root}")
    plan = ResetPlan(root)

    if backup and not dry_run:
        dest = root.parent / f".ainet-db-backup-{_utc_stamp()}"
        shutil.copytree(root, dest, dirs_exist_ok=False)
        plan.log(f"backup {dest}")
    elif backup and dry_run:
        plan.log(f"backup {root.parent / ('.ainet-db-backup-' + _utc_stamp())}")

    _reset_queues(root, plan, dry_run=dry_run)
    _ensure_skeleton(root, plan, dry_run=dry_run)
    if queues_only:
        for pattern in (
            "runtime/oac/sessions/*.json",
            "runtime/oac/current.json",
            "runtime/soi/*.json",
            "runtime/soi/*.jsonl",
        ):
            for path in root.glob(pattern):
                if path.name in KEEP_NAMES:
                    continue
                plan.apply(
                    dry_run,
                    lambda p=path: _unlink(p),
                    f"delete {_rel(root, path)}",
                )
    else:
        _clear_generated(root, plan, dry_run=dry_run)
        _reset_content(root, plan, dry_run=dry_run)
        _ensure_skeleton(root, plan, dry_run=dry_run)
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reset the local AINet db/ skeleton")
    parser.add_argument(
        "--db",
        type=Path,
        default=default_db_root(),
        help="Database root (default: <repo>/db)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Apply the reset. Without this flag the script only prints the plan.",
    )
    parser.add_argument(
        "--queues-only",
        action="store_true",
        help="Only clear Changelog, Masterlog, Calendar, and runtime/ (keep filed leaves)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip copying db/ to .ainet-db-backup-<timestamp>/",
    )
    args = parser.parse_args(argv)

    root = args.db.resolve()
    dry_run = not args.yes
    plan = reset_db(
        root,
        dry_run=dry_run,
        queues_only=args.queues_only,
        backup=not args.no_backup,
    )

    mode = "DRY RUN" if dry_run else "APPLIED"
    scope = "queues-only" if args.queues_only else "full"
    print(f"{mode}  scope={scope}  db={root}")
    for line in plan.actions:
        print(f"  {line}")
    print(f"{len(plan.actions)} actions")
    if dry_run:
        print("No files changed. Re-run with --yes to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
