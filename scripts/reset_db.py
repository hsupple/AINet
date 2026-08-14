#!/usr/bin/env python3
"""Reset / bootstrap the local AINet database to an empty skeleton.

Keeps schema (Rules.txt, Folderrules.json) from git. Everything else under
db/ is local — after a fresh clone, create it with:

  python scripts/reset_db.py --yes

That writes the folder tree + empty JSON leaves from ainet/defaults templates,
clears OAC/SOI queues/runtime, and empties living content.

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
    "Chats",
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
    "Hayden/Planner",
    "Hayden/Plans",
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
    "Questions",
)

# Empty leaves created on fresh clone (shapes come from ainet/defaults).
SEED_FILES = (
    "Hayden/Body/Energy.json",
    "Hayden/Body/Health.json",
    "Hayden/Body/History.json",
    "Hayden/Body/Notes.json",
    "Hayden/Body/Read.json",
    "Hayden/Body/Sensory.json",
    "Hayden/Desires/Goals.json",
    "Hayden/Desires/History.json",
    "Hayden/Desires/Longings.json",
    "Hayden/Desires/Notes.json",
    "Hayden/Desires/Read.json",
    "Hayden/Desires/Wants.json",
    "Hayden/Habits/Disciplines.json",
    "Hayden/Habits/History.json",
    "Hayden/Habits/Notes.json",
    "Hayden/Habits/Patterns.json",
    "Hayden/Habits/Read.json",
    "Hayden/Habits/Routines.json",
    "Hayden/Habits/Vices.json",
    "Hayden/History.json",
    "Hayden/Identity/Boundaries.json",
    "Hayden/Identity/Core.json",
    "Hayden/Identity/History.json",
    "Hayden/Identity/Notes.json",
    "Hayden/Identity/Personality.json",
    "Hayden/Identity/Read.json",
    "Hayden/Identity/Sides.json",
    "Hayden/Identity/Voice.json",
    "Hayden/Inbox/Captures.json",
    "Hayden/Inbox/History.json",
    "Hayden/Inbox/Notes.json",
    "Hayden/Inbox/Read.json",
    "Hayden/Memories/History.json",
    "Hayden/Memories/Index.json",
    "Hayden/Memories/Milestones/History.json",
    "Hayden/Memories/Milestones/Log.json",
    "Hayden/Memories/Milestones/Notes.json",
    "Hayden/Memories/Milestones/Read.json",
    "Hayden/Memories/Notes.json",
    "Hayden/Memories/Read.json",
    "Hayden/Notes.json",
    "Hayden/Plan.json",
    "Hayden/Planner/History.json",
    "Hayden/Planner/Notes.json",
    "Hayden/Planner/Read.json",
    "Hayden/Plans/History.json",
    "Hayden/Plans/Notes.json",
    "Hayden/Plans/Read.json",
    "Hayden/Preferences/Aesthetic.json",
    "Hayden/Preferences/Dislikes.json",
    "Hayden/Preferences/Food.json",
    "Hayden/Preferences/History.json",
    "Hayden/Preferences/Lifestyle.json",
    "Hayden/Preferences/Likes.json",
    "Hayden/Preferences/Media.json",
    "Hayden/Preferences/Music/Spotify.json",
    "Hayden/Preferences/Notes.json",
    "Hayden/Preferences/Read.json",
    "Hayden/Profile.json",
    "Hayden/Psychology/Attachments.json",
    "Hayden/Psychology/Coping.json",
    "Hayden/Psychology/Defense.json",
    "Hayden/Psychology/Fears.json",
    "Hayden/Psychology/History.json",
    "Hayden/Psychology/Notes.json",
    "Hayden/Psychology/Read.json",
    "Hayden/Psychology/Schedule.json",
    "Hayden/Psychology/Triggers.json",
    "Hayden/Read.json",
    "Hayden/Relationships/History.json",
    "Hayden/Relationships/Index.json",
    "Hayden/Relationships/Notes.json",
    "Hayden/Relationships/Read.json",
    "Hayden/Relationships/Schedule.json",
    "Hayden/Secrets/History.json",
    "Hayden/Secrets/Index.json",
    "Hayden/Secrets/Notes.json",
    "Hayden/Secrets/Read.json",
    "Hayden/Values/History.json",
    "Hayden/Values/Notes.json",
    "Hayden/Values/Principles.json",
    "Hayden/Values/Priorities.json",
    "Hayden/Values/Read.json",
    "Household/History.json",
    "Household/Maintenance/History.json",
    "Household/Maintenance/Notes.json",
    "Household/Maintenance/Read.json",
    "Household/Notes.json",
    "Household/Pantry/History.json",
    "Household/Pantry/Notes.json",
    "Household/Pantry/Read.json",
    "Household/Read.json",
    "Household/Wants.json",
    "Questions/History.json",
    "Questions/Notes.json",
    "Questions/Read.json",
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


def _ensure_seed_files(root: Path, plan: ResetPlan, *, dry_run: bool) -> None:
    """Create missing standard JSON leaves from ainet/defaults (fresh-clone path)."""
    for rel in SEED_FILES:
        path = root / rel
        if path.exists():
            continue

        def write_seed(p: Path = path, r: str = rel) -> None:
            p.parent.mkdir(parents=True, exist_ok=True)
            _write_json(p, load_default_for_path(r))

        plan.apply(dry_run, write_seed, f"seed {rel}")


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
    if rel in {
        "Changelog.json",
        "Masterlog.json",
        "Calendar.json",
        "Hayden/Preferences/Music/Spotify.json",
    }:
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


def _preserve_chats(root: Path, plan: ResetPlan, *, dry_run: bool) -> None:
    """Copy live OAC session logs into db/Chats/ so a personal-DB reset keeps them."""
    dest = root / "Chats"
    if not dest.exists():
        plan.apply(dry_run, lambda: dest.mkdir(parents=True, exist_ok=True), "mkdir Chats")
    src = root / "runtime" / "oac" / "sessions"
    if src.is_dir():
        for path in src.glob("*.json"):
            target = dest / path.name
            if target.exists():
                continue
            plan.apply(
                dry_run,
                lambda p=path, t=target: shutil.copy2(p, t),
                f"keep chat {path.name}",
            )
    current = root / "runtime" / "oac" / "current.json"
    dest_current = dest / "current.json"
    if current.is_file() and not dest_current.exists():
        plan.apply(
            dry_run,
            lambda: shutil.copy2(current, dest_current),
            "keep Chats/current.json",
        )


def _reset_content(root: Path, plan: ResetPlan, *, dry_run: bool) -> None:
    for path in sorted(root.rglob("*.json")):
        if "runtime" in path.parts or "Chats" in path.parts:
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
        if dry_run:
            plan = ResetPlan(root)
            plan.log(f"mkdir {root}")
            return plan
        root.mkdir(parents=True, exist_ok=True)
    schema = (root / "Rules.txt", root / "Folderrules.json")
    if not all(p.is_file() for p in schema):
        raise SystemExit(
            "db/Rules.txt and db/Folderrules.json are required (they come from git). "
            "Clone/pull the repo first, then run: python scripts/reset_db.py --yes"
        )
    plan = ResetPlan(root)

    if backup and not dry_run:
        dest = root.parent / f".ainet-db-backup-{_utc_stamp()}"
        shutil.copytree(root, dest, dirs_exist_ok=False)
        plan.log(f"backup {dest}")
    elif backup and dry_run:
        plan.log(f"backup {root.parent / ('.ainet-db-backup-' + _utc_stamp())}")

    _preserve_chats(root, plan, dry_run=dry_run)
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
        _ensure_seed_files(root, plan, dry_run=dry_run)
        _reset_content(root, plan, dry_run=dry_run)
        _ensure_skeleton(root, plan, dry_run=dry_run)
        _ensure_seed_files(root, plan, dry_run=dry_run)
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
