#!/usr/bin/env python3
"""Reset / bootstrap the local AINet database to an empty skeleton.

Keeps schema (Rules.txt, Folderrules.json) from git. Everything else under
db/ is local — after a fresh clone, create it with:

  python scripts/reset_db.py --yes

Does not touch mac/db. Real db/ is not changed unless you pass --yes.

Examples (from repo root):
  python scripts/reset_db.py
  python scripts/reset_db.py --yes
  python scripts/reset_db.py --yes --no-backup
  python scripts/reset_db.py --yes --queues-only
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

from ainet.defaults import load_default_for_path
from ainet.logstore import ROOT_FILES, empty_hayden, empty_log, ensure_knowledge_files
from ainet.tools import changelog as changelog_mod
from ainet.tools.fsutil import atomic_write_text
from ollama.config import default_db_root

KEEP_ROOT_FILES = {"Rules.txt", "Folderrules.json"}
KEEP_NAMES = {".gitkeep"}

SKELETON_DIRS = (
    "runtime",
    "runtime/oac",
    "runtime/oac/sessions",
    "runtime/soi",
    "runtime/spotify",
    "Chats",
    "Projects",
)

PURGE_DIRS = (
    "Hayden",
    "School",
    "Work",
    "Household",
    "Questions",
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


class ResetPlan:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.actions: list[str] = []

    def log(self, action: str) -> None:
        self.actions.append(action)

    def apply(self, dry_run: bool, fn, action: str) -> None:
        self.log(action)
        if not dry_run:
            fn()


def _unlink(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _ensure_skeleton(root: Path, plan: ResetPlan, *, dry_run: bool) -> None:
    for rel in SKELETON_DIRS:
        path = root / rel
        if path.is_dir():
            continue
        plan.apply(dry_run, lambda p=path: p.mkdir(parents=True, exist_ok=True), f"mkdir {rel}")


def _ensure_knowledge(root: Path, plan: ResetPlan, *, dry_run: bool) -> None:
    for name in ROOT_FILES:
        path = root / name
        if path.exists():
            continue
        payload = empty_hayden() if name == "hayden.json" else empty_log()
        plan.apply(dry_run, lambda p=path, d=payload: _write_json(p, d), f"seed {name}")


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


def _preserve_chats(root: Path, plan: ResetPlan, *, dry_run: bool) -> None:
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


def _purge_nested(root: Path, plan: ResetPlan, *, dry_run: bool) -> None:
    for rel in PURGE_DIRS:
        path = root / rel
        if path.exists():
            plan.apply(dry_run, lambda p=path: _unlink(p), f"delete {rel}")


def _reset_knowledge_content(root: Path, plan: ResetPlan, *, dry_run: bool) -> None:
    for name in ROOT_FILES:
        path = root / name
        payload = load_default_for_path(name)

        def write(p: Path = path, d: Any = payload, n: str = name) -> None:
            _write_json(p, d)

        plan.apply(dry_run, write, f"empty {name}")


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
                    f"delete {path.relative_to(root).as_posix()}",
                )
    else:
        _purge_nested(root, plan, dry_run=dry_run)
        _ensure_knowledge(root, plan, dry_run=dry_run)
        _reset_knowledge_content(root, plan, dry_run=dry_run)
        _ensure_skeleton(root, plan, dry_run=dry_run)
        if not dry_run:
            ensure_knowledge_files(root)
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
        help="Only clear Changelog, Masterlog, Calendar, and runtime/ (keep knowledge files)",
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
