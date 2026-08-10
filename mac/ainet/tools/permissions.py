"""Permission checks driven by Rules.txt + Folderrules.json."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from ainet.tools.paths import DbPaths, normalize_relpath


class PermissionError_(PermissionError):
    """Raised when an AI tool action is denied."""


class Action(str, Enum):
    READ = "read"
    WRITE = "write"
    CREATE_FILE = "create_file"
    CREATE_DIR = "create_dir"
    DELETE = "delete"
    MOVE = "move"
    UPDATE_FOLDERRULES = "update_folderrules"


PROTECTED_READ_ONLY = {"rules.txt"}
CODE_ONLY_WRITE = {"calendar.json"}
APPEND_ONLY = {"changelog.json"}


@dataclass
class FolderRules:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> FolderRules:
        with path.open("r", encoding="utf-8") as fh:
            return cls(raw=json.load(fh))

    @property
    def domains(self) -> list[str]:
        return list(self.raw.get("domains", []))

    @property
    def immutable_roots(self) -> set[str]:
        return {name.casefold() for name in self.raw.get("immutable_roots", [])}

    @property
    def protected_paths(self) -> set[str]:
        return set(self.raw.get("protected_paths", []))

    @property
    def max_json_bytes(self) -> int:
        return int(self.raw.get("limits", {}).get("max_json_bytes", 524288))

    @property
    def max_path_depth(self) -> int:
        return int(self.raw.get("limits", {}).get("max_path_depth", 12))

    @property
    def max_children(self) -> int:
        return int(self.raw.get("limits", {}).get("max_children_per_folder", 200))

    @property
    def folder_pattern(self) -> re.Pattern[str]:
        return re.compile(self.raw["naming"]["folder"])

    @property
    def json_pattern(self) -> re.Pattern[str]:
        return re.compile(self.raw["naming"]["json_file"])

    @property
    def forbidden_names(self) -> set[str]:
        return {n.casefold() for n in self.raw.get("naming", {}).get("forbidden_names", [])}

    def create_allowed_prefixes(self) -> list[str]:
        return list(self.raw.get("ai_may_create", {}).get("under", []))

    def nested_under_project_allowed(self) -> bool:
        return bool(self.raw.get("ai_may_create", {}).get("nested_under_project", False))

    def cop_template(self, kind: str) -> list[str]:
        return list(self.raw.get("cop_templates", {}).get(kind, []))


class Permissions:
    def __init__(self, paths: DbPaths) -> None:
        self.paths = paths
        self.reload()

    def reload(self) -> None:
        rules_path = self.paths.resolve("Folderrules.json", must_exist=True)
        self.rules = FolderRules.load(rules_path)

    def assert_name_ok(self, name: str, *, kind: str) -> None:
        if name.casefold() in self.rules.forbidden_names or name.startswith("."):
            raise PermissionError_(f"Forbidden name: {name}")
        if kind == "folder" and not self.rules.folder_pattern.match(name):
            raise PermissionError_(f"Folder name rejected by Folderrules: {name}")
        if kind == "json" and not self.rules.json_pattern.match(name):
            raise PermissionError_(f"JSON file name rejected by Folderrules: {name}")

    def assert_depth_ok(self, relative: str) -> None:
        depth = 0 if relative in ("", ".") else len(PurePosixPath(relative).parts)
        if depth > self.rules.max_path_depth:
            raise PermissionError_(
                f"Path depth {depth} exceeds max_path_depth={self.rules.max_path_depth}"
            )

    def assert_can(self, action: Action, relative: str) -> None:
        rel = normalize_relpath(relative)
        if rel == ".":
            if action in {Action.DELETE, Action.MOVE, Action.WRITE}:
                raise PermissionError_("Cannot mutate the database root.")
            return

        self.assert_depth_ok(rel)
        parts = PurePosixPath(rel).parts
        top = parts[0]
        key = rel.casefold()

        if key in PROTECTED_READ_ONLY:
            if action != Action.READ:
                raise PermissionError_("Rules.txt is read-only for the AI.")
            return

        if key in CODE_ONLY_WRITE:
            if action != Action.READ:
                raise PermissionError_("Calendar.json is mutable by code only.")
            return

        if key in APPEND_ONLY:
            if action == Action.READ:
                return
            if action == Action.WRITE:
                raise PermissionError_(
                    "Changelog.json is append-only; use the changelog tool."
                )
            if action in {Action.DELETE, Action.MOVE, Action.CREATE_FILE, Action.CREATE_DIR}:
                raise PermissionError_("Changelog.json cannot be restructured by the AI.")
            return

        if key == "folderrules.json":
            if action == Action.READ:
                return
            if action in {Action.WRITE, Action.UPDATE_FOLDERRULES}:
                return
            raise PermissionError_("Folderrules.json may only be read or rewritten carefully.")

        if top.casefold() in self.rules.immutable_roots and len(parts) == 1:
            if action in {Action.DELETE, Action.MOVE}:
                raise PermissionError_(f"Domain root is immutable: {top}")

        if action in {Action.CREATE_DIR, Action.CREATE_FILE}:
            self._assert_create_location(rel, action)

    def _assert_create_location(self, relative: str, action: Action) -> None:
        posix = PurePosixPath(relative)
        parent = posix.parent.as_posix()
        if parent == ".":
            raise PermissionError_("AI cannot create new top-level entries.")

        name = posix.name
        self.assert_name_ok(name, kind="json" if action == Action.CREATE_FILE else "folder")

        for prefix in self.rules.create_allowed_prefixes():
            if relative == prefix or relative.startswith(prefix + "/"):
                return
            if parent == prefix or parent.startswith(prefix + "/"):
                return

        domains = {d.casefold() for d in self.rules.domains}
        if parent.casefold() in domains and action == Action.CREATE_FILE:
            return

        parts = posix.parts
        if (
            self.rules.nested_under_project_allowed()
            and len(parts) >= 3
            and parts[0].casefold() == "work"
            and parts[1].casefold() == "projects"
        ):
            return

        if (
            len(parts) >= 3
            and parts[0].casefold() == "school"
            and parts[1].casefold() == "courses"
        ):
            return

        # Full personal tree under Hayden/ may grow freely (not the domain root itself).
        if parts[0].casefold() == "hayden" and len(parts) >= 2:
            return

        raise PermissionError_(
            f"Create not allowed at '{relative}'. Update Folderrules.json if this location should be writable."
        )

    def assert_parent_capacity(self, parent: Path) -> None:
        if not parent.is_dir():
            return
        count = sum(1 for child in parent.iterdir() if not child.name.startswith("."))
        if count >= self.rules.max_children:
            raise PermissionError_(
                f"Folder child limit reached ({self.rules.max_children}): {self.paths.relative_of(parent)}"
            )
