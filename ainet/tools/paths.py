"""Path sandboxing under the database root (POSIX + Windows)."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath


class PathError(ValueError):
    """Raised when a path escapes the database root or is otherwise invalid."""


_DRIVE_RE = re.compile(r"^[A-Za-z]:$")
_RESERVED_WIN = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_BAD_CHARS = set('<>:"|?*')


def normalize_relpath(relative: str | Path) -> str:
    """Normalize a DB-relative path to forward-slash form without traversal."""
    text = str(relative).strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    if text in ("", "."):
        return "."
    if text.startswith("/") or text.startswith("~"):
        raise PathError(f"Absolute paths are not allowed: {relative}")
    if text.startswith("//") or text.startswith("\\\\"):
        raise PathError(f"UNC paths are not allowed: {relative}")

    parts: list[str] = []
    for part in PurePosixPath(text).parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise PathError(f"Parent traversal is not allowed: {relative}")
        if _DRIVE_RE.match(part) or (len(part) >= 2 and part[1] == ":"):
            raise PathError(f"Drive-letter paths are not allowed: {relative}")
        _assert_component_ok(part)
        parts.append(part)
    return "/".join(parts) if parts else "."


def _assert_component_ok(part: str) -> None:
    stem = part.split(".")[0].upper()
    if stem in _RESERVED_WIN:
        raise PathError(f"Windows reserved name not allowed: {part}")
    if part.endswith(" ") or part.endswith("."):
        raise PathError(f"Name cannot end with space or dot on Windows: {part!r}")
    if any(ch in _BAD_CHARS or ord(ch) < 32 for ch in part):
        raise PathError(f"Name contains characters illegal on Windows: {part!r}")


class DbPaths:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise PathError(f"Database root does not exist: {self.root}")

    def resolve(self, relative: str | Path, *, must_exist: bool = False) -> Path:
        rel = normalize_relpath(relative)
        # Build with PurePosixPath parts so Windows path separators stay correct.
        target = self.root.joinpath(*([] if rel == "." else rel.split("/"))).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise PathError(f"Path escapes database root: {relative}") from exc
        if must_exist and not target.exists():
            raise PathError(f"Path does not exist: {rel}")
        return target

    def relative_of(self, absolute: Path) -> str:
        abs_resolved = absolute.resolve()
        try:
            return abs_resolved.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise PathError(f"Path outside database root: {absolute}") from exc

    def norm(self, relative: str | Path) -> str:
        return normalize_relpath(relative)
