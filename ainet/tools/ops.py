"""High-level database operations for the AI tool layer."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ainet.defaults import load_default, load_default_for_path
from ainet.tools import changelog
from ainet.tools.fsutil import atomic_write_bytes
from ainet.tools.paths import DbPaths, PathError, normalize_relpath
from ainet.tools.permissions import Action, PermissionError_, Permissions


class DatabaseTools:
    """Sandboxed create / read / write / mkdir / move / archive tools."""

    def __init__(self, root: Path | str) -> None:
        self.paths = DbPaths(root)
        self.permissions = Permissions(self.paths)

    # ---- reads -------------------------------------------------------------

    def read_text(self, path: str) -> dict[str, Any]:
        self.permissions.assert_can(Action.READ, path)
        target = self.paths.resolve(path, must_exist=True)
        if not target.is_file():
            raise PathError(f"Not a file: {path}")
        return {
            "ok": True,
            "path": path,
            "content": target.read_text(encoding="utf-8"),
        }

    def read_json(self, path: str) -> dict[str, Any]:
        self.permissions.assert_can(Action.READ, path)
        target = self.paths.resolve(path, must_exist=True)
        if not target.is_file():
            raise PathError(f"Not a file: {path}")
        try:
            with target.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"File is not valid JSON: {path} ({exc})") from exc
        return {"ok": True, "path": path, "data": data}

    def list_dir(self, path: str = ".") -> dict[str, Any]:
        self.permissions.assert_can(Action.READ, path)
        target = self.paths.resolve(path, must_exist=True)
        if not target.is_dir():
            raise PathError(f"Not a directory: {path}")
        if target == self.paths.root:
            return self._list_knowledge_index()
        children = []
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith("."):
                continue
            # Hide host-only runtime / chats from AI listings (SOI + OAC).
            if child.name.casefold() in {"runtime", "chats"}:
                continue
            children.append(
                {
                    "name": child.name,
                    "type": "dir" if child.is_dir() else "file",
                    "path": self.paths.relative_of(child),
                }
            )
        return {"ok": True, "path": path, "children": children}

    def _list_knowledge_index(self) -> dict[str, Any]:
        """Root listing is the knowledge files OAC should read."""
        from ainet.logstore import knowledge_file_names, root_listing_hint

        files: list[str] = []
        for name in knowledge_file_names():
            if (self.paths.root / name).is_file():
                files.append(name)
        projects: list[str] = []
        root = self.paths.root / "Projects"
        if root.is_dir():
            for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
                if child.is_dir() and not child.name.startswith("."):
                    projects.append(f"Projects/{child.name}/project.json")
        return {
            "ok": True,
            "path": ".",
            "files": files,
            "projects": projects,
            "hint": root_listing_hint(),
        }

    def tree(self, path: str = ".", max_depth: int = 3) -> dict[str, Any]:
        self.permissions.assert_can(Action.READ, path)
        if max_depth < 0 or max_depth > 8:
            raise PermissionError_("max_depth must be between 0 and 8")
        target = self.paths.resolve(path, must_exist=True)
        if not target.is_dir():
            raise PathError(f"Not a directory: {path}")

        def walk(node: Path, depth: int) -> dict[str, Any]:
            item: dict[str, Any] = {
                "name": node.name if node != self.paths.root else ".",
                "type": "dir",
                "path": self.paths.relative_of(node) if node != self.paths.root else ".",
            }
            if depth >= max_depth:
                item["truncated"] = True
                return item
            kids = []
            for child in sorted(node.iterdir(), key=lambda p: p.name.lower()):
                if child.name.startswith("."):
                    continue
                if child.name.casefold() in {"runtime", "chats"}:
                    continue
                if child.is_dir():
                    kids.append(walk(child, depth + 1))
                else:
                    kids.append(
                        {
                            "name": child.name,
                            "type": "file",
                            "path": self.paths.relative_of(child),
                        }
                    )
            item["children"] = kids
            return item

        return {"ok": True, "tree": walk(target, 0)}

    def write_text(
        self,
        path: str,
        content: str,
        *,
        create: bool = True,
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Write a UTF-8 text document (.txt/.md/etc.)."""
        from ainet.tools.fsutil import atomic_write_text

        exists = self.paths.resolve(path).exists()
        if exists:
            self.permissions.assert_can(Action.WRITE, path)
        else:
            if not create:
                raise PathError(f"File does not exist (pass create=true): {path}")
            self.permissions.assert_can(Action.CREATE_FILE, path)

        target = self.paths.resolve(path)
        if not exists:
            target.parent.mkdir(parents=True, exist_ok=True)
            self.permissions.assert_parent_capacity(target.parent)
            kind = "json" if target.name.lower().endswith(".json") else "file"
            self.permissions.assert_name_ok(target.name, kind=kind)

        text = content if isinstance(content, str) else str(content or "")
        atomic_write_text(target, text)
        entry = changelog.append_entry(
            self.paths,
            action="write_text" if exists else "create_text",
            path=path,
            summary=summary or ("Updated text" if exists else "Created text"),
            details={"chars": len(text)},
        )
        return {
            "ok": True,
            "path": path,
            "created": not exists,
            "chars": len(text),
            "changelog": entry,
        }

    def create_json(
        self,
        path: str,
        data: Any | None = None,
        *,
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Create a new JSON file from an explicit payload or the matching default template."""
        if self.paths.resolve(path).exists():
            raise PathError(f"File already exists: {path}")
        payload = load_default_for_path(path) if data is None else data
        return self.write_json(
            path,
            payload,
            create=True,
            summary=summary or f"Created JSON from {'template' if data is None else 'payload'}",
        )

    def get_default_template(self, filename: str) -> dict[str, Any]:
        """Return the default template payload that would be used for a new file."""
        return {"ok": True, "filename": Path(filename).name, "data": load_default(filename)}

    # ---- writes ------------------------------------------------------------

    def write_json(
        self,
        path: str,
        data: Any,
        *,
        create: bool = False,
        summary: str | None = None,
    ) -> dict[str, Any]:
        exists = self.paths.resolve(path).exists()
        if exists:
            self.permissions.assert_can(Action.WRITE, path)
        else:
            if not create:
                raise PathError(f"File does not exist (pass create=true): {path}")
            self.permissions.assert_can(Action.CREATE_FILE, path)

        norm = normalize_relpath(path)
        if norm.casefold() == "folderrules.json":
            self.permissions.assert_can(Action.UPDATE_FOLDERRULES, path)

        payload = self._encode_json(data)
        target = self.paths.resolve(path)
        if not exists:
            target.parent.mkdir(parents=True, exist_ok=True)
            self.permissions.assert_parent_capacity(target.parent)
            self.permissions.assert_name_ok(target.name, kind="json")

        atomic_write_bytes(target, payload)
        if norm.casefold() == "folderrules.json":
            self.permissions.reload()

        entry = changelog.append_entry(
            self.paths,
            action="write_json" if exists else "create_json",
            path=path,
            summary=summary or ("Updated JSON" if exists else "Created JSON"),
        )
        return {
            "ok": True,
            "path": path,
            "created": not exists,
            "changelog": entry,
        }

    def patch_json(
        self,
        path: str,
        patch: dict[str, Any],
        *,
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Shallow-or-deep merge patch into an existing JSON object."""
        self.permissions.assert_can(Action.WRITE, path)
        target = self.paths.resolve(path, must_exist=True)
        with target.open("r", encoding="utf-8") as fh:
            current = json.load(fh)
        if not isinstance(current, dict) or not isinstance(patch, dict):
            raise ValueError("patch_json requires the file and patch to be JSON objects.")

        merged = _deep_merge(current, patch)
        payload = self._encode_json(merged)
        atomic_write_bytes(target, payload)
        if normalize_relpath(path).casefold() == "folderrules.json":
            self.permissions.reload()

        entry = changelog.append_entry(
            self.paths,
            action="patch_json",
            path=path,
            summary=summary or "Patched JSON",
            details={"keys": sorted(patch.keys())},
        )
        return {
            "ok": True,
            "path": path,
            "data": merged,
            "changelog": entry,
        }

    def set_json_path(
        self,
        path: str,
        json_path: str,
        value: Any,
        *,
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Set a dotted path inside a JSON object, creating intermediate objects."""
        self.permissions.assert_can(Action.WRITE, path)
        target = self.paths.resolve(path, must_exist=True)
        with target.open("r", encoding="utf-8") as fh:
            current = json.load(fh)

        keys = [k for k in json_path.split(".") if k]
        if not keys:
            raise ValueError("json_path must not be empty")
        cursor: Any = current
        for key in keys[:-1]:
            if not isinstance(cursor, dict):
                raise ValueError(f"Cannot descend into non-object at '{key}'")
            nxt = cursor.get(key)
            if nxt is None:
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        if not isinstance(cursor, dict):
            raise ValueError("Cannot set key on non-object parent")
        cursor[keys[-1]] = value

        payload = self._encode_json(current)
        atomic_write_bytes(target, payload)
        entry = changelog.append_entry(
            self.paths,
            action="set_json_path",
            path=path,
            summary=summary or f"Set {json_path}",
            details={"json_path": json_path},
        )
        return {
            "ok": True,
            "path": path,
            "json_path": json_path,
            "changelog": entry,
        }

    def create_folder(self, path: str, *, summary: str | None = None) -> dict[str, Any]:
        self.permissions.assert_can(Action.CREATE_DIR, path)
        target = self.paths.resolve(path)
        if target.exists():
            if target.is_dir():
                return {"ok": True, "path": path, "created": False}
            raise PathError(f"A file already exists at: {path}")
        self.permissions.assert_name_ok(target.name, kind="folder")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.permissions.assert_parent_capacity(target.parent)
        target.mkdir()
        entry = changelog.append_entry(
            self.paths,
            action="create_folder",
            path=path,
            summary=summary or "Created folder",
        )
        return {
            "ok": True,
            "path": path,
            "created": True,
            "changelog": entry,
        }

    def move_path(
        self,
        src: str,
        dest: str,
        *,
        summary: str | None = None,
    ) -> dict[str, Any]:
        self.permissions.assert_can(Action.MOVE, src)
        source = self.paths.resolve(src, must_exist=True)
        if source.is_dir():
            self.permissions.assert_can(Action.CREATE_DIR, dest)
        else:
            self.permissions.assert_can(Action.CREATE_FILE, dest)
        destination = self.paths.resolve(dest)
        if destination.exists():
            raise PathError(f"Destination already exists: {dest}")
        if source.is_file():
            kind = "json" if destination.name.lower().endswith(".json") else "file"
            self.permissions.assert_name_ok(destination.name, kind=kind)
        else:
            self.permissions.assert_name_ok(destination.name, kind="folder")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.permissions.assert_parent_capacity(destination.parent)
        shutil.move(str(source), str(destination))
        entry = changelog.append_entry(
            self.paths,
            action="move",
            path=dest,
            summary=summary or f"Moved {src} -> {dest}",
            details={"from": src, "to": dest},
        )
        return {
            "ok": True,
            "from": src,
            "to": dest,
            "changelog": entry,
        }

    def append_changelog(
        self,
        action: str,
        path: str,
        summary: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = changelog.append_entry(
            self.paths,
            action=action,
            path=path,
            summary=summary,
            details=details,
        )
        return {"ok": True, "changelog": entry}

    # ---- internals ---------------------------------------------------------

    def _encode_json(self, data: Any) -> bytes:
        try:
            text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Data is not JSON-serializable: {exc}") from exc
        payload = text.encode("utf-8")
        limit = self.permissions.rules.max_json_bytes
        if len(payload) > limit:
            raise PermissionError_(f"JSON exceeds max_json_bytes ({limit})")
        json.loads(payload)
        return payload


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
