"""High-level database operations for the AI tool layer."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from ainet.defaults import load_default, load_default_for_path
from ainet.tools import changelog, readlog
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
            return self._list_read_index()
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

    def _list_read_index(self) -> dict[str, Any]:
        """Root listing is the whole-database index: every folder's Read.json."""
        reads: list[str] = []
        for candidate in sorted(self.paths.root.rglob("Read.json")):
            parts = candidate.relative_to(self.paths.root).parts
            if any(p.startswith(".") for p in parts):
                continue
            if any(p.casefold() in {"runtime", "chats"} for p in parts):
                continue
            reads.append(self.paths.relative_of(candidate))
        return {
            "ok": True,
            "path": ".",
            "read_files": reads,
            "hint": (
                "Every Read.json in the database. Pick the folder that matches the "
                "request and read that file."
            ),
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
        read_touch = self._maybe_mark_read_stale(
            path, summary=summary or ("Updated text" if exists else "Created text")
        )
        return {
            "ok": True,
            "path": path,
            "created": not exists,
            "chars": len(text),
            "changelog": entry,
            "read_stale": read_touch,
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
        return {
            "ok": True,
            "filename": Path(filename).name,
            "data": load_default(filename),
        }

    def capture_inbox(
        self,
        text: str,
        *,
        tags: list[str] | None = None,
        suggested_home: str = "",
        source: str = "conversation",
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Append one unsorted scrap to Hayden/Inbox/Captures.json."""
        from datetime import datetime, timezone
        from uuid import uuid4

        path = "Hayden/Inbox/Captures.json"
        cleaned = (text or "").strip()
        if not cleaned:
            raise ValueError("capture_inbox requires non-empty text")

        if not self.paths.resolve(path).exists():
            self.write_json(
                path,
                load_default_for_path(path),
                create=True,
                summary="Seed Inbox Captures.json",
            )

        current = self.read_json(path)["data"]
        if not isinstance(current, dict):
            raise ValueError("Captures.json must be a JSON object")
        captures = current.setdefault("captures", [])
        if not isinstance(captures, list):
            raise ValueError("Captures.json 'captures' must be a list")

        entry = {
            "id": uuid4().hex[:12],
            "text": cleaned,
            "captured_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "source": source or "conversation",
            "tags": list(tags or []),
            "suggested_home": suggested_home or "",
            "status": "unfiled",
            "filed_to": "",
        }
        captures.append(entry)
        current["last_updated"] = entry["captured_at"]
        self.write_json(
            path,
            current,
            create=False,
            summary=summary or f"Inbox capture: {cleaned[:80]}",
        )
        return {"ok": True, "path": path, "capture": entry}

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

        trim_notes: list[str] = []
        if readlog.is_read_json_path(norm):
            data, trim_notes = readlog.prepare_read_payload(data)

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
        read_touch = self._maybe_mark_read_stale(
            path, summary=summary or ("Updated JSON" if exists else "Created JSON")
        )
        result: dict[str, Any] = {
            "ok": True,
            "path": path,
            "created": not exists,
            "changelog": entry,
            "read_stale": read_touch,
        }
        if trim_notes:
            result["read_trimmed"] = trim_notes
        return result

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
        trim_notes: list[str] = []
        if readlog.is_read_json_path(path):
            merged, trim_notes = readlog.prepare_read_payload(merged)
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
        read_touch = self._maybe_mark_read_stale(path, summary=summary or "Patched JSON")
        result = {
            "ok": True,
            "path": path,
            "data": merged,
            "changelog": entry,
            "read_stale": read_touch,
        }
        if trim_notes:
            result["read_trimmed"] = trim_notes
        return result

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

        trim_notes: list[str] = []
        if readlog.is_read_json_path(path):
            if not isinstance(current, dict):
                raise ValueError("Read.json must be a JSON object")
            current, trim_notes = readlog.prepare_read_payload(current)

        payload = self._encode_json(current)
        atomic_write_bytes(target, payload)
        entry = changelog.append_entry(
            self.paths,
            action="set_json_path",
            path=path,
            summary=summary or f"Set {json_path}",
            details={"json_path": json_path},
        )
        read_touch = self._maybe_mark_read_stale(path, summary=summary or f"Set {json_path}")
        result = {
            "ok": True,
            "path": path,
            "json_path": json_path,
            "changelog": entry,
            "read_stale": read_touch,
        }
        if trim_notes:
            result["read_trimmed"] = trim_notes
        return result

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
        # Mark nearest parent Read (new empty folders usually have no own Read yet).
        read_touch = self._maybe_mark_read_stale(
            path, summary=summary or "Created folder", folder_hint=True
        )
        return {
            "ok": True,
            "path": path,
            "created": True,
            "changelog": entry,
            "read_stale": read_touch,
        }

    def create_cop(
        self,
        path: str,
        kind: str,
        *,
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Create a course/project COP from Folderrules templates."""
        template = self.permissions.rules.cop_template(kind)
        if not template:
            raise PermissionError_(f"Unknown COP kind: {kind}")

        with changelog.defer_masterlog(self.paths):
            created = self.create_folder(path, summary=summary or f"Created {kind} COP root")
            made: list[str] = []
            root = normalize_relpath(path)
            for item in template:
                child = f"{root}/{item}".replace("//", "/")
                if item.endswith("/"):
                    if self.paths.resolve(child.rstrip("/")).exists():
                        made.append(child.rstrip("/"))
                        continue
                    result = self.create_folder(child.rstrip("/"), summary=f"COP folder {item}")
                else:
                    if self.paths.resolve(child).exists():
                        made.append(child)
                        continue
                    result = self.write_json(
                        child,
                        load_default_for_path(child),
                        create=True,
                        summary=f"COP file {item}",
                    )
                made.append(result["path"])
        return {
            "ok": True,
            "path": path,
            "kind": kind,
            "created_root": created.get("created", False),
            "created": made,
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
        note = summary or f"Moved {src} -> {dest}"
        read_touch = [
            t
            for t in (
                self._maybe_mark_read_stale(src, summary=note),
                self._maybe_mark_read_stale(dest, summary=note),
            )
            if t
        ]
        return {
            "ok": True,
            "from": src,
            "to": dest,
            "changelog": entry,
            "read_stale": read_touch or None,
        }

    def archive_to_history(
        self,
        path: str,
        *,
        history_dir: str | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Move a file/folder into the nearest or provided History/ folder."""
        source = self.paths.resolve(path, must_exist=True)
        if history_dir:
            hist_rel = normalize_relpath(history_dir)
        else:
            hist_rel = _nearest_history(self.paths, path)
        self.create_folder(hist_rel, summary="Ensure History folder")
        dest_rel = f"{hist_rel}/{source.name}"
        return self.move_path(
            path,
            dest_rel,
            summary=summary or f"Archived {path} -> {dest_rel}",
        )

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

    def mark_read_stale(
        self,
        folder_or_path: str,
        summary: str,
        *,
        source_path: str | None = None,
    ) -> dict[str, Any]:
        """Append to the nearest Read.json read_changelog and set needs_update=true.

        Call after lasting content lands in a folder (SOI filing, captures, etc.).
        Mutating writers already invoke this automatically for non-Read paths.
        """
        read_path = readlog.find_nearest_read_path(self.paths, folder_or_path)
        if not read_path:
            return {
                "ok": True,
                "marked": False,
                "reason": "no Read.json found for path",
                "path": folder_or_path,
            }
        self.permissions.assert_can(Action.WRITE, read_path)
        data = readlog.load_read_doc(self.paths, read_path)
        entry = readlog.append_read_log_entry(
            data,
            summary=summary,
            source_path=source_path if source_path is not None else normalize_relpath(folder_or_path),
        )
        readlog.save_read_doc(self.paths, read_path, data)
        return {
            "ok": True,
            "marked": True,
            "read_path": read_path,
            "entry": entry,
            "needs_update": True,
        }

    def mark_read_refreshed(self, read_path: str) -> dict[str, Any]:
        """After a successful Read rewrite: needs_update=false, consume pending log entries."""
        norm = normalize_relpath(read_path)
        if not readlog.is_read_json_path(norm):
            candidate = f"{norm.rstrip('/')}/Read.json"
            if self.paths.resolve(candidate).is_file():
                norm = candidate
            else:
                raise PathError(f"No Read.json for: {read_path}")
        self.permissions.assert_can(Action.WRITE, norm)
        data = readlog.load_read_doc(self.paths, norm)
        consumed = readlog.consume_read_log(data)
        readlog.save_read_doc(self.paths, norm, data)
        return {
            "ok": True,
            "read_path": norm,
            "needs_update": False,
            "consumed": consumed,
        }

    def refresh_read(self, read_path: str, digest: dict[str, Any]) -> dict[str, Any]:
        """Write a compact Read digest and consume its pending freshness log."""
        norm = normalize_relpath(read_path)
        if not readlog.is_read_json_path(norm):
            candidate = f"{norm.rstrip('/')}/Read.json"
            if self.paths.resolve(candidate).is_file():
                norm = candidate
            else:
                raise PathError(f"No Read.json for: {read_path}")
        if not isinstance(digest, dict):
            raise ValueError("refresh_read requires a digest object.")

        list_fields = (
            "important_context",
            "recent_changes",
            "active_items",
            "known_facts",
            "uncertainties",
        )
        for key in list_fields:
            if key in digest and not isinstance(digest[key], list):
                raise ValueError(f"refresh_read digest field '{key}' must be an array.")
        patch = {
            "summary": str(digest.get("summary") or "").strip(),
            "state": str(digest.get("state") or "").strip(),
            **{key: list(digest.get(key) or []) for key in list_fields},
            "last_updated": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        if not patch["summary"] and not any(patch[key] for key in list_fields):
            raise ValueError("refresh_read digest cannot be entirely empty.")

        written = self.patch_json(norm, patch, summary="Refresh compact Read.json digest")
        marked = self.mark_read_refreshed(norm)
        return {
            "ok": True,
            "read_path": norm,
            "updated_fields": sorted(patch),
            "consumed": marked.get("consumed", 0),
            "needs_update": False,
            "write": written,
        }

    def list_stale_reads(self) -> dict[str, Any]:
        """Discover Read.json paths with needs_update or pending read_changelog entries."""
        root = self.paths.root
        stale: list[str] = []
        for path in sorted(root.rglob("Read.json")):
            if "runtime" in path.parts or "Chats" in path.parts:
                continue
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                continue
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            if readlog.read_needs_refresh(data):
                stale.append(rel)
        return {"ok": True, "count": len(stale), "paths": stale}

    # ---- internals ---------------------------------------------------------

    def _maybe_mark_read_stale(
        self,
        path: str,
        *,
        summary: str,
        folder_hint: bool = False,
    ) -> dict[str, Any] | None:
        if readlog.should_skip_stale_mark(path):
            return None
        # For brand-new folders, start search from parent so we don't require a Read inside.
        look = path
        if folder_hint:
            parent = str(PurePosixPath(normalize_relpath(path)).parent)
            if parent and parent != ".":
                look = parent
        try:
            result = self.mark_read_stale(look, summary, source_path=normalize_relpath(path))
        except (PermissionError_, PathError, ValueError, OSError):
            return None
        return result if result.get("marked") else None

    def _encode_json(self, data: Any) -> bytes:
        try:
            text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Data is not JSON-serializable: {exc}") from exc
        payload = text.encode("utf-8")
        limit = self.permissions.rules.max_json_bytes
        if len(payload) > limit:
            raise PermissionError_(f"JSON exceeds max_json_bytes ({limit})")
        # Round-trip validate
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


def _nearest_history(paths: DbPaths, relative: str) -> str:
    parts = list(PurePosixPath(normalize_relpath(relative)).parts)
    # Walk up looking for a sibling History or parent History.
    for i in range(len(parts) - 1, 0, -1):
        parent = "/".join(parts[:i])
        candidate = f"{parent}/History"
        abs_candidate = paths.resolve(candidate)
        if abs_candidate.is_dir() or i <= 2:
            return candidate
    top = parts[0] if parts else "."
    return f"{top}/History"
