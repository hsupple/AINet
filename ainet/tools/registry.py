"""Ollama-compatible tool definitions and dispatcher."""

from __future__ import annotations

from typing import Any, Callable

from ainet.tools.ops import DatabaseTools
from ainet.tools.paths import PathError
from ainet.tools.permissions import PermissionError_


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List immediate children of a database folder (relative to db/).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative folder path. Use '.' for db root.",
                        "default": ".",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tree",
            "description": "Return a truncated directory tree under a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "max_depth": {"type": "integer", "default": 3},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_text",
            "description": "Read a text file such as Rules.txt.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_json",
            "description": "Read and parse a JSON file from the database.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_json",
            "description": (
                "Write a full JSON document. Set create=true to create a new file. "
                "Prefer patch_json for partial updates. For new files from templates, prefer create_json."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "data": {"description": "JSON-serializable object/array/value."},
                    "create": {"type": "boolean", "default": False},
                    "summary": {"type": "string"},
                },
                "required": ["path", "data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_json",
            "description": (
                "Create a new JSON file. If data is omitted, uses the matching template from "
                "ainet/defaults/ (Profile.json, Read.json, Plan.json, etc., else generic.json)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "data": {
                        "description": "Optional JSON payload. Omit to use the default template."
                    },
                    "summary": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_default_template",
            "description": "Inspect the default JSON template that would seed a new file of this name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Template filename, e.g. Profile.json or Read.json",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch_json",
            "description": "Deep-merge a patch object into an existing JSON object file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "patch": {"type": "object"},
                    "summary": {"type": "string"},
                },
                "required": ["path", "patch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_json_path",
            "description": "Set a dotted key path inside an existing JSON object (e.g. 'goals.0').",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "json_path": {"type": "string"},
                    "value": {},
                    "summary": {"type": "string"},
                },
                "required": ["path", "json_path", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_folder",
            "description": "Create a folder under an allowed location (see Folderrules.json).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_cop",
            "description": (
                "Create a course or project context-of-purpose folder from Folderrules templates. "
                "kind must be 'course' or 'project'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "kind": {"type": "string", "enum": ["course", "project"]},
                    "summary": {"type": "string"},
                },
                "required": ["path", "kind"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_path",
            "description": "Move/rename a file or folder inside the database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "src": {"type": "string"},
                    "dest": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["src", "dest"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archive_to_history",
            "description": "Move a path into History/ (nearest domain History, or an explicit history_dir).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "history_dir": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_changelog",
            "description": "Manually append a changelog entry (most tools already log automatically).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "path": {"type": "string"},
                    "summary": {"type": "string"},
                    "details": {"type": "object"},
                },
                "required": ["action", "path", "summary"],
            },
        },
    },
]


def _handlers(db: DatabaseTools) -> dict[str, Callable[..., dict[str, Any]]]:
    return {
        "list_dir": lambda **kw: db.list_dir(kw.get("path", ".")),
        "tree": lambda **kw: db.tree(kw.get("path", "."), int(kw.get("max_depth", 3))),
        "read_text": lambda **kw: db.read_text(kw["path"]),
        "read_json": lambda **kw: db.read_json(kw["path"]),
        "write_json": lambda **kw: db.write_json(
            kw["path"],
            kw["data"],
            create=bool(kw.get("create", False)),
            summary=kw.get("summary"),
        ),
        "create_json": lambda **kw: db.create_json(
            kw["path"],
            kw.get("data"),
            summary=kw.get("summary"),
        ),
        "get_default_template": lambda **kw: db.get_default_template(kw["filename"]),
        "patch_json": lambda **kw: db.patch_json(
            kw["path"], kw["patch"], summary=kw.get("summary")
        ),
        "set_json_path": lambda **kw: db.set_json_path(
            kw["path"], kw["json_path"], kw["value"], summary=kw.get("summary")
        ),
        "create_folder": lambda **kw: db.create_folder(kw["path"], summary=kw.get("summary")),
        "create_cop": lambda **kw: db.create_cop(
            kw["path"], kw["kind"], summary=kw.get("summary")
        ),
        "move_path": lambda **kw: db.move_path(
            kw["src"], kw["dest"], summary=kw.get("summary")
        ),
        "archive_to_history": lambda **kw: db.archive_to_history(
            kw["path"], history_dir=kw.get("history_dir"), summary=kw.get("summary")
        ),
        "append_changelog": lambda **kw: db.append_changelog(
            kw["action"], kw["path"], kw["summary"], kw.get("details")
        ),
    }


def dispatch(db: DatabaseTools, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a named tool and always return a JSON-serializable result dict."""
    args = arguments or {}
    handlers = _handlers(db)
    if name not in handlers:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        return handlers[name](**args)
    except (PermissionError_, PathError, ValueError, KeyError, TypeError, OSError) as exc:
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
