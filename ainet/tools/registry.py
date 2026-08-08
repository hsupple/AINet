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
    {
        "type": "function",
        "function": {
            "name": "capture_inbox",
            "description": (
                "Append a lasting but unsorted scrap to Hayden/Inbox/Captures.json. "
                "Use when info matters but has no clear home yet. Do not use for ephemeral chatter."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "suggested_home": {
                        "type": "string",
                        "description": "Optional guess like Preferences/Food.json or Relationships/People/Jake.json",
                    },
                    "source": {"type": "string", "default": "conversation"},
                    "summary": {"type": "string"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_read_stale",
            "description": (
                "Append to the nearest folder Read.json read_changelog and set needs_update=true. "
                "Mutating writers usually do this automatically; call explicitly if needed after filing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "folder_or_path": {
                        "type": "string",
                        "description": "Folder or file path whose nearest Read should be marked stale",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Short note of what changed (kept in read_changelog)",
                    },
                    "source_path": {
                        "type": "string",
                        "description": "Optional explicit source path for the changelog entry",
                    },
                },
                "required": ["folder_or_path", "summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_read_refreshed",
            "description": (
                "After a successful Read.json rewrite: set needs_update=false and mark pending "
                "read_changelog entries consumed. Pass the Read.json path or its folder."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "read_path": {
                        "type": "string",
                        "description": "Read.json path or containing folder",
                    }
                },
                "required": ["read_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_stale_reads",
            "description": (
                "List Read.json paths with needs_update=true or pending read_changelog entries "
                "(SOI Phase 2 refresh candidates)."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tools",
            "description": (
                "List the full AINet database tool catalog (all operational tools). "
                "Call this when you need a tool that is not in your current lean set; "
                "after calling, the full catalog is unlocked for later tool calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "detail": {
                        "type": "boolean",
                        "default": True,
                        "description": "If true, include parameter schemas.",
                    }
                },
            },
        },
    },
]


def catalog_tools(
    *,
    detail: bool = True,
    include_meta: bool = False,
    read_only: bool = False,
) -> dict[str, Any]:
    """Return the tool catalog.

    read_only=True → OAC view (list/tree/read only).
    Otherwise → all operational DB tools (excludes get_tools unless include_meta).
    """
    read_names = {"list_dir", "tree", "read_text", "read_json"}
    tools = []
    for spec in TOOL_DEFINITIONS:
        fn = spec.get("function") or {}
        name = fn.get("name")
        if not name or (name == "get_tools" and not include_meta):
            continue
        if read_only and name not in read_names:
            continue
        item: dict[str, Any] = {
            "name": name,
            "description": fn.get("description", ""),
        }
        if detail:
            item["parameters"] = fn.get("parameters") or {}
        tools.append(item)
    return {
        "ok": True,
        "count": len(tools),
        "read_only": read_only,
        "tools": tools,
        "unlocks_full_access": not read_only,
        "note": (
            "Read-only catalog for OAC (no mutations)."
            if read_only
            else "Full tool access unlocked for subsequent calls this session."
        ),
    }


def tools_subset(names: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
    """Return Ollama tool defs; None means the full catalog. Always includes get_tools when filtered."""
    if names is None:
        return list(TOOL_DEFINITIONS)
    wanted = set(names)
    wanted.add("get_tools")
    return [t for t in TOOL_DEFINITIONS if t.get("function", {}).get("name") in wanted]


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
        "capture_inbox": lambda **kw: db.capture_inbox(
            kw["text"],
            tags=kw.get("tags"),
            suggested_home=kw.get("suggested_home", ""),
            source=kw.get("source", "conversation"),
            summary=kw.get("summary"),
        ),
        "mark_read_stale": lambda **kw: db.mark_read_stale(
            kw["folder_or_path"],
            kw["summary"],
            source_path=kw.get("source_path"),
        ),
        "mark_read_refreshed": lambda **kw: db.mark_read_refreshed(kw["read_path"]),
        "list_stale_reads": lambda **kw: db.list_stale_reads(),
        "get_tools": lambda **kw: catalog_tools(
            detail=bool(kw.get("detail", True)),
            read_only=bool(kw.get("read_only", False)),
        ),
        # camelCase alias some models emit
        "getTools": lambda **kw: catalog_tools(
            detail=bool(kw.get("detail", True)),
            read_only=bool(kw.get("read_only", False)),
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
