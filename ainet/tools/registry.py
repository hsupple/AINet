"""Ollama-compatible tool definitions and dispatcher."""

from __future__ import annotations

from typing import Any, Callable

from ainet.tools.browser import open_chrome
from ainet.tools.ops import DatabaseTools
from ainet.tools.paths import PathError
from ainet.tools.permissions import PermissionError_
from ainet.tools.web import web_fetch, web_search

# OAC-safe tools (no general DB mutations). Kept in sync with ollama.modes.base.
READ_TOOL_NAMES = frozenset(
    {
        "list_dir",
        "tree",
        "read_text",
        "read_json",
        "web_search",
        "web_fetch",
        "open_chrome",
    }
)
OAC_TOOL_NAMES = READ_TOOL_NAMES



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
                "Create a course or project COP from Folderrules templates "
                "(Profile/Read/Plan/History). "
                "path = COP root (School/Courses/<Code> or Work/Projects/<Name>). "
                "kind = course | project."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "COP root path, e.g. School/Courses/ME365",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["course", "project"],
                        "description": "course → School/Courses; project → Work/Projects",
                    },
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
            "name": "web_search",
            "description": (
                "Search the public internet via Brave Search. Returns concise title/url/snippet "
                "results. The host auto-opens the best result in Chrome unless the user opted out. "
                "Use for external facts; do not invent. Cite titles/urls briefly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "count": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of results (1-8).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch a public http(s) URL and return truncated plain text. "
                "Use after web_search for a deeper dive into one page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute http(s) URL",
                    },
                    "max_chars": {
                        "type": "integer",
                        "default": 4000,
                        "description": "Max characters of plain text to return (capped).",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_chrome",
            "description": (
                "Open an http(s) URL as a new tab in Google Chrome on this PC. "
                "web_search already auto-opens the top hit — use this for extra URLs you cite, "
                "or when the user asks to open/show a page. Call once per URL. "
                "Never claim a tab opened unless this tool ran (or search auto_opened it)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute http(s) URL to open",
                    },
                    "new_tab": {
                        "type": "boolean",
                        "default": True,
                        "description": "Open in a new tab (default true).",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tools",
            "description": (
                "List the AINet tool catalog (DB ops + web_search/web_fetch/open_chrome). "
                "Call this when you need a tool that is not in your current lean set; "
                "after calling, the full catalog is unlocked for later tool calls "
                "(OAC stays read-only + web)."
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
    {
        "type": "function",
        "function": {
            "name": "file_note",
            "description": (
                "Phase 1 filing — preferred for SOI. Pass entry_id, dest (simple label), "
                "and text (a short note YOU write about the turn). Host stores the note in "
                "<folder>/Notes.json with id as evidence, and the raw message + id in History.json. "
                "Call again with the same entry_id and a different dest when the turn belongs "
                "in multiple folders (e.g. Preferences + Pantry). "
                "dest examples: Values, Memories, Psychology, Habits, Desires, Pantry, discard."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "string",
                        "description": "Changelog entry id",
                    },
                    "dest": {
                        "type": "string",
                        "description": (
                            "Folder from the folders list — path (Hayden/School, Hayden/Values) "
                            "or folder name (Values, Pantry). discard for greetings only."
                        ),
                    },
                    "text": {
                        "type": "string",
                        "description": "Short note you write summarizing what to keep (not a raw paste)",
                    },
                    "summary": {"type": "string"},
                },
                "required": ["entry_id", "dest"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_by_id",
            "description": (
                "SOI preferred filing tool. Pass Changelog entry_id(s) or Inbox inbox_id only — "
                "the host copies stored user_text. Do NOT paste turn bodies. "
                "dest: 'identity' | 'voice' | 'psychology' | 'habits' | "
                "'discard' | a leaf path like Hayden/Preferences/Food.json. File by content, not mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "string",
                        "description": "One Changelog.json entry id (e.g. ad7b021d33e644d6)",
                    },
                    "id": {
                        "type": "string",
                        "description": "Alias for entry_id",
                    },
                    "entry_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Several related Changelog ids",
                    },
                    "inbox_id": {
                        "type": "string",
                        "description": "Hayden/Inbox/Captures.json capture id",
                    },
                    "dest": {
                        "type": "string",
                        "description": (
                            "'identity' | 'voice' | 'psychology' | 'habits' | "
                            "'discard' (greetings only) | or a Folderrules JSON leaf. "
                            "Not Inbox. Not a dump for schedules/courses."
                        ),
                    },
                    "summary": {"type": "string"},
                },
                "required": ["dest"],
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

    read_only=True → OAC view (list/tree/read + web).
    Otherwise → all operational tools (excludes get_tools unless include_meta).
    """
    tools = []
    for spec in TOOL_DEFINITIONS:
        fn = spec.get("function") or {}
        name = fn.get("name")
        if not name or (name == "get_tools" and not include_meta):
            continue
        if read_only and name not in OAC_TOOL_NAMES:
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
            "OAC catalog: read + web (no general DB writes)."
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
    from ollama import file_by_id as file_by_id_mod
    from ollama import file_note as file_note_mod

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
            str(kw.get("path") or kw.get("folder_path") or ""),
            str(kw.get("kind") or kw.get("cop_type") or ""),
            summary=kw.get("summary"),
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
        "web_search": lambda **kw: web_search(
            kw["query"],
            count=int(kw.get("count", 5)),
        ),
        "web_fetch": lambda **kw: web_fetch(
            kw["url"],
            max_chars=int(kw.get("max_chars", 4000)),
        ),
        "open_chrome": lambda **kw: open_chrome(
            kw["url"],
            new_tab=bool(kw["new_tab"]) if "new_tab" in kw else True,
        ),
        "file_by_id": lambda **kw: file_by_id_mod.file_by_id(
            db,
            entry_id=str(kw.get("entry_id") or kw.get("id") or ""),
            entry_ids=kw.get("entry_ids"),
            inbox_id=str(kw.get("inbox_id") or ""),
            dest=str(kw.get("dest") or ""),
            summary=kw.get("summary"),
        ),
        "file_note": lambda **kw: file_note_mod.file_note(
            db,
            entry_id=str(kw.get("entry_id") or kw.get("id") or ""),
            dest=str(kw.get("dest") or ""),
            text=str(kw.get("text") or ""),
            summary=kw.get("summary"),
        ),
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
