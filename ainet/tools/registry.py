"""Ollama-compatible tool definitions and dispatcher."""

from __future__ import annotations

from typing import Any, Callable

from ainet.tools.ops import DatabaseTools
from ainet.tools.paths import PathError
from ainet.tools.permissions import PermissionError_
from ainet.tools.web import web_fetch, web_search

# OAC-safe tools (no general DB mutations). Kept in sync with ollama.modes.base.
READ_TOOL_NAMES = frozenset(
    {"list_dir", "tree", "read_text", "read_json", "web_search", "web_fetch"}
)
QUIZ_TOOL_NAMES = frozenset(
    {
        "should_suggest_quiz",
        "list_quiz_candidates",
        "start_quiz",
        "record_quiz_answer",
        "get_quiz_status",
    }
)
OAC_TOOL_NAMES = READ_TOOL_NAMES | QUIZ_TOOL_NAMES



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
                "results. Use for external facts; do not invent. Cite titles/urls briefly."
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
            "name": "get_tools",
            "description": (
                "List the AINet tool catalog (DB ops + web_search/web_fetch + quiz helpers). "
                "Call this when you need a tool that is not in your current lean set; "
                "after calling, the full catalog is unlocked for later tool calls "
                "(OAC stays read-only + web + allowlisted quiz tools)."
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
            "name": "should_suggest_quiz",
            "description": (
                "Heuristic: whether OAC may casually suggest a short research quiz now. "
                "Anti-spam — occasional only (turn/time gaps). Call before suggesting; "
                "never suggest every message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "turn_count": {
                        "type": "integer",
                        "description": "Optional approximate turns in this chat.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_quiz_candidates",
            "description": (
                "Rank research sessions for quizzing: prefer recent sessions and low "
                "memory scores / previously wrong items. Returns sample details_covered."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 12},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_quiz",
            "description": (
                "Start a quiz loop after Hayden confirms. Pass drafted questions "
                "(prompt + expected_answer + session_id) or omit to auto-seed from "
                "ranked research session details. Active state lives under runtime/oac/."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "description": (
                            "Optional drafted questions: "
                            "{prompt, expected_answer, session_id?, topic_slug?}"
                        ),
                        "items": {"type": "object"},
                    },
                    "count": {"type": "integer", "default": 5},
                    "session_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional restrict auto-seed to these session ids.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_quiz_answer",
            "description": (
                "After grading Hayden's answer conversationally: record correct/incorrect, "
                "persist memory scores under Hayden/Research/Scores.json, advance to next "
                "question (or complete the quiz)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_answer": {"type": "string"},
                    "correct": {"type": "boolean"},
                    "brief_correction": {
                        "type": "string",
                        "description": "Short correction/teach note if wrong.",
                    },
                    "question_id": {
                        "type": "string",
                        "description": "Optional; defaults to current question.",
                    },
                },
                "required": ["user_answer", "correct"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_quiz_status",
            "description": "Get active/idle/completed quiz state and current question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reveal_answer": {
                        "type": "boolean",
                        "default": False,
                        "description": "If true, include expected_answer for grading.",
                    }
                },
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
                "dest: 'research' | 'identity' | 'voice' | 'psychology' | 'habits' | "
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
                        "description": "Several related Changelog ids (one research rabbit hole)",
                    },
                    "inbox_id": {
                        "type": "string",
                        "description": "Hayden/Inbox/Captures.json capture id",
                    },
                    "dest": {
                        "type": "string",
                        "description": (
                            "'research' | 'identity' | 'voice' | 'psychology' | 'habits' | "
                            "'discard' (greetings only) | or a Folderrules JSON leaf. "
                            "Not Inbox. Not a dump for schedules/courses."
                        ),
                    },
                    "subject": {
                        "type": "string",
                        "description": "Research subject when dest=research (optional)",
                    },
                    "topic_slug": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["dest"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upsert_research_session",
            "description": (
                "SOI: create/update a Hayden/Research/Sessions/<Id>.json entity. "
                "Prefer file_by_id(dest='research', entry_ids=[...]). If you use this, "
                "pass changelog_entry_ids only — host fills details_covered from Changelog. "
                "Do not paste full turn text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Omit to allocate a new id.",
                    },
                    "subject": {"type": "string"},
                    "title": {"type": "string"},
                    "topic_slug": {"type": "string"},
                    "topic_path": {"type": "string"},
                    "details_covered": {
                        "type": "array",
                        "description": (
                            "Structured points/mechanisms/QAs: "
                            "{kind, text, question?, answer?, tags?}"
                        ),
                        "items": {},
                    },
                    "append_details": {"type": "boolean", "default": True},
                    "length_turns": {"type": "integer"},
                    "started_at": {"type": "string"},
                    "related_topic": {"type": "string"},
                    "source_session_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "changelog_entry_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "notes": {"type": "string"},
                    "status": {"type": "string", "enum": ["open", "complete"]},
                    "summary": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_research_session",
            "description": (
                "Mark a research session complete: set ended_at, duration_seconds, status=complete. "
                "Call when research mode ends or the rabbit hole clearly wraps up."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "ended_at": {"type": "string"},
                    "details_covered": {
                        "type": "array",
                        "items": {},
                    },
                    "length_turns": {"type": "integer"},
                    "summary": {"type": "string"},
                },
                "required": ["session_id"],
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

    read_only=True → OAC view (list/tree/read + web + quiz helpers).
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
            "OAC catalog: read + web + allowlisted quiz tools (no general DB writes)."
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
    from ollama import quiz as quiz_mod
    from ollama import research_sessions as sessions_mod

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
        "should_suggest_quiz": lambda **kw: quiz_mod.should_suggest_quiz(
            db,
            turn_count=kw.get("turn_count"),
        ),
        "list_quiz_candidates": lambda **kw: quiz_mod.list_quiz_candidates(
            db,
            limit=int(kw.get("limit", 12)),
        ),
        "start_quiz": lambda **kw: quiz_mod.start_quiz(
            db,
            questions=kw.get("questions"),
            count=int(kw.get("count", 5)),
            session_ids=kw.get("session_ids"),
        ),
        "record_quiz_answer": lambda **kw: quiz_mod.record_quiz_answer(
            db,
            user_answer=str(kw.get("user_answer") or ""),
            correct=bool(kw["correct"]),
            brief_correction=str(kw.get("brief_correction") or ""),
            question_id=kw.get("question_id"),
        ),
        "get_quiz_status": lambda **kw: quiz_mod.get_quiz_status(
            db,
            reveal_answer=bool(kw.get("reveal_answer", False)),
        ),
        "file_by_id": lambda **kw: file_by_id_mod.file_by_id(
            db,
            entry_id=str(kw.get("entry_id") or kw.get("id") or ""),
            entry_ids=kw.get("entry_ids"),
            inbox_id=str(kw.get("inbox_id") or ""),
            dest=str(kw.get("dest") or ""),
            subject=str(kw.get("subject") or ""),
            topic_slug=str(kw.get("topic_slug") or ""),
            summary=kw.get("summary"),
        ),
        "upsert_research_session": lambda **kw: sessions_mod.upsert_research_session(
            db,
            session_id=kw.get("session_id"),
            subject=str(kw.get("subject") or ""),
            title=str(kw.get("title") or ""),
            topic_slug=str(kw.get("topic_slug") or ""),
            topic_path=str(kw.get("topic_path") or ""),
            details_covered=kw.get("details_covered"),
            append_details=bool(kw.get("append_details", True)),
            length_turns=kw.get("length_turns"),
            started_at=kw.get("started_at"),
            related_topic=str(kw.get("related_topic") or ""),
            source_session_ids=kw.get("source_session_ids"),
            changelog_entry_ids=kw.get("changelog_entry_ids"),
            notes=str(kw.get("notes") or ""),
            status=kw.get("status"),
            summary=kw.get("summary"),
        ),
        "complete_research_session": lambda **kw: sessions_mod.complete_research_session(
            db,
            str(kw["session_id"]),
            ended_at=kw.get("ended_at"),
            details_covered=kw.get("details_covered"),
            length_turns=kw.get("length_turns"),
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
