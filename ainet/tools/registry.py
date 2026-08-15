"""Ollama-compatible tool definitions and dispatcher."""

from __future__ import annotations

from typing import Any, Callable

from ainet.tools.browser import open_chrome
from ainet.tools.ops import DatabaseTools
from ainet.tools.paths import PathError
from ainet.tools.permissions import PermissionError_
from ainet.tools.research import inspect_research, save_research
from ainet.tools.web import image_search, web_fetch, web_search
from ainet.tools.plot import create_plot
from ainet.tools.spotify import spotify

# OAC-safe tools (no general DB mutations). Kept in sync with ollama.modes.base.
READ_TOOL_NAMES = frozenset(
    {
        "list_dir",
        "tree",
        "read_text",
        "read_json",
        "web_search",
        "web_fetch",
        "image_search",
        "create_plot",
        "open_chrome",
        "spotify",
        "list_projects",
    }
)
OAC_TOOL_NAMES = READ_TOOL_NAMES
# Session-scoped; OAC may call even when allow_mutations is False.
PROJECT_SESSION_TOOLS = frozenset(
    {
        "create_project",
        "list_projects",
        "open_project",
        "close_project",
    }
)



TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "path='.' returns every Read.json in the database — use this first to "
                "find where something about Hayden lives. Any other path lists that "
                "folder's immediate children."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative folder path. Use '.' for the whole-database Read.json index.",
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
            "description": (
                "Create a subfolder only (inside an allowed location or a focused project). "
                "Do NOT use this to start a new project — call create_project instead."
            ),
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
            "name": "write_text",
            "description": (
                "Create or overwrite a UTF-8 text document (.txt, .md, etc.). "
                "Prefer write_json/create_json for .json files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string", "description": "Full file text"},
                    "create": {"type": "boolean", "default": True},
                    "summary": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_project",
            "description": (
                "PREFERRED way to start any new project. Creates a project directory "
                "at Projects/<Name>/ with Read.json, History.json, Notes.json, Plan.json, "
                "Profile.json, Files/, and History/. Never use create_folder or create_cop "
                "for this. Then call open_project to focus this chat on it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Project folder name (e.g. AINet or My App)",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Short description for Read.json",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_projects",
            "description": "List user projects under Projects/ (name, path, Read summary).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_project",
            "description": (
                "Focus this chat on Projects/<Name>. While focused, that folder is the only "
                "accessible DB path — create folders/text, rename, read by filename. "
                "Call close_project to leave."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Project name or Projects/<Name> path",
                    },
                    "path": {
                        "type": "string",
                        "description": "Alias for name",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_project",
            "description": "Leave project focus and restore normal OAC access to the full db/.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_cop",
            "description": (
                "Create a school course or Work COP from Folderrules templates "
                "(Profile/Read/Plan/History). "
                "path = COP root (School/Courses/<Code> or Work/Projects/<Name>). "
                "kind = course | project. "
                "Not for Hayden's user projects — those use create_project."
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
            "name": "refresh_read",
            "description": (
                "SOI Phase 2 only: atomically update a folder's compact Read.json digest "
                "and mark its pending freshness log consumed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "read_path": {
                        "type": "string",
                        "description": "Exact Read.json path supplied in the phase-2 payload",
                    },
                    "digest": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "state": {"type": "string"},
                            "important_context": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "recent_changes": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "active_items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "known_facts": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "uncertainties": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "summary",
                            "state",
                            "important_context",
                            "recent_changes",
                            "active_items",
                            "known_facts",
                            "uncertainties",
                        ],
                    },
                },
                "required": ["read_path", "digest"],
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
                "Use for external facts; do not invent. Cite titles/urls briefly. "
                "Include the current month/year from the system prompt in the query "
                "(e.g. August 2026) — do not search as if it were 2024."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query; include current year (e.g. 2026) when time-sensitive",
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
            "name": "image_search",
            "description": (
                "Search for pictures (photos, diagrams, screenshots). Returns image URLs "
                "and thumbnails that the chat UI will show. Also opens Google Images in "
                "Chrome unless the user opted out. Use when Hayden asks to see photos, "
                "what something looks like, or images from the web / Google."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to find pictures of",
                    },
                    "count": {
                        "type": "integer",
                        "default": 6,
                        "description": "Number of images (1-8).",
                    },
                    "open_google": {
                        "type": "boolean",
                        "default": True,
                        "description": "Open a Google Images tab in Chrome (default true).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_plot",
            "description": (
                "Render a clean interactive chart in the chat (2D or 3D). "
                "Use for graphs, curves, stress-strain, surfaces, comparisons. "
                "Pass numeric series and/or an equation like 'sin(x)' or 'x**2 + y**2'. "
                "IF Hayden needs real material/data first -> web_search/web_fetch, then create_plot. "
                "Never invent measured data; use search or equations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Chart title"},
                    "chart": {
                        "type": "string",
                        "description": (
                            "line | scatter | bar | area | histogram | box | pie | "
                            "heatmap | contour | surface | isosurface | scatter3d | line3d. "
                            "Use isosurface (or surface) for F(x,y,z)=0 implicit surfaces."
                        ),
                        "default": "line",
                    },
                    "xlab": {"type": "string", "description": "X-axis label"},
                    "ylab": {"type": "string", "description": "Y-axis label"},
                    "zlab": {"type": "string", "description": "Z-axis label (3D)"},
                    "equation": {
                        "type": "string",
                        "description": (
                            "Math expression. 2D: in x. Explicit surface: z=f(x,y) as f only. "
                            "Implicit: F(x,y,z)=0 (LaTeX ok). Use ** or ^ for powers."
                        ),
                    },
                    "x": {
                        "description": "X values (numbers) for a single series shortcut",
                    },
                    "y": {
                        "description": "Y values (numbers) for a single series shortcut",
                    },
                    "z": {
                        "description": "Z values or 2D grid for 3D/heatmap",
                    },
                    "x_min": {"type": "number", "description": "Equation domain min x"},
                    "x_max": {"type": "number", "description": "Equation domain max x"},
                    "y_min": {"type": "number", "description": "Equation domain min y"},
                    "y_max": {"type": "number", "description": "Equation domain max y"},
                    "z_min": {"type": "number", "description": "Equation domain min z (isosurface)"},
                    "z_max": {"type": "number", "description": "Equation domain max z (isosurface)"},
                    "n": {"type": "integer", "description": "Sample count for equations"},
                    "series": {
                        "type": "array",
                        "description": (
                            "Multiple series: [{name, x, y, z?, equation?, x_min?, x_max?, "
                            "y_min?, y_max?, z_min?, z_max?, n?, color?, labels? (pie)}]"
                        ),
                    },
                    "source": {
                        "type": "string",
                        "description": "Short citation if data came from the web",
                    },
                    "animate": {
                        "type": "boolean",
                        "default": True,
                        "description": "Animate draw-in along x (default true)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spotify",
            "description": (
                "Control Hayden's Spotify: what's playing, search, play/pause/skip, "
                "volume, queue, Liked Songs, devices. "
                "IF not connected -> action=connect (opens login). "
                "Playback needs an active Spotify app on PC/phone. "
                "IF Hayden names a song/artist/vibe to hear -> action=play with query; "
                "IF Hayden says queue -> action=queue with query. Both auto-pick the best "
                "match and queue the remaining results, so never just search and ask. "
                "action=search is only for questions with no playback intent. "
                "IF play Liked Songs / saved songs -> action=play query='liked songs' "
                "(never search the catalog for that phrase)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": (
                            "status | connect | now_playing | search | liked | play | pause | "
                            "next | previous | volume | queue | devices | transfer"
                        ),
                        "default": "status",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search text or song/artist to play/queue; use 'liked songs' to play that library",
                    },
                    "uri": {
                        "type": "string",
                        "description": "spotify:track:... (or album/playlist) URI",
                    },
                    "device_id": {
                        "type": "string",
                        "description": "Target device id (from devices / transfer)",
                    },
                    "volume": {
                        "type": "integer",
                        "description": "Volume 0-100 (action=volume)",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "description": "Search / liked list count",
                    },
                },
                "required": [],
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
                "Open one or more http(s) URLs as new Chrome tabs on this PC. "
                "Pass url and/or urls=[...] to open several at once (max 8). "
                "web_search auto-opens top results — use this for extra pages. "
                "Never claim a tab opened unless this tool ran (or search auto_opened it)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Primary absolute http(s) URL to open",
                    },
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional http(s) URLs to open together (max 8 total).",
                    },
                    "new_tab": {
                        "type": "boolean",
                        "default": True,
                        "description": "Open in a new tab (default true).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_research",
            "description": (
                "Save a Deep Research brief into the private host vault "
                "(runtime/research — invisible to SOI and normal DB reads). "
                "The Doc panel shows the preview. Call once after fetching sources. "
                "Body is markdown (1–2 pages) with numbered citations; "
                "sources must include at least two http(s) URLs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Brief title",
                    },
                    "question": {
                        "type": "string",
                        "description": "Hayden's original research question",
                    },
                    "summary": {
                        "type": "string",
                        "description": "≤400 char abstract",
                    },
                    "body": {
                        "type": "string",
                        "description": "Markdown 1–2 pager with [1] [2] citations",
                    },
                    "key_findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "3–8 bullet findings",
                    },
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "publisher": {"type": "string"},
                                "year": {"type": "string"},
                                "note": {"type": "string"},
                            },
                            "required": ["url"],
                        },
                        "description": "Cited sources (IEEE, ACM, NIH, journals, societies — any credible publisher)",
                    },
                },
                "required": ["title", "body", "sources"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_research",
            "description": (
                "Only way to list/read Deep Research briefs in the private vault. "
                "Use list_only=true to list, or pass brief_id/path to load one brief."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "list_only": {
                        "type": "boolean",
                        "default": False,
                        "description": "If true, list recent briefs (id/title/path).",
                    },
                    "brief_id": {
                        "type": "string",
                        "description": "Brief id from a prior save/list",
                    },
                    "path": {
                        "type": "string",
                        "description": "Vault-relative path e.g. runtime/research/briefs/Foo.json",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Max briefs when listing",
                    },
                },
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
                "Phase 1 filing — preferred for SOI. Pass dest, text (a short note YOU write), "
                "and entry_id or entry_ids. Same-session threads MAY be one synthesized note "
                "with entry_ids covering the whole inquiry. Host stores the note in "
                "<folder>/Notes.json and each raw message in History.json. "
                "Call again with the same entry_id and a different dest when one turn "
                "contains several kinds of fact (friends -> Relationships, feelings -> "
                "Psychology, wants -> Desires). Each call's text covers only that dest. "
                "dest=discard for greetings and acknowledgment-only turns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "string",
                        "description": "One Changelog entry id",
                    },
                    "entry_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Several Changelog ids for one synthesized same-session note",
                    },
                    "dest": {
                        "type": "string",
                        "description": (
                            "Folder from the folders list (Values, Pantry, Hayden/Values) "
                            "or Questions (the only allowed root). discard for greetings/"
                            "acknowledgments with no new information."
                        ),
                    },
                    "text": {
                        "type": "string",
                        "description": "Short note that will make sense months later (not a raw paste)",
                    },
                    "summary": {"type": "string"},
                },
                "required": ["dest"],
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
    from ainet.tools import project as project_mod
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
        "write_text": lambda **kw: db.write_text(
            kw["path"],
            str(kw.get("content") or ""),
            create=bool(kw.get("create", True)),
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
        "create_project": lambda **kw: project_mod.create_project(
            db,
            name=str(kw.get("name") or kw.get("path") or ""),
            summary=str(kw.get("summary") or ""),
        ),
        "list_projects": lambda **kw: project_mod.list_projects(db),
        "open_project": lambda **kw: {
            "ok": False,
            "error": "open_project is handled by the chat session host",
        },
        "close_project": lambda **kw: {
            "ok": False,
            "error": "close_project is handled by the chat session host",
        },
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
        "refresh_read": lambda **kw: db.refresh_read(kw["read_path"], kw["digest"]),
        "mark_read_refreshed": lambda **kw: db.mark_read_refreshed(kw["read_path"]),
        "list_stale_reads": lambda **kw: db.list_stale_reads(),
        "web_search": lambda **kw: web_search(
            kw["query"],
            count=int(kw.get("count", 5)),
        ),
        "image_search": lambda **kw: image_search(
            kw["query"],
            count=int(kw.get("count", 6)),
            open_google=bool(kw["open_google"]) if "open_google" in kw else True,
        ),
        "create_plot": lambda **kw: create_plot(
            str(kw.get("title") or ""),
            chart=str(kw.get("chart") or "line"),
            xlab=str(kw.get("xlab") or ""),
            ylab=str(kw.get("ylab") or ""),
            zlab=str(kw.get("zlab") or ""),
            series=kw.get("series"),
            x=kw.get("x"),
            y=kw.get("y"),
            z=kw.get("z"),
            equation=str(kw.get("equation") or ""),
            x_min=float(kw["x_min"]) if kw.get("x_min") is not None else None,
            x_max=float(kw["x_max"]) if kw.get("x_max") is not None else None,
            y_min=float(kw["y_min"]) if kw.get("y_min") is not None else None,
            y_max=float(kw["y_max"]) if kw.get("y_max") is not None else None,
            z_min=float(kw["z_min"]) if kw.get("z_min") is not None else None,
            z_max=float(kw["z_max"]) if kw.get("z_max") is not None else None,
            n=int(kw["n"]) if kw.get("n") is not None else None,
            animate=bool(kw["animate"]) if "animate" in kw else True,
            source=str(kw.get("source") or ""),
        ),
        "spotify": lambda **kw: spotify(
            str(kw.get("action") or "status"),
            query=str(kw.get("query") or ""),
            uri=str(kw.get("uri") or ""),
            device_id=str(kw.get("device_id") or ""),
            volume=int(kw["volume"]) if kw.get("volume") is not None else None,
            limit=int(kw.get("limit", 5)),
        ),
        "web_fetch": lambda **kw: web_fetch(
            kw["url"],
            max_chars=int(kw.get("max_chars", 4000)),
        ),
        "open_chrome": lambda **kw: open_chrome(
            str(kw.get("url") or ""),
            urls=kw.get("urls") if isinstance(kw.get("urls"), list) else None,
            new_tab=bool(kw["new_tab"]) if "new_tab" in kw else True,
        ),
        "save_research": lambda **kw: save_research(
            db,
            title=str(kw.get("title") or ""),
            body=str(kw.get("body") or ""),
            question=str(kw.get("question") or ""),
            summary=str(kw.get("summary") or ""),
            key_findings=kw.get("key_findings"),
            sources=kw.get("sources"),
            url=str(kw.get("url") or ""),
            link=str(kw.get("link") or ""),
            href=str(kw.get("href") or ""),
        ),
        "inspect_research": lambda **kw: inspect_research(
            db,
            brief_id=str(kw.get("brief_id") or kw.get("id") or ""),
            path=str(kw.get("path") or ""),
            list_only=bool(kw.get("list_only", False)),
            limit=int(kw.get("limit", 20)),
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
            entry_ids=kw.get("entry_ids"),
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
