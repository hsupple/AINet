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
        "query_db",
        "web_search",
        "web_fetch",
        "image_search",
        "create_plot",
        "open_chrome",
        "spotify",
        "list_projects",
        "query_calendar",
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
# Calendar mutations are allowed for OAC the same way project session tools are.
CALENDAR_SESSION_TOOLS = frozenset(
    {
        "query_calendar",
        "add_calendar_event",
        "update_calendar_event",
        "cancel_calendar_event",
    }
)

# Display-only phrase for the chat tool card. Handlers ignore this key.
_CARD_ABOUT: dict[str, Any] = {
    "type": "string",
    "description": (
        "A few words for the chat card saying what you are looking up "
        "(e.g. 'human biology overview'). Not used to search or filter."
    ),
}


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "path='.' lists the knowledge files at db/ root. Prefer query_db to look "
                "up stored facts by name, words, and dates instead of dumping whole files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative folder path. Use '.' for the knowledge-file index.",
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
            "description": (
                "Read and parse a whole JSON file. Prefer query_db for personal facts "
                "(people, traits, dates, words). Use this when you need the raw file."
            ),
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
            "name": "query_db",
            "description": (
                "Look up Hayden's stored personal database. The host may already inject "
                "relevant facts as KNOWN CONTEXT — use those silently (do not recap his bio). "
                "Do not use this for Hayden's calendar or schedule — use query_calendar. "
                "Call this when you still need a stored fact that is not in that block: "
                "identity, friends, habits, preferences, home, feelings, memories, or past questions. "
                "First-person statements count, not only questions. "
                "Do not invent those from general knowledge or the web. "
                "q accepts multiple words at once; matches if any of those words hit. "
                "Omit dest to search all files except secrets. Prefer broad search, then narrow. "
                "Answer from digest / entries[].text in second person (you/your). "
                "Only use the facts that matter to this message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dest": {
                        "type": "string",
                        "description": (
                            "hayden (Hayden himself — characteristics, preferences, …), "
                            "people (others in his life only), questions, household, memories, "
                            "secrets, a hayden section (preferences, habits, …), a project name, or a filename"
                        ),
                    },
                    "file": {
                        "type": "string",
                        "description": "Alias of dest (people.json, hayden.json, ...)",
                    },
                    "name": {
                        "type": "string",
                        "description": "Person, trait, or topic key (substring ok)",
                    },
                    "q": {
                        "type": "string",
                        "description": (
                            "One or more words to match in the key or observation text "
                            "(space-separated; any word can hit — useful for short phrases)"
                        ),
                    },
                    "after": {
                        "type": "string",
                        "description": "Keep observations on/after this date (YYYY-MM-DD or ISO)",
                    },
                    "before": {
                        "type": "string",
                        "description": "Keep observations on/before this date (YYYY-MM-DD or ISO)",
                    },
                    "since_days": {
                        "type": "integer",
                        "description": "Only observations from the last N days",
                    },
                    "keys_only": {
                        "type": "boolean",
                        "description": "Return names/counts without observation text",
                        "default": False,
                    },
                    "include_secrets": {
                        "type": "boolean",
                        "description": "Include secrets.json when dest is omitted",
                        "default": False,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max matching keys to return (default 16, max 40)",
                        "default": 16,
                    },
                    "about": _CARD_ABOUT,
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_calendar",
            "description": (
                "Look up Hayden's calendar (db/Calendar.json). "
                "ALWAYS pass start and end as YYYY-MM-DD for the exact day Hayden named. "
                "today/tonight → both = today's date; tomorrow → tomorrow; "
                "Monday/next Monday → that Monday; this week/next week → that week's Monday–Sunday. "
                "Phrases like 'today's stuff' still mean today — do not put that text in q. "
                "Leave q empty to list everything on those dates. "
                "Only set q for a real event: a course code (MA 265), test/exam/quiz/lab, or a title word. "
                "Never put greetings or filler in q (pal, hey, stuff, classes, schedule). "
                "Do not invent events. Do not use query_db for the calendar."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": (
                            "Range start YYYY-MM-DD. Required. "
                            "Must be the date Hayden asked about (today, tomorrow, that Monday, etc.)."
                        ),
                    },
                    "end": {
                        "type": "string",
                        "description": (
                            "Range end YYYY-MM-DD inclusive. Same as start for a single day."
                        ),
                    },
                    "q": {
                        "type": "string",
                        "description": (
                            "Optional. Course code or event kind only (e.g. 'MA 265 test'). "
                            "Empty for a full day. Never 'today', 'stuff', or a greeting."
                        ),
                    },
                    "upcoming": {
                        "type": "integer",
                        "description": "Return the next N upcoming occurrences from now",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max occurrences (default 40, max 80)",
                        "default": 40,
                    },
                    "about": _CARD_ABOUT,
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_calendar_event",
            "description": (
                "Add an event to Hayden's calendar. OAC may call this even without a focused project. "
                "Set when it starts, how long (end or duration_minutes), whether it repeats, "
                "and until when. Use local times unless timezone is given."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title"},
                    "start": {
                        "type": "string",
                        "description": "Start as YYYY-MM-DD or ISO local datetime",
                    },
                    "end": {
                        "type": "string",
                        "description": "End as YYYY-MM-DD or ISO local datetime",
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Length in minutes if end is omitted (default 60, or 1440 if all-day)",
                    },
                    "all_day": {
                        "type": "boolean",
                        "description": "True for an all-day event",
                        "default": False,
                    },
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone; defaults to local",
                    },
                    "repeat": {
                        "type": "string",
                        "description": "none | daily | weekly | monthly | yearly",
                        "default": "none",
                    },
                    "interval": {
                        "type": "integer",
                        "description": "Repeat every N periods (default 1)",
                        "default": 1,
                    },
                    "until": {
                        "type": "string",
                        "description": "Repeat through this date (YYYY-MM-DD)",
                    },
                    "byweekday": {
                        "description": "For weekly: MO,TU,WE,TH,FR,SA,SU or a list of those",
                    },
                    "location": {"type": "string"},
                    "notes": {"type": "string"},
                    "category": {
                        "type": "string",
                        "description": "school | work | personal | health | social | other",
                    },
                    "color": {"type": "string", "description": "Optional hex color"},
                    "reminder_minutes": {
                        "description": "Minutes before start, e.g. 15 or [15, 60]",
                    },
                    "about": _CARD_ABOUT,
                },
                "required": ["title", "start"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_calendar_event",
            "description": (
                "Update an existing calendar event by id. Pass only the fields that change. "
                "OAC may call this even without a focused project."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Event id from query_calendar"},
                    "title": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "duration_minutes": {"type": "integer"},
                    "all_day": {"type": "boolean"},
                    "timezone": {"type": "string"},
                    "repeat": {"type": "string"},
                    "interval": {"type": "integer"},
                    "until": {"type": "string"},
                    "byweekday": {"description": "Weekly weekdays (MO,TU,…)"},
                    "location": {"type": "string"},
                    "notes": {"type": "string"},
                    "category": {"type": "string"},
                    "color": {"type": "string"},
                    "reminder_minutes": {"description": "Minutes before start"},
                    "cancelled": {"type": "boolean"},
                    "about": _CARD_ABOUT,
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_calendar_event",
            "description": (
                "Cancel (or permanently delete) a calendar event by id. "
                "OAC may call this even without a focused project."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Event id from query_calendar"},
                    "delete": {
                        "type": "boolean",
                        "description": "If true, remove the event instead of marking cancelled",
                        "default": False,
                    },
                    "about": _CARD_ABOUT,
                },
                "required": ["id"],
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
                "ainet/defaults/ (Hayden.json, LogFile.json, project.json, else generic.json)."
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
                        "description": "Template filename, e.g. Hayden.json or project.json",
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
                "PREFERRED way to start any new project. Creates Projects/<Name>/ "
                "with project.json and Files/. Never use create_folder for this. "
                "Then call open_project to focus this chat on it."
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
                        "description": "Optional short description",
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
            "description": "List user projects under Projects/ (name, path, top labels).",
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
            "name": "log_item",
            "description": (
                "SOI filing: append one lasting observation under a stable key. "
                "Reuse an existing key from the labels list whenever the subject matches "
                "(case-insensitive). Create a new key only when no existing key fits. "
                "Prefer coarse durable keys over narrow one-off phrases. "
                "dest chooses the file/section; label is the key; reason is the fact text. "
                "For people, label is the person's name. For hayden, use a broad trait bucket "
                "(e.g. personality, interests, education). "
                "reason states what is true — not that someone asked or confirmed something. "
                "dest=discard when the turn has no lasting fact. "
                "Split kinds of fact with multiple calls: same entry_id, different dest/label/reason. "
                "Only file content that belongs to the cited entry_id(s)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entry_id": {"type": "string", "description": "One Changelog entry id"},
                    "entry_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Several Changelog ids for one synthesized same-session item",
                    },
                    "dest": {
                        "type": "string",
                        "description": (
                            "hayden, preferences, habits, values, desires, body, psychology, "
                            "people, questions, household, memories, secrets, "
                            "discard, or a project name"
                        ),
                    },
                    "label": {
                        "type": "string",
                        "description": (
                            "Stable subject key to create or append. Reuse existing keys when possible. "
                            "Person name for people; broad trait/topic for other dests — not a full sentence."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "One third-person sentence stating the lasting fact to store under that key"
                        ),
                    },
                    "summary": {"type": "string"},
                },
                "required": ["dest", "label", "reason"],
            },
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
                    "about": _CARD_ABOUT,
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
                    "about": _CARD_ABOUT,
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
                    "about": _CARD_ABOUT,
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
                    "about": _CARD_ABOUT,
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
                    "about": _CARD_ABOUT,
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
        if read_only and name not in OAC_TOOL_NAMES and name not in CALENDAR_SESSION_TOOLS:
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
            "OAC catalog: read + web + calendar (no general DB writes)."
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


def _tool_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def _calendar_root(db: DatabaseTools) -> Any:
    return db.paths.root


def _query_calendar_tool(db: DatabaseTools, **kw: Any) -> dict[str, Any]:
    from ainet.calendar_store import infer_schedule_query, query_events

    blob = " ".join(
        str(kw.get(k) or "")
        for k in ("q", "query", "about")
        if str(kw.get(k) or "").strip()
    )
    inferred = infer_schedule_query(blob) if blob.strip() else {}
    start = str(kw.get("start") or inferred.get("start") or "")
    end = str(kw.get("end") or inferred.get("end") or "")
    q = str(inferred.get("q") or "").strip()
    upcoming = kw.get("upcoming")
    if inferred.get("date_explicit"):
        # Natural-language q/about named a day — that day wins over a stale range.
        start = str(inferred.get("start") or start)
        end = str(inferred.get("end") or end)
        upcoming = None
    elif inferred.get("upcoming") and upcoming in (None, ""):
        upcoming = inferred.get("upcoming")
    result = query_events(
        _calendar_root(db),
        start=start,
        end=end,
        q=q,
        upcoming=int(upcoming) if upcoming not in (None, "") else None,
        limit=int(kw.get("limit", 40)),
        include_cancelled=_tool_bool(kw.get("include_cancelled"), False),
    )
    rows = [e for e in (result.get("events") or []) if isinstance(e, dict)]
    slim_rows = [
        {
            "title": r.get("title"),
            "start": r.get("start"),
            "end": r.get("end"),
            "location": r.get("location") or "",
            "occurrence_date": r.get("occurrence_date"),
        }
        for r in rows
    ]
    return {
        "ok": True,
        "start": result.get("start"),
        "end": result.get("end"),
        "count": len(slim_rows),
        "digest": result.get("digest") or "",
        "events": slim_rows,
        "hint": (
            "Answer ONLY from digest. One line per event. "
            "Only the dates in this result — no weekly recap, no markdown tables, "
            "no extra days, no Repeat fields."
        ),
    }


def _add_calendar_tool(db: DatabaseTools, **kw: Any) -> dict[str, Any]:
    from ainet.calendar_store import add_event

    return add_event(_calendar_root(db), kw)


def _update_calendar_tool(db: DatabaseTools, **kw: Any) -> dict[str, Any]:
    from ainet.calendar_store import update_event

    return update_event(_calendar_root(db), kw)


def _cancel_calendar_tool(db: DatabaseTools, **kw: Any) -> dict[str, Any]:
    from ainet.calendar_store import cancel_event

    return cancel_event(
        _calendar_root(db),
        str(kw.get("id") or ""),
        delete=_tool_bool(kw.get("delete"), False),
    )


def _handlers(db: DatabaseTools) -> dict[str, Callable[..., dict[str, Any]]]:
    from ainet.logstore import log_item as log_item_fn
    from ainet.logstore import query_db as query_db_fn
    from ainet.tools import project as project_mod

    return {
        "list_dir": lambda **kw: db.list_dir(kw.get("path", ".")),
        "tree": lambda **kw: db.tree(kw.get("path", "."), int(kw.get("max_depth", 3))),
        "read_text": lambda **kw: db.read_text(kw["path"]),
        "read_json": lambda **kw: db.read_json(kw["path"]),
        "query_calendar": lambda **kw: _query_calendar_tool(db, **kw),
        "add_calendar_event": lambda **kw: _add_calendar_tool(db, **kw),
        "update_calendar_event": lambda **kw: _update_calendar_tool(db, **kw),
        "cancel_calendar_event": lambda **kw: _cancel_calendar_tool(db, **kw),
        "query_db": lambda **kw: query_db_fn(
            db,
            dest=str(kw.get("dest") or ""),
            file=str(kw.get("file") or ""),
            name=str(kw.get("name") or kw.get("label") or ""),
            q=str(kw.get("q") or kw.get("query") or kw.get("words") or ""),
            after=str(kw.get("after") or kw.get("since") or ""),
            before=str(kw.get("before") or kw.get("until") or ""),
            since_days=int(kw["since_days"]) if kw.get("since_days") not in (None, "") else None,
            keys_only=_tool_bool(kw.get("keys_only"), False),
            include_secrets=_tool_bool(kw.get("include_secrets"), False),
            limit=int(kw.get("limit", 16)),
        ),
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
        "move_path": lambda **kw: db.move_path(
            kw["src"], kw["dest"], summary=kw.get("summary")
        ),
        "append_changelog": lambda **kw: db.append_changelog(
            kw["action"], kw["path"], kw["summary"], kw.get("details")
        ),
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
        "log_item": lambda **kw: log_item_fn(
            db,
            dest=str(kw.get("dest") or ""),
            label=str(kw.get("label") or ""),
            reason=str(kw.get("reason") or kw.get("text") or ""),
            entry_id=str(kw.get("entry_id") or kw.get("id") or ""),
            entry_ids=kw.get("entry_ids"),
            summary=kw.get("summary"),
        ),
        "file_note": lambda **kw: log_item_fn(
            db,
            dest=str(kw.get("dest") or ""),
            label=str(kw.get("label") or kw.get("dest") or ""),
            reason=str(kw.get("reason") or kw.get("text") or ""),
            entry_id=str(kw.get("entry_id") or kw.get("id") or ""),
            entry_ids=kw.get("entry_ids"),
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
