"""Mode definition — prompt lives in ollama.prompts.*, not here."""

from __future__ import annotations

from dataclasses import dataclass


READ_TOOLS = (
    "list_dir",
    "tree",
    "read_text",
    "read_json",
    "web_search",
    "web_fetch",
    "image_search",
    "open_chrome",
    "list_projects",
)
WRITE_LIGHT = ("patch_json", "set_json_path", "create_json", "write_json", "capture_inbox")
STRUCT_TOOLS = ("create_folder", "create_cop", "move_path", "archive_to_history")
META_TOOLS = ("get_tools",)
PROJECT_TOOLS = ("create_project", "list_projects", "open_project", "close_project")


@dataclass(frozen=True)
class Mode:
    id: str
    name: str
    description: str
    prompt: str
    tools_enabled: bool = True
    tool_names: tuple[str, ...] | None = None
    role: str = "oac"
    allow_mutations: bool = False
