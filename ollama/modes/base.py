"""Mode definition — prompt lives in ollama.prompts.*, not here."""

from __future__ import annotations

from dataclasses import dataclass


# Lean tool sets
READ_TOOLS = ("list_dir", "tree", "read_text", "read_json")
WRITE_LIGHT = ("patch_json", "set_json_path", "create_json", "write_json", "capture_inbox")
STRUCT_TOOLS = ("create_folder", "create_cop", "move_path", "archive_to_history")
META_TOOLS = ("get_tools",)


@dataclass(frozen=True)
class Mode:
    id: str
    name: str
    description: str
    prompt: str
    tools_enabled: bool = True
    # If set, only these tools are sent to the model (saves tokens).
    tool_names: tuple[str, ...] | None = None
    # Research topic binder may attach a tiny continuity stub.
    allows_topic: bool = False
    # "oac" = live read-only orchestrator; "soi" = dormant filer with mutations
    role: str = "oac"
    # If False, never unlock or dispatch mutating tools (OAC).
    allow_mutations: bool = False
