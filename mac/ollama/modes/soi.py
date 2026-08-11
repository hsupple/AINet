"""SOI mode — Slave of Information (AI2 dormant filer)."""

from ollama.modes.base import Mode
from ollama.prompts import soi as soi_prompt

MODE = Mode(
    id="soi",
    name="SOI",
    description="AI2 Slave of Information. Idle filer: changelog + inbox → DB (full tools).",
    prompt=soi_prompt.PROMPT,
    tools_enabled=True,
    tool_names=(
        "list_dir",
        "tree",
        "read_json",
        "create_cop",
        "create_folder",
        "write_json",
        "patch_json",
        "create_json",
        "file_by_id",
    ),
    role="soi",
    allow_mutations=True,
)
