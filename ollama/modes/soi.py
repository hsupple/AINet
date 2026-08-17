"""SOI mode — Slave of Information (AI2 dormant filer)."""

from ollama.modes.base import Mode
from ollama.prompts import soi as soi_prompt

MODE = Mode(
    id="soi",
    name="SOI",
    description="AI2 Slave of Information. Idle filer: changelog → log_item.",
    prompt=soi_prompt.PROMPT,
    tools_enabled=True,
    tool_names=("log_item",),
    role="soi",
    allow_mutations=True,
)
