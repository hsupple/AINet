"""SOI mode — Slave of Information (AI2 dormant filer)."""

from ollama.modes.base import Mode
from ollama.prompts import soi as soi_prompt

MODE = Mode(
    id="soi",
    name="SOI",
    description="AI2 Slave of Information. Idle filer: changelog + inbox → DB (full tools).",
    prompt=soi_prompt.PROMPT,
    tools_enabled=True,
    tool_names=None,  # full catalog
    allows_topic=False,
    role="soi",
    allow_mutations=True,
)
