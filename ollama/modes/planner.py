"""OAC planner mode — planning discussion (read-only)."""

from ollama.modes.base import META_TOOLS, PROJECT_TOOLS, READ_TOOLS, Mode
from ollama.prompts import planner as planner_prompt

MODE = Mode(
    id="planner",
    name="Planner",
    description="OAC planning discussion. Read-only + project create/open; SOI files plan edits later.",
    prompt=planner_prompt.PROMPT,
    tools_enabled=True,
    tool_names=READ_TOOLS + PROJECT_TOOLS + META_TOOLS,
    role="oac",
    allow_mutations=False,
)
