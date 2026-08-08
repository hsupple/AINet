"""OAC research mode — deep topic exploration (read-only)."""

from ollama.modes.base import META_TOOLS, READ_TOOLS, Mode
from ollama.prompts import research as research_prompt

MODE = Mode(
    id="research",
    name="Research",
    description="OAC deep rabbit holes. Read-only; SOI files lasting notes later.",
    prompt=research_prompt.PROMPT,
    tools_enabled=True,
    tool_names=READ_TOOLS + META_TOOLS,
    allows_topic=True,
    role="oac",
    allow_mutations=False,
)
