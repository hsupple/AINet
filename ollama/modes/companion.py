"""OAC companion mode — default live talk (read-only)."""

from ollama.modes.base import META_TOOLS, PROJECT_TOOLS, READ_TOOLS, Mode
from ollama.prompts import companion as companion_prompt

MODE = Mode(
    id="companion",
    name="Companion",
    description="OAC live talk (spoken). Read-only tools + project create/open.",
    prompt=companion_prompt.PROMPT,
    tools_enabled=True,
    tool_names=READ_TOOLS + PROJECT_TOOLS + META_TOOLS,
    role="oac",
    allow_mutations=False,
)
