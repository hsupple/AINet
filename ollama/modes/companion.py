"""OAC companion mode — default live talk (read-only)."""

from ollama.modes.base import META_TOOLS, READ_TOOLS, Mode
from ollama.prompts import companion as companion_prompt

MODE = Mode(
    id="companion",
    name="Companion",
    description="OAC live talk (spoken). Read-only tools.",
    prompt=companion_prompt.PROMPT,
    tools_enabled=True,
    tool_names=READ_TOOLS + META_TOOLS,
    allows_topic=False,
    role="oac",
    allow_mutations=False,
)
