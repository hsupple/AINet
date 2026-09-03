"""OAC — single live-talk personality (read-only + project session tools)."""

from ollama.modes.base import CALENDAR_WRITE, META_TOOLS, PROJECT_TOOLS, READ_TOOLS, Mode
from ollama.prompts import companion as companion_prompt

MODE = Mode(
    id="companion",
    name="OAC",
    description="Hayden's live conversational interface. One personality; tools as needed.",
    prompt=companion_prompt.PROMPT,
    tools_enabled=True,
    tool_names=READ_TOOLS + PROJECT_TOOLS + META_TOOLS + CALENDAR_WRITE,
    role="oac",
    allow_mutations=False,
)
