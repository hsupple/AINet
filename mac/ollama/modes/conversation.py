"""OAC conversation mode — long-form dialogue (read-only)."""

from ollama.modes.base import META_TOOLS, READ_TOOLS, Mode
from ollama.prompts import conversation as conversation_prompt

MODE = Mode(
    id="conversation",
    name="Conversation",
    description="OAC long-form dialogue. Read-only tools.",
    prompt=conversation_prompt.PROMPT,
    tools_enabled=True,
    tool_names=READ_TOOLS + META_TOOLS,
    role="oac",
    allow_mutations=False,
)
