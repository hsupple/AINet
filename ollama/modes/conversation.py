"""OAC conversation mode — long-form dialogue (read-only)."""

from ollama.modes.base import META_TOOLS, PROJECT_TOOLS, READ_TOOLS, Mode
from ollama.prompts import conversation as conversation_prompt

MODE = Mode(
    id="conversation",
    name="Conversation",
    description="OAC long-form dialogue. Read-only tools + project create/open.",
    prompt=conversation_prompt.PROMPT,
    tools_enabled=True,
    tool_names=READ_TOOLS + PROJECT_TOOLS + META_TOOLS,
    role="oac",
    allow_mutations=False,
)
