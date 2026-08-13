"""OAC project-focus mode — mutations allowed only inside the focused project."""

from ollama.modes.base import META_TOOLS, READ_TOOLS, WRITE_LIGHT, Mode
from ollama.prompts import project as project_prompt

_PROJECT_TOOLS = (
    "create_folder",
    "write_text",
    "move_path",
    "create_project",
    "list_projects",
    "open_project",
    "close_project",
)

MODE = Mode(
    id="project",
    name="Project",
    description="Focused on one Projects/<Name>/ folder (create folders, text, rename, read).",
    prompt=project_prompt.PROMPT,
    tools_enabled=True,
    tool_names=READ_TOOLS + WRITE_LIGHT + _PROJECT_TOOLS + META_TOOLS,
    role="oac",
    allow_mutations=True,
)
