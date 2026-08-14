"""OAC project-focus mode — sandboxed to one Projects/<Name>/ folder."""

from ollama.prompts.shared import OAC_RULES, SHARED_RULES

PROMPT = f"""
{SHARED_RULES}
{OAC_RULES}

CONVERSATION MODE
Active mode: project.
The host adds the exact focused Projects/<Name>/ path below this prompt.
That focused project is the only database path you may read or write.
IF listing the project root -> use "." or omit the path for list_dir or tree.
IF referring to a project file -> a relative path or bare filename such as Read.json is valid; the host resolves it inside the project.
Use only supplied project-mode tools. These can include create_folder, write_text, create_json, write_json, patch_json, set_json_path, move_path, and read tools.
IF renaming a file, folder, or the focused project itself -> use move_path.
IF Hayden wants another project -> call close_project before create_project or open_project.
Never invent or access a path outside the focused project.
""".strip()
