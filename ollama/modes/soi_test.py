"""SOI test harness mode — separate prompt, same tool surface as production SOI."""

from ollama.modes.base import Mode
from ollama.prompts import soi_test as soi_test_prompt

MODE = Mode(
    id="soi_test",
    name="SOI Test",
    description="Developer SOI harness — isolated filing run with test prompt.",
    prompt=soi_test_prompt.PROMPT,
    tools_enabled=True,
    tool_names=(
        "list_dir",
        "tree",
        "read_json",
        "create_folder",
        "file_note",
        "mark_read_refreshed",
    ),
    role="soi",
    allow_mutations=True,
)
