"""SOI test harness modes — separate prompts, same tool surface as production SOI."""

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

MODE_P2 = Mode(
    id="soi_test_p2",
    name="SOI Test Phase 2",
    description="Developer SOI harness — read refresh / compaction with test prompt.",
    prompt=soi_test_prompt.PROMPT_P2,
    tools_enabled=True,
    tool_names=(
        "refresh_read",
        "patch_json",
    ),
    role="soi",
    allow_mutations=True,
)
