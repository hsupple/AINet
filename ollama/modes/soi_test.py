"""SOI test harness modes."""

from ollama.modes.base import Mode
from ollama.prompts import soi_test as soi_test_prompt

MODE = Mode(
    id="soi_test",
    name="SOI Test",
    description="Developer SOI harness — isolated filing run.",
    prompt=soi_test_prompt.PROMPT,
    tools_enabled=True,
    tool_names=("log_item",),
    role="soi",
    allow_mutations=True,
)

MODE_P2 = Mode(
    id="soi_test_p2",
    name="SOI Test Phase 2",
    description="Removed — knowledge files need no compaction.",
    prompt=soi_test_prompt.PROMPT_P2,
    tools_enabled=False,
    tool_names=(),
    role="soi",
    allow_mutations=False,
)
