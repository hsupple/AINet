"""OAC Deep Research mode — search, cite, save a brief under Questions/Research."""

from ollama.modes.base import META_TOOLS, READ_TOOLS, Mode
from ollama.prompts import deep_research as deep_research_prompt

MODE = Mode(
    id="deep_research",
    name="Deep Research",
    description="Search reputable sources, write a cited 1–2 pager, save for SOI.",
    prompt=deep_research_prompt.PROMPT,
    tools_enabled=True,
    tool_names=READ_TOOLS + META_TOOLS + ("save_research", "inspect_research"),
    role="oac",
    allow_mutations=False,
)
