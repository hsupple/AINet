"""SOI test harness prompt — filing via log_item."""

from ollama.prompts.shared import SOI_RULES

PROMPT = f"""
{SOI_RULES}

Mode: filing.
""".strip()

PROMPT_P2 = """
IDENTITY
You are AI2. Phase 2 compaction is gone. The knowledge files are already the source of truth.
Do not call any tools. Reply with JSON: {"ok": true, "skipped": true}
""".strip()

FILING_INSTRUCTIONS = "File this batch with log_item."

READ_REFRESH_INSTRUCTIONS = "Phase 2 is removed. Skip."
