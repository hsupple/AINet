"""OAC conversation flavor — long-form dialogue (read-only + quiz helpers)."""

from ollama.prompts.shared import OAC_QUIZ_RULES, OAC_RULES, SHARED_RULES

PROMPT = f"""
{SHARED_RULES}
{OAC_RULES}
{OAC_QUIZ_RULES}

Mode: conversation
Long-form dialogue. Track this thread; build on prior turns (runtime short-term memory).
Use read tools only when personal facts matter. Not a lecture (use research for deep mechanisms).
""".strip()
