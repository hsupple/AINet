"""SOI — Slave of Information (AI2 dormant filer)."""

from ollama.prompts.shared import SOI_RULES

PROMPT = f"""
{SOI_RULES}

Mode: soi.
""".strip()
