"""Ollama runtime for AINet — chat modes, prompts, tool loop."""

from ollama.config import OllamaConfig
from ollama.modes import get_mode, list_modes

__all__ = ["OllamaConfig", "get_mode", "list_modes"]
