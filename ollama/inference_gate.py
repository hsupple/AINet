"""Serialize Ollama inference so OAC and SOI do not pile onto one GPU request."""

from __future__ import annotations

import threading

# One local Ollama instance — avoid overlapping chat/SOI requests that wedge the UI.
INFERENCE_GATE = threading.RLock()
