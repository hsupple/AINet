"""Runtime configuration for the local Ollama server (Windows PC)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def default_db_root() -> Path:
    return Path(__file__).resolve().parent.parent / "db"


@dataclass(frozen=True)
class OllamaConfig:
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen3:8b"
    # Qwen3 thinking: always off by default (OAC + SOI)
    oac_think: bool = False
    soi_think: bool = False
    db_root: Path = field(default_factory=default_db_root)
    timeout_s: float = 120.0
    # SOI with tools can still take a while on 8B
    soi_timeout_s: float = 600.0
    max_tool_rounds: int = 8
    # Token discipline
    auto_mode: bool = True
    auto_mode_min_confidence: float = 0.7
    max_history_messages: int = 16
    max_tool_result_chars: int = 1500
    lean_topic_context: bool = True
    # Dual-AI
    persist_oac_conversation: bool = True
    soi_enabled: bool = True
    # Phase 1: wake SOI filing when OAC quiet this long
    soi_idle_seconds: float = 45.0
    # Phase 2: refresh Read.json after filing clear + this long OAC idle
    soi_read_refresh_idle_seconds: float = 600.0

    @classmethod
    def from_env(cls) -> OllamaConfig:
        db = os.environ.get("AINET_DB")
        return cls(
            host=os.environ.get("AINET_OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"),
            model=os.environ.get("AINET_OLLAMA_MODEL", "qwen3:8b"),
            oac_think=os.environ.get("AINET_OAC_THINK", "0") not in {"0", "false", "False"},
            soi_think=os.environ.get("AINET_SOI_THINK", "0") not in {"0", "false", "False"},
            db_root=Path(db) if db else default_db_root(),
            timeout_s=float(os.environ.get("AINET_OLLAMA_TIMEOUT", "120")),
            soi_timeout_s=float(os.environ.get("AINET_SOI_TIMEOUT", "600")),
            max_tool_rounds=int(os.environ.get("AINET_OLLAMA_MAX_TOOL_ROUNDS", "8")),
            auto_mode=os.environ.get("AINET_AUTO_MODE", "1") not in {"0", "false", "False"},
            auto_mode_min_confidence=float(os.environ.get("AINET_AUTO_MODE_MIN_CONF", "0.7")),
            max_history_messages=int(os.environ.get("AINET_MAX_HISTORY_MESSAGES", "16")),
            max_tool_result_chars=int(os.environ.get("AINET_MAX_TOOL_RESULT_CHARS", "1500")),
            lean_topic_context=os.environ.get("AINET_LEAN_TOPIC", "1") not in {"0", "false", "False"},
            persist_oac_conversation=os.environ.get("AINET_PERSIST_OAC", "1")
            not in {"0", "false", "False"},
            soi_enabled=os.environ.get("AINET_SOI_ENABLED", "1") not in {"0", "false", "False"},
            soi_idle_seconds=float(os.environ.get("AINET_SOI_IDLE_SECONDS", "45")),
            soi_read_refresh_idle_seconds=float(
                os.environ.get("AINET_SOI_READ_REFRESH_IDLE_SECONDS", "600")
            ),
        )
