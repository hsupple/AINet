"""OAC short-term conversation persistence (runtime-owned, not AI write tools)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ainet.tools import changelog
from ainet.tools.fsutil import atomic_write_text
from ainet.tools.paths import DbPaths


# Keep changelog usable; still store far more than the old 240-char preview.
_MAX_ASSISTANT_CHARS = 8000


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ConversationStore:
    """Persists OAC turns under db/runtime/oac/ and stacks full Changelog handoffs for SOI."""

    def __init__(self, db_root: Path | str) -> None:
        self.paths = DbPaths(db_root)
        changelog.ensure_changelog_file(self.paths.root)
        changelog.ensure_masterlog_file(self.paths.root)
        changelog.migrate_resolved_to_masterlog(self.paths)
        self.root = self.paths.root / "runtime" / "oac"
        self.sessions_dir = self.root / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.current_path = self.root / "current.json"

    def new_session(self, *, mode_id: str, topic: str | None = None) -> str:
        session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        data = {
            "id": session_id,
            "mode_id": mode_id,
            "topic": topic,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "turns": [],
        }
        self._write_session(session_id, data)
        self._set_current(session_id)
        return session_id

    def load_session(self, session_id: str) -> dict[str, Any]:
        path = self.sessions_dir / f"{session_id}.json"
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def current_session_id(self) -> str | None:
        if not self.current_path.exists():
            return None
        try:
            data = json.loads(self.current_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return data.get("id")

    def ensure_session(self, *, mode_id: str, topic: str | None = None) -> str:
        sid = self.current_session_id()
        if sid and (self.sessions_dir / f"{sid}.json").exists():
            return sid
        return self.new_session(mode_id=mode_id, topic=topic)

    def session_exists(self, session_id: str | None) -> bool:
        if not session_id:
            return False
        return (self.sessions_dir / f"{session_id}.json").exists()

    def append_turn(
        self,
        session_id: str,
        *,
        user_text: str,
        assistant_text: str,
        mode_id: str,
        topic: str | None = None,
    ) -> dict[str, Any]:
        # Heal after db reset / deleted runtime while the chat process is still up.
        if not self.session_exists(session_id):
            session_id = self.new_session(mode_id=mode_id, topic=topic)

        data = self.load_session(session_id)
        turn = {
            "ts": _utc_now(),
            "mode_id": mode_id,
            "topic": topic,
            "user": user_text,
            "assistant": assistant_text,
        }
        data.setdefault("turns", []).append(turn)
        data["updated_at"] = turn["ts"]
        data["mode_id"] = mode_id
        if topic is not None:
            data["topic"] = topic
        self._write_session(session_id, data)
        self._set_current(session_id)

        # Full user utterance stays on Changelog until SOI files or discards it.
        user_full = user_text or ""
        assistant_full = assistant_text or ""
        if len(assistant_full) > _MAX_ASSISTANT_CHARS:
            assistant_full = assistant_full[:_MAX_ASSISTANT_CHARS] + "…"

        changelog.append_entry(
            self.paths,
            action="oac_turn",
            path=f"runtime/oac/sessions/{session_id}.json",
            summary=user_full[:240] if user_full else "(empty user turn)",
            details={
                "session_id": session_id,
                "mode_id": mode_id,
                "topic": topic,
                "user_text": user_full,
                "assistant_text": assistant_full,
            },
            actor="oac",
            soi_status="pending",
        )
        return turn

    def recent_turns(self, session_id: str, *, limit: int = 8) -> list[dict[str, Any]]:
        data = self.load_session(session_id)
        turns = data.get("turns") or []
        if not isinstance(turns, list):
            return []
        return turns[-limit:]

    def turns_as_messages(self, session_id: str, *, limit: int = 8) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for turn in self.recent_turns(session_id, limit=limit):
            if turn.get("user"):
                messages.append({"role": "user", "content": str(turn["user"])})
            if turn.get("assistant"):
                messages.append({"role": "assistant", "content": str(turn["assistant"])})
        return messages

    def _write_session(self, session_id: str, data: dict[str, Any]) -> None:
        path = self.sessions_dir / f"{session_id}.json"
        atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    def _set_current(self, session_id: str) -> None:
        atomic_write_text(
            self.current_path,
            json.dumps({"id": session_id, "updated_at": _utc_now()}, indent=2) + "\n",
        )
