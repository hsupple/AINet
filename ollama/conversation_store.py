"""OAC conversation persistence — chat logs under db/Chats/ (host-owned)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ainet.tools import changelog
from ainet.tools.fsutil import atomic_write_text
from ainet.tools.paths import DbPaths


_MAX_ASSISTANT_CHARS = 8000
_TITLE_CHARS = 56


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_session_id(session_id: str) -> str:
    text = str(session_id or "").strip()
    if not text or "/" in text or "\\" in text or ".." in text:
        raise ValueError("invalid session id")
    return text


class ConversationStore:
    """Persists OAC chats under db/Chats/ and stacks Changelog handoffs for SOI."""

    def __init__(self, db_root: Path | str) -> None:
        self.paths = DbPaths(db_root)
        changelog.ensure_changelog_file(self.paths.root)
        changelog.ensure_masterlog_file(self.paths.root)
        changelog.migrate_resolved_to_masterlog(self.paths)
        self.root = self.paths.root / "Chats"
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions_dir = self.root
        self.current_path = self.root / "current.json"
        self.legacy_dir = self.paths.root / "runtime" / "oac" / "sessions"
        self._migrate_legacy()

    def _migrate_legacy(self) -> None:
        """Copy older runtime/oac/sessions logs into db/Chats/ once."""
        if not self.legacy_dir.is_dir():
            return
        for path in self.legacy_dir.glob("*.json"):
            dest = self.root / path.name
            if dest.exists():
                continue
            try:
                dest.write_bytes(path.read_bytes())
            except OSError:
                continue
        legacy_current = self.paths.root / "runtime" / "oac" / "current.json"
        if legacy_current.is_file() and not self.current_path.exists():
            try:
                self.current_path.write_bytes(legacy_current.read_bytes())
            except OSError:
                pass

    def new_session(self, *, mode_id: str, topic: str | None = None) -> str:
        self._drop_empty_session(self.current_session_id())
        session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        data = {
            "id": session_id,
            "mode_id": mode_id,
            "topic": topic,
            "title": "",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "memory": "",
            "turns": [],
        }
        self._write_session(session_id, data)
        self._set_current(session_id)
        return session_id

    def load_session(self, session_id: str) -> dict[str, Any]:
        path = self.sessions_dir / f"{_safe_session_id(session_id)}.json"
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("chat file is not an object")
        return data

    def current_session_id(self) -> str | None:
        if not self.current_path.exists():
            return None
        try:
            data = json.loads(self.current_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        sid = data.get("id") if isinstance(data, dict) else None
        return str(sid) if sid else None

    def ensure_session(self, *, mode_id: str, topic: str | None = None) -> str:
        sid = self.current_session_id()
        if sid and (self.sessions_dir / f"{sid}.json").exists():
            return sid
        return self.new_session(mode_id=mode_id, topic=topic)

    def session_exists(self, session_id: str | None) -> bool:
        if not session_id:
            return False
        try:
            sid = _safe_session_id(session_id)
        except ValueError:
            return False
        return (self.sessions_dir / f"{sid}.json").exists()

    def append_turn(
        self,
        session_id: str,
        *,
        user_text: str,
        assistant_text: str,
        mode_id: str,
        topic: str | None = None,
        memory: str | None = None,
    ) -> dict[str, Any]:
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
        if memory is not None:
            data["memory"] = str(memory)
        if topic is not None:
            data["topic"] = topic
        if not str(data.get("title") or "").strip():
            data["title"] = _title_from_text(user_text)
        self._write_session(session_id, data)
        self._set_current(session_id)

        user_full = user_text or ""
        assistant_full = assistant_text or ""
        if len(assistant_full) > _MAX_ASSISTANT_CHARS:
            assistant_full = assistant_full[:_MAX_ASSISTANT_CHARS] + "…"

        changelog.append_entry(
            self.paths,
            action="oac_turn",
            path=f"Chats/{session_id}.json",
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

    def load_memory(self, session_id: str) -> str:
        try:
            data = self.load_session(session_id)
        except (OSError, json.JSONDecodeError, FileNotFoundError, ValueError):
            return ""
        return str(data.get("memory") or "").strip()

    def turns_as_messages(self, session_id: str, *, limit: int = 8) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for turn in self.recent_turns(session_id, limit=limit):
            if turn.get("user"):
                messages.append({"role": "user", "content": str(turn["user"])})
            if turn.get("assistant"):
                messages.append({"role": "assistant", "content": str(turn["assistant"])})
        return messages

    def list_sessions(self, *, current_id: str | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        current = current_id or self.current_session_id()
        for path in self.sessions_dir.glob("*.json"):
            if path.name == "current.json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            sid = str(data.get("id") or path.stem)
            turns = data.get("turns") if isinstance(data.get("turns"), list) else []
            if not turns and sid != current:
                continue
            rows.append(self._summary(data, sid, current=current))
        rows.sort(key=lambda r: str(r.get("updated_at") or ""), reverse=True)
        return rows

    def session_payload(self, session_id: str) -> dict[str, Any]:
        data = self.load_session(session_id)
        sid = str(data.get("id") or session_id)
        current = self.current_session_id()
        summary = self._summary(data, sid, current=current)
        turns = data.get("turns") if isinstance(data.get("turns"), list) else []
        summary["turns"] = [
            {
                "ts": t.get("ts"),
                "user": t.get("user") or "",
                "assistant": t.get("assistant") or "",
            }
            for t in turns
            if isinstance(t, dict)
        ]
        summary["memory"] = str(data.get("memory") or "")
        summary["mode_id"] = data.get("mode_id") or ""
        return summary

    def _summary(self, data: dict[str, Any], sid: str, *, current: str | None) -> dict[str, Any]:
        turns = data.get("turns") if isinstance(data.get("turns"), list) else []
        title = str(data.get("title") or "").strip()
        if not title:
            for turn in turns:
                if isinstance(turn, dict) and turn.get("user"):
                    title = _title_from_text(str(turn.get("user") or ""))
                    break
        if not title:
            title = "New chat"
        preview = ""
        if turns and isinstance(turns[-1], dict):
            preview = str(turns[-1].get("user") or turns[-1].get("assistant") or "").strip()
        return {
            "id": sid,
            "title": title,
            "created_at": data.get("created_at") or "",
            "updated_at": data.get("updated_at") or data.get("created_at") or "",
            "turn_count": len(turns),
            "preview": preview[:120],
            "current": sid == current,
        }

    def _drop_empty_session(self, session_id: str | None) -> None:
        """Remove a current chat that never got a turn so reopen does not pile empties."""
        if not session_id or not self.session_exists(session_id):
            return
        try:
            data = self.load_session(session_id)
        except (OSError, json.JSONDecodeError, FileNotFoundError, ValueError):
            return
        turns = data.get("turns") if isinstance(data.get("turns"), list) else []
        if turns:
            return
        try:
            (self.sessions_dir / f"{_safe_session_id(session_id)}.json").unlink()
        except OSError:
            pass

    def set_current(self, session_id: str) -> None:
        self._set_current(session_id)

    def _write_session(self, session_id: str, data: dict[str, Any]) -> None:
        path = self.sessions_dir / f"{_safe_session_id(session_id)}.json"
        atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    def _set_current(self, session_id: str) -> None:
        atomic_write_text(
            self.current_path,
            json.dumps({"id": session_id, "updated_at": _utc_now()}, indent=2) + "\n",
        )


def _title_from_text(text: str) -> str:
    line = " ".join((text or "").strip().split())
    if not line:
        return ""
    if len(line) <= _TITLE_CHARS:
        return line
    return line[: _TITLE_CHARS - 1].rstrip() + "…"
