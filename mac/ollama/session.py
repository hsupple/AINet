"""Chat session: OAC (read-only live) + optional SOI persistence hooks."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from ainet.tools.ops import DatabaseTools
from ainet.tools.registry import catalog_tools, dispatch, tools_subset
from ollama.client import OllamaClient, ThinkingCallback, TokenCallback
from ollama.config import OllamaConfig
from ollama.conversation_store import ConversationStore
from ollama.modes import get_mode
from ollama.modes.base import QUIZ_TOOLS, READ_TOOLS, Mode
from ollama.router import suggest_mode
from ollama.topics import ensure_topic, load_topic_context

# on_tool(phase, name, detail) — phase is "start" | "done"
ToolCallback = Callable[[str, str, dict[str, Any]], None]


_MUTATING = {
    "write_json",
    "create_json",
    "patch_json",
    "set_json_path",
    "create_folder",
    "create_cop",
    "move_path",
    "archive_to_history",
    "append_changelog",
    "capture_inbox",
    "upsert_research_session",
    "complete_research_session",
}

# Host-owned quiz helpers — OAC may call these even when allow_mutations=False.
_OAC_QUIZ = set(QUIZ_TOOLS)


class ChatSession:
    def __init__(
        self,
        mode: Mode,
        config: OllamaConfig | None = None,
        client: OllamaClient | None = None,
        topic_title: str | None = None,
        *,
        auto_mode: bool | None = None,
        persist_conversation: bool | None = None,
        resume_session: bool = True,
    ) -> None:
        self.config = config or OllamaConfig.from_env()
        self.client = client or OllamaClient(self.config)
        self.db = DatabaseTools(self.config.db_root)
        self.mode = mode
        self.auto_mode = self.config.auto_mode if auto_mode is None else auto_mode
        self.mode_locked = False
        self.full_tools_unlocked = False
        self.topic: dict[str, Any] | None = None
        self.messages: list[dict[str, Any]] = []
        self.last_activity = time.monotonic()
        self.persist_conversation = (
            self.config.persist_oac_conversation
            if persist_conversation is None
            else persist_conversation
        )
        self.store: ConversationStore | None = None
        self.session_id: str | None = None
        if self.persist_conversation and mode.role == "oac":
            self.store = ConversationStore(self.config.db_root)
            self.session_id = self.store.ensure_session(
                mode_id=mode.id,
                topic=topic_title,
            )
        self._rebuild_system()
        if resume_session and self.store and self.session_id:
            prior = self.store.turns_as_messages(
                self.session_id,
                limit=max(2, self.config.max_history_messages // 2),
            )
            self.messages.extend(prior)
        if topic_title:
            self.bind_topic(topic_title)

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_activity

    def _active_tools(self) -> list[dict[str, Any]] | None:
        if not self.mode.tools_enabled:
            return None
        if not self.mode.allow_mutations:
            # OAC: read + web + quiz helpers only (never general writes)
            if self.full_tools_unlocked:
                return tools_subset(READ_TOOLS + QUIZ_TOOLS + ("get_tools",))
            return tools_subset(self.mode.tool_names or READ_TOOLS + QUIZ_TOOLS + ("get_tools",))
        if self.full_tools_unlocked or self.mode.tool_names is None:
            return tools_subset(None)
        return tools_subset(self.mode.tool_names)

    def _rebuild_system(self) -> None:
        system: list[dict[str, Any]] = [{"role": "system", "content": self.mode.prompt}]
        if self.topic and self.mode.allows_topic:
            system.append(
                {
                    "role": "system",
                    "content": (
                        f"Topic '{self.topic.get('title')}' bound. "
                        f"{load_topic_context(self.db, self.topic['slug'], lean=self.config.lean_topic_context)}"
                    ),
                }
            )
        dialogue = [m for m in self.messages if m.get("role") != "system"]
        self.messages = system + dialogue

    def reset(self) -> None:
        self.messages = []
        self.full_tools_unlocked = False
        if self.store and self.mode.role == "oac":
            self.session_id = self.store.new_session(
                mode_id=self.mode.id,
                topic=self.topic["title"] if self.topic else None,
            )
        self._rebuild_system()
        self.touch()

    def set_mode(self, mode_id: str, *, lock: bool = False) -> Mode:
        self.mode = get_mode(mode_id)
        self.mode_locked = lock
        self.full_tools_unlocked = False
        self._rebuild_system()
        self.touch()
        return self.mode

    def bind_topic(self, title: str) -> dict[str, Any]:
        self.topic = ensure_topic(self.db, title)
        if not self.mode.allows_topic:
            self.set_mode("research", lock=self.mode_locked)
        else:
            self._rebuild_system()
        self.touch()
        return self.topic

    def ask(
        self,
        user_text: str,
        *,
        stream: bool = False,
        on_token: TokenCallback | None = None,
        on_thinking: ThinkingCallback | None = None,
        on_tool: ToolCallback | None = None,
    ) -> str:
        self.touch()
        route_note = self._maybe_autoroute(user_text)
        if route_note and on_token is not None:
            # Stream path: emit route note as its own line before tokens.
            on_token(f"{route_note}\n")

        self.messages.append({"role": "user", "content": user_text})
        self._trim_history()

        tools = self._active_tools()
        final_text = ""
        streamed_any = False
        think = self.config.soi_think if self.mode.role == "soi" else self.config.oac_think
        # SOI jobs (esp. with thinking) need a longer socket timeout.
        req_timeout = (
            self.config.soi_timeout_s if self.mode.role == "soi" else self.config.timeout_s
        )

        def _token(delta: str) -> None:
            nonlocal streamed_any
            streamed_any = True
            if on_token:
                on_token(delta)

        for _ in range(self.config.max_tool_rounds):
            if stream or on_token or on_thinking:
                response = self.client.chat_stream(
                    self.messages,
                    tools=tools,
                    think=think,
                    on_token=_token if on_token else None,
                    on_thinking=on_thinking,
                    timeout_s=req_timeout,
                )
            else:
                response = self.client.chat(
                    self.messages,
                    tools=tools,
                    think=think,
                    timeout_s=req_timeout,
                )
            message = response.get("message") or {}
            self.messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                final_text = (message.get("content") or "").strip()
                break

            # Finish the streamed line before tool banners.
            if streamed_any and on_token:
                on_token("\n")
                streamed_any = False
            elif streamed_any and on_thinking and not on_token:
                # Thinking-only stream still needs a visual break before tools.
                on_thinking("\n")

            for call in tool_calls:
                name, args = self._tool_call_parts(call)
                if on_tool:
                    on_tool("start", name, {"arguments": args})
                result = self._run_tool_call(call)
                if name in {"get_tools", "getTools"}:
                    self.full_tools_unlocked = True
                    tools = self._active_tools()
                if on_tool:
                    on_tool(
                        "done",
                        name,
                        {
                            "ok": bool(result.get("ok", True)),
                            "summary": self._tool_result_summary(name, result),
                        },
                    )
                payload = self._truncate_tool_result(result)
                self.messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(payload, ensure_ascii=False),
                    }
                )
            self._trim_history()
        else:
            final_text = "I hit the tool-call limit for this turn. Try again with a narrower ask."

        if self.store and self.session_id and self.mode.role == "oac":
            self.store.append_turn(
                self.session_id,
                user_text=user_text,
                assistant_text=final_text,
                mode_id=self.mode.id,
                topic=self.topic["title"] if self.topic else None,
            )

        self.touch()
        # When streaming, tokens were already printed — return text for callers/store only.
        if stream and on_token is not None:
            if route_note and final_text:
                return f"{route_note}\n{final_text}"
            return final_text or route_note or ""
        if route_note:
            return f"{route_note}\n{final_text}" if final_text else route_note
        return final_text

    @staticmethod
    def _tool_call_parts(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        fn = call.get("function") or {}
        name = str(fn.get("name") or "")
        raw_args = fn.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        return name, args if isinstance(args, dict) else {}

    @staticmethod
    def _tool_result_summary(name: str, result: dict[str, Any]) -> str:
        if not result.get("ok", True) and result.get("error"):
            return str(result["error"])[:160]
        if name == "web_search":
            n = result.get("count")
            if n is None and isinstance(result.get("results"), list):
                n = len(result["results"])
            return f"{n} hits" if n is not None else "ok"
        if name == "web_fetch":
            text = result.get("text") or ""
            return f"{len(text)} chars"
        if name in {"read_json", "read_text"}:
            path = result.get("path") or ""
            return path or "ok"
        if name in {"list_dir", "tree"}:
            kids = result.get("children") or result.get("entries")
            if isinstance(kids, list):
                return f"{len(kids)} entries"
        return "ok"

    def _maybe_autoroute(self, user_text: str) -> str | None:
        if not self.auto_mode or self.mode_locked or self.mode.role != "oac":
            return None
        decision = suggest_mode(user_text, self.mode.id)
        if (
            decision.mode_id != self.mode.id
            and decision.confidence >= self.config.auto_mode_min_confidence
            and decision.mode_id != "soi"
        ):
            old = self.mode.id
            self.set_mode(decision.mode_id, lock=False)
            return f"(auto → {decision.mode_id}; was {old}; {decision.reason})"
        return None

    def _trim_history(self) -> None:
        system = [m for m in self.messages if m.get("role") == "system"]
        rest = [m for m in self.messages if m.get("role") != "system"]
        limit = max(4, self.config.max_history_messages)
        if len(rest) > limit:
            rest = rest[-limit:]
        self.messages = system + rest

    def _truncate_tool_result(self, result: dict[str, Any]) -> dict[str, Any]:
        raw = json.dumps(result, ensure_ascii=False)
        limit = self.config.max_tool_result_chars
        if len(raw) <= limit:
            return result
        return {
            "ok": result.get("ok", True),
            "truncated": True,
            "error": None,
            "preview": raw[:limit] + "…",
            "hint": "Result truncated for tokens. Re-read a narrower path if needed.",
        }

    def _run_tool_call(self, call: dict[str, Any]) -> dict[str, Any]:
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        raw_args = fn.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                return {"ok": False, "error": f"Invalid tool arguments JSON: {raw_args}"}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            return {"ok": False, "error": f"Unsupported arguments type: {type(raw_args).__name__}"}

        if not name:
            return {"ok": False, "error": "Tool call missing function name"}

        if name in {"get_tools", "getTools"}:
            return catalog_tools(
                detail=bool(args.get("detail", True)),
                read_only=not self.mode.allow_mutations,
            )

        if not self.mode.allow_mutations:
            allowed = set(READ_TOOLS) | _OAC_QUIZ | {"get_tools", "getTools"}
            if self.mode.tool_names:
                allowed |= set(self.mode.tool_names)
            if name not in allowed or name in _MUTATING:
                return {
                    "ok": False,
                    "error": (
                        f"OAC cannot use tool '{name}'. "
                        "Allowed: read/web + quiz helpers (should_suggest_quiz, "
                        "list_quiz_candidates, start_quiz, record_quiz_answer, get_quiz_status). "
                        "SOI files lasting DB writes from the changelog after idle."
                    ),
                }
        elif (
            not self.full_tools_unlocked
            and self.mode.tool_names is not None
            and name not in self.mode.tool_names
            and name not in {"get_tools", "getTools"}
        ):
            return {
                "ok": False,
                "error": (
                    f"Tool '{name}' not in the lean set for mode '{self.mode.id}'. "
                    "Call get_tools to unlock the full catalog."
                ),
            }

        result = dispatch(self.db, name, args)
        if self.mode.role == "soi":
            result = _redact_assistant_fields(result)
        return result


def _redact_assistant_fields(obj: Any) -> Any:
    """SOI never sees OAC assistant_text, even via tool reads."""
    if isinstance(obj, dict):
        return {
            k: _redact_assistant_fields(v)
            for k, v in obj.items()
            if k not in {"assistant_text", "assistant"}
        }
    if isinstance(obj, list):
        return [_redact_assistant_fields(x) for x in obj]
    return obj
