"""Chat session: OAC (read-only live) + optional SOI persistence hooks."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from typing import Any

from ainet.tools.ops import DatabaseTools
from ainet.tools.registry import catalog_tools, dispatch, tools_subset
from ollama.client import OllamaCancelled, OllamaClient, ThinkingCallback, TokenCallback
from ollama.content_filing import cop_name_in_text
from ollama.content_tools import normalize_soi_tool, parse_content_tool_calls
from ollama.config import OllamaConfig
from ollama.conversation_store import ConversationStore
from ollama.inference_gate import INFERENCE_GATE
from ollama.modes import get_mode
from ollama.modes.base import READ_TOOLS, Mode
from ollama.router import suggest_mode

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
    "file_by_id",
    "file_note",
}


class ChatSession:
    def __init__(
        self,
        mode: Mode,
        config: OllamaConfig | None = None,
        client: OllamaClient | None = None,
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
            self.session_id = self.store.ensure_session(mode_id=mode.id)
        self._rebuild_system()
        if resume_session and self.store and self.session_id:
            prior = self.store.turns_as_messages(
                self.session_id,
                limit=max(2, self.config.max_history_messages // 2),
            )
            self.messages.extend(prior)
        self.cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self.cancel_event.set()
        try:
            self.client.cancel_active()
        except Exception:
            pass

    def clear_cancel(self) -> None:
        self.cancel_event.clear()

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_activity

    def _active_tools(self) -> list[dict[str, Any]] | None:
        if not self.mode.tools_enabled:
            return None
        if self.mode.role == "soi":
            names = self.mode.tool_names or ()
            return [
                t
                for t in tools_subset(names)
                if (t.get("function") or {}).get("name") not in {"get_tools", "getTools"}
            ]
        if not self.mode.allow_mutations:
            if self.full_tools_unlocked:
                return tools_subset(READ_TOOLS + ("get_tools",))
            return tools_subset(self.mode.tool_names or READ_TOOLS + ("get_tools",))
        if self.full_tools_unlocked or self.mode.tool_names is None:
            return tools_subset(None)
        return tools_subset(self.mode.tool_names)

    def _rebuild_system(self) -> None:
        system: list[dict[str, Any]] = [{"role": "system", "content": self.mode.prompt}]
        dialogue = [m for m in self.messages if m.get("role") != "system"]
        self.messages = system + dialogue

    def reset(self) -> None:
        self.messages = []
        self.full_tools_unlocked = False
        if self.store and self.mode.role == "oac":
            self.session_id = self.store.new_session(mode_id=self.mode.id)
        self._rebuild_system()
        self.touch()

    def set_mode(self, mode_id: str, *, lock: bool = False) -> Mode:
        self.mode = get_mode(mode_id)
        self.mode_locked = lock
        self.full_tools_unlocked = False
        self._rebuild_system()
        self.touch()
        return self.mode

    def ask(
        self,
        user_text: str,
        *,
        stream: bool = False,
        on_token: TokenCallback | None = None,
        on_thinking: ThinkingCallback | None = None,
        on_tool: ToolCallback | None = None,
    ) -> str:
        # OAC + SOI share one Ollama — never overlap inference (UI lockups / hung streams).
        with INFERENCE_GATE:
            return self._ask_locked(
                user_text,
                stream=stream,
                on_token=on_token,
                on_thinking=on_thinking,
                on_tool=on_tool,
            )

    def _ask_locked(
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
            on_token(f"{route_note}\n")

        self.clear_cancel()
        self.messages.append({"role": "user", "content": user_text})
        self._trim_history()

        tools = self._active_tools()
        final_text = ""
        streamed_any = False
        streamed_parts: list[str] = []
        self.last_tool_names: list[str] = []
        self.last_mutating_calls: list[dict[str, Any]] = []
        self.last_tool_rounds = 0
        seen_tool_keys: set[str] = set()
        think = self.config.soi_think if self.mode.role == "soi" else self.config.oac_think
        req_timeout = (
            self.config.soi_timeout_s if self.mode.role == "soi" else self.config.timeout_s
        )

        def _token(delta: str) -> None:
            nonlocal streamed_any
            streamed_any = True
            if delta:
                streamed_parts.append(delta)
            if on_token:
                on_token(delta)

        extra_mutating = {
            "mark_read_stale",
            "mark_read_refreshed",
        }
        soi_opts = (
            {"temperature": 0, "num_predict": 900}
            if self.mode.role == "soi"
            else None
        )

        try:
            for _round in range(self.config.max_tool_rounds):
                if self.cancelled():
                    raise OllamaCancelled("Cancelled")
                self._trim_history()
                if stream or on_token or on_thinking:
                    response = self.client.chat_stream(
                        self.messages,
                        tools=tools,
                        think=think,
                        on_token=_token if on_token else None,
                        on_thinking=on_thinking,
                        timeout_s=req_timeout,
                        options=soi_opts,
                        should_cancel=self.cancelled,
                    )
                else:
                    response = self.client.chat(
                        self.messages,
                        tools=tools,
                        think=think,
                        timeout_s=req_timeout,
                        options=soi_opts,
                    )
                message = response.get("message") or {}
                self.messages.append(message)

                tool_calls = message.get("tool_calls") or []
                parsed_from_text = False
                if not tool_calls:
                    final_text = (message.get("content") or "").strip()
                    if self.mode.role == "soi":
                        tool_calls = parse_content_tool_calls(final_text)
                        parsed_from_text = bool(tool_calls)
                    if not tool_calls:
                        break

                self.last_tool_rounds += 1
                if streamed_any and on_token:
                    on_token("\n")
                    streamed_any = False
                elif streamed_any and on_thinking and not on_token:
                    on_thinking("\n")

                for call in tool_calls:
                    if self.cancelled():
                        raise OllamaCancelled("Cancelled")
                    name, args = self._tool_call_parts(call)
                    if self.mode.role == "soi":
                        name, args = normalize_soi_tool(name, args)
                        call = {"function": {"name": name, "arguments": args}}
                    if not name:
                        continue
                    tool_key = self._tool_call_key(name, args)
                    if tool_key in seen_tool_keys:
                        # Model asked for the same tool+args again — don't re-run; nudge it.
                        result = {
                            "ok": True,
                            "duplicate": True,
                            "error": None,
                            "hint": (
                                "You already ran this tool with the same arguments in this turn. "
                                "Use the earlier tool result and answer the user now."
                            ),
                        }
                        if on_tool:
                            on_tool("start", name, {"arguments": args})
                            on_tool(
                                "done",
                                name,
                                {"ok": True, "summary": "duplicate — use prior result"},
                            )
                        self.messages.append(
                            {
                                "role": "tool",
                                "content": json.dumps(result, ensure_ascii=False),
                            }
                        )
                        continue
                    seen_tool_keys.add(tool_key)
                    if on_tool:
                        on_tool("start", name, {"arguments": args})
                    result = self._run_tool_call(call)
                    self.last_tool_names.append(name)
                    if (
                        name == "web_search"
                        and self.mode.role == "oac"
                        and isinstance(result, dict)
                        and result.get("ok")
                        and not result.get("duplicate")
                        and not self._user_opts_out_of_open(user_text)
                    ):
                        result = self._auto_open_from_search(
                            result,
                            on_tool=on_tool,
                            seen_tool_keys=seen_tool_keys,
                        )
                    if name in _MUTATING or name in extra_mutating:
                        self.last_mutating_calls.append(
                            {
                                "tool": name,
                                "args": args,
                                "ok": bool(result.get("ok", True)) if isinstance(result, dict) else True,
                                "result": result if isinstance(result, dict) else {},
                            }
                        )
                    if name in {"get_tools", "getTools"} and self.mode.role != "soi":
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
                if parsed_from_text:
                    break
                self._trim_history()
            else:
                final_text = "I hit the tool-call limit for this turn. Try again with a narrower ask."
        except OllamaCancelled:
            partial = "".join(streamed_parts).strip()
            if partial:
                final_text = partial
                if not (self.messages and self.messages[-1].get("role") == "assistant"):
                    self.messages.append({"role": "assistant", "content": final_text})
            else:
                # Drop the unanswered user turn so a retry is clean.
                if self.messages and self.messages[-1].get("role") == "user":
                    self.messages.pop()
                raise

        if not (final_text or "").strip() and self.last_tool_names:
            chrome_n = sum(1 for n in self.last_tool_names if n == "open_chrome")
            if chrome_n and all(n == "open_chrome" for n in self.last_tool_names):
                final_text = "Opened in Chrome." if chrome_n == 1 else f"Opened {chrome_n} tabs in Chrome."
            elif chrome_n:
                final_text = f"Done — opened {chrome_n} Chrome tab(s)."
            if final_text and on_token is not None:
                on_token(final_text)
            if final_text:
                if self.messages and self.messages[-1].get("role") == "assistant":
                    if not str(self.messages[-1].get("content") or "").strip():
                        self.messages[-1]["content"] = final_text
                else:
                    self.messages.append({"role": "assistant", "content": final_text})

        if self.store and self.mode.role == "oac":
            if not self.session_id or not self.store.session_exists(self.session_id):
                self.session_id = self.store.ensure_session(mode_id=self.mode.id)
            self.store.append_turn(
                self.session_id,
                user_text=user_text,
                assistant_text=final_text,
                mode_id=self.mode.id,
            )
            # append_turn may have minted a new session after a db wipe
            current = self.store.current_session_id()
            if current:
                self.session_id = current

        self.touch()
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
    def _tool_call_key(name: str, args: dict[str, Any]) -> str:
        try:
            raw = json.dumps(args or {}, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            raw = str(args)
        return f"{name}:{raw}"

    @staticmethod
    def _tool_result_summary(name: str, result: dict[str, Any]) -> str:
        if not result.get("ok", True):
            if result.get("error"):
                return str(result["error"])[:160]
            return "failed"
        if name == "web_search":
            n = result.get("count")
            if n is None and isinstance(result.get("results"), list):
                n = len(result["results"])
            base = f"{n} hits" if n is not None else "ok"
            if result.get("cached"):
                base += " (cached)"
            if result.get("duplicate"):
                return "already searched — use prior result"
            opened = result.get("auto_opened")
            if isinstance(opened, list) and opened:
                base += f" · opened {len(opened)}"
            return base
        if name == "web_fetch":
            text = result.get("text") or ""
            return f"{len(text)} chars"
        if name == "open_chrome":
            url = str(result.get("url") or "")
            if url:
                return url if len(url) <= 80 else url[:77] + "…"
            return "ok"
        if name in {"read_json", "read_text"}:
            path = result.get("path") or ""
            return path or "ok"
        if name in {"list_dir", "tree"}:
            kids = result.get("children") or result.get("entries")
            if isinstance(kids, list):
                return f"{len(kids)} entries"
        return "ok"

    @staticmethod
    def _user_opts_out_of_open(user_text: str) -> bool:
        t = (user_text or "").lower()
        phrases = (
            "don't open",
            "do not open",
            "dont open",
            "no browser",
            "don't browse",
            "do not browse",
            "dont browse",
            "just list",
            "links only",
            "no tabs",
            "without opening",
            "don't open chrome",
            "do not open chrome",
        )
        return any(p in t for p in phrases)

    @staticmethod
    def _pick_search_urls_to_open(query: str, results: list[Any], *, limit: int = 1) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url.startswith(("http://", "https://")):
                continue
            title = str(item.get("title") or "").strip()
            rows.append({"title": title, "url": url})
        if not rows:
            return []

        q = (query or "").lower()
        want_video = any(
            w in q for w in ("youtube", "youtu.be", "video", "watch", "tutorial", "how to")
        )
        if want_video:
            yt = [
                r
                for r in rows
                if "youtube.com" in r["url"].lower() or "youtu.be" in r["url"].lower()
            ]
            if yt:
                return yt[:limit]
        return rows[:limit]

    def _auto_open_from_search(
        self,
        search_result: dict[str, Any],
        *,
        on_tool: ToolCallback | None,
        seen_tool_keys: set[str],
    ) -> dict[str, Any]:
        results = search_result.get("results")
        if not isinstance(results, list) or not results:
            return search_result

        picks = self._pick_search_urls_to_open(
            str(search_result.get("query") or ""),
            results,
            limit=1,
        )
        if not picks:
            return search_result

        from ainet.tools.browser import open_chrome

        opened: list[dict[str, Any]] = []
        for pick in picks:
            url = pick["url"]
            tool_key = self._tool_call_key("open_chrome", {"url": url, "new_tab": True})
            if tool_key in seen_tool_keys:
                continue
            if on_tool:
                on_tool("start", "open_chrome", {"arguments": {"url": url, "new_tab": True}})
            try:
                chrome_result = open_chrome(url, new_tab=True)
                ok = bool(chrome_result.get("ok", True))
                err = None
            except Exception as exc:  # noqa: BLE001 — surface to model/UI
                chrome_result = {"ok": False, "opened": False, "url": url, "error": str(exc)}
                ok = False
                err = str(exc)
            seen_tool_keys.add(tool_key)
            seen_tool_keys.add(self._tool_call_key("open_chrome", {"url": url}))
            self.last_tool_names.append("open_chrome")
            if on_tool:
                on_tool(
                    "done",
                    "open_chrome",
                    {
                        "ok": ok,
                        "summary": err or self._tool_result_summary("open_chrome", chrome_result),
                    },
                )
            if ok:
                opened.append({"title": pick.get("title") or "", "url": url})

        if not opened:
            return search_result

        out = dict(search_result)
        out["auto_opened"] = opened
        out["hint"] = (
            "Host already opened the URL(s) in auto_opened in Chrome. "
            "Tell Hayden what opened. Do not claim other tabs opened unless you call open_chrome."
        )
        return out

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
        """Keep recent dialogue under a message/char budget without breaking tool sequences."""
        system = [m for m in self.messages if m.get("role") == "system"]
        rest = [self._compact_message(m) for m in self.messages if m.get("role") != "system"]
        rest = self._drop_orphan_tools(rest)

        msg_limit = max(4, self.config.max_history_messages)
        char_limit = max(2000, int(getattr(self.config, "max_history_chars", 12000) or 12000))

        last_user = 0
        for i, m in enumerate(rest):
            if m.get("role") == "user":
                last_user = i
        head, tail = rest[:last_user], rest[last_user:]

        while len(head) + len(tail) > msg_limit and head:
            nxt = self._drop_oldest_turn(head)
            if nxt == head:
                head = head[1:]
            else:
                head = nxt
        while head and self._dialogue_chars(head) + self._dialogue_chars(tail) > char_limit:
            nxt = self._drop_oldest_turn(head)
            if nxt == head:
                head = head[1:]
            else:
                head = nxt

        rest = self._drop_orphan_tools(head + tail)
        self.messages = system + rest

    @staticmethod
    def _dialogue_chars(messages: list[dict[str, Any]]) -> int:
        total = 0
        for m in messages:
            total += len(str(m.get("content") or ""))
            tools = m.get("tool_calls")
            if tools:
                try:
                    total += len(json.dumps(tools, ensure_ascii=False))
                except (TypeError, ValueError):
                    total += 200
        return total

    @staticmethod
    def _compact_message(message: dict[str, Any]) -> dict[str, Any]:
        """Shrink oversized message bodies so long chats stay model-friendly."""
        out = dict(message)
        content = out.get("content")
        if isinstance(content, str) and len(content) > 4000:
            out["content"] = content[:4000] + "…"
        # Thinking blobs are unused for follow-up turns and burn context.
        out.pop("thinking", None)
        return out

    @staticmethod
    def _has_tool_calls(message: dict[str, Any]) -> bool:
        tools = message.get("tool_calls")
        return isinstance(tools, list) and bool(tools)

    def _drop_orphan_tools(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove tool results that aren't preceded by an assistant tool_calls message."""
        kept: list[dict[str, Any]] = []
        pending_tools = 0
        for m in messages:
            role = m.get("role")
            if role == "assistant" and self._has_tool_calls(m):
                # Previous assistant tool_calls never received its results — drop it.
                if pending_tools > 0 and kept and kept[-1].get("role") == "assistant" and self._has_tool_calls(kept[-1]):
                    kept.pop()
                pending_tools = len(m.get("tool_calls") or [])
                kept.append(m)
                continue
            if role == "tool":
                if pending_tools > 0:
                    kept.append(m)
                    pending_tools -= 1
                # else: orphan tool result — skip
                continue
            # New user/assistant prose — reset pending tool expectations.
            pending_tools = 0
            kept.append(m)
        # Incomplete tail: assistant tool_calls with no results yet (mid-flight) — keep it.
        # Only drop if it has tool_calls AND we somehow have zero following tools while
        # pending_tools still equals the full call count (no results appended).
        if (
            pending_tools > 0
            and kept
            and kept[-1].get("role") == "assistant"
            and self._has_tool_calls(kept[-1])
            and pending_tools == len(kept[-1].get("tool_calls") or [])
        ):
            # No results followed this tool_calls message — drop the incomplete call.
            kept.pop()
        return kept

    def _drop_oldest_turn(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop the oldest user→… block (including any tool sequence under it)."""
        if len(messages) <= 2:
            return messages[-2:] if len(messages) == 2 else messages
        # Find second user message; keep from there. If none, drop first message safely.
        user_idxs = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if len(user_idxs) >= 2:
            return messages[user_idxs[1] :]
        # No clean user boundary — drop from the front until role changes past first item.
        return messages[1:]

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
            allowed = set(READ_TOOLS) | {"get_tools", "getTools"}
            if self.mode.tool_names:
                allowed |= set(self.mode.tool_names)
            if name not in allowed or name in _MUTATING:
                return {
                    "ok": False,
                    "error": (
                        f"OAC cannot use tool '{name}'. "
                        "Allowed: read/web. "
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

        if self.mode.role == "soi" and name in {"create_cop", "create_folder"}:
            path = str(args.get("path") or args.get("folder_path") or "")
            if "/Courses/" in path.replace("\\", "/") or "/Projects/" in path.replace("\\", "/"):
                src = self._soi_source_text()
                if src and not cop_name_in_text(path, src):
                    return {
                        "ok": False,
                        "error": (
                            f"{name} refused — {path} is not named in user_text. "
                            "Do not invent COPs."
                        ),
                    }

        result = dispatch(self.db, name, args)
        if self.mode.role == "soi":
            result = _redact_assistant_fields(result)
        return result

    def _soi_source_text(self) -> str:
        parts: list[str] = []
        for message in reversed(self.messages):
            if message.get("role") != "user":
                continue
            raw = str(message.get("content") or "")
            label = "changelog_entries:"
            label_idx = raw.find(label)
            if label_idx >= 0:
                chunk = raw[label_idx + len(label) :].strip()
                if chunk.startswith("inbox_unfiled:"):
                    chunk = chunk.split("inbox_unfiled:", 1)[0].strip()
                try:
                    entries = json.loads(chunk)
                except json.JSONDecodeError:
                    entries = None
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict) and entry.get("user_text"):
                            parts.append(str(entry["user_text"]))
                    break
            idx = raw.find("{")
            if idx < 0:
                parts.append(raw)
                break
            try:
                obj = json.loads(raw[idx:])
            except json.JSONDecodeError:
                parts.append(raw)
                break
            if isinstance(obj, dict):
                for entry in obj.get("changelog_entries") or []:
                    if isinstance(entry, dict) and entry.get("user_text"):
                        parts.append(str(entry["user_text"]))
            break
        return "\n".join(parts)


def _redact_assistant_fields(obj: Any) -> Any:
    """SOI never sees OAC assistant_text, even via tool reads (source-of-truth path)."""
    if isinstance(obj, dict):
        return {
            k: _redact_assistant_fields(v)
            for k, v in obj.items()
            if k not in {"assistant_text", "assistant"}
        }
    if isinstance(obj, list):
        return [_redact_assistant_fields(x) for x in obj]
    return obj
