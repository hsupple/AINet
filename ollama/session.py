"""Chat session: OAC (read-only live) + optional SOI persistence hooks."""

from __future__ import annotations

import json
import threading
import time
import urllib.parse
from collections.abc import Callable
from typing import Any

from ainet.tools.ops import DatabaseTools
from ainet.tools.project import (
    find_project,
    resolve_project_rename_dest,
    resolve_under_project,
    user_project_name_from_path,
)
from ainet.tools.registry import PROJECT_SESSION_TOOLS, catalog_tools, dispatch, tools_subset
from ollama.client import OllamaCancelled, OllamaClient, OllamaError, ThinkingCallback, TokenCallback
from ollama.content_filing import cop_name_in_text
from ollama.content_tools import normalize_soi_tool, parse_content_tool_calls
from ollama.config import OllamaConfig
from ollama.convo_memory import (
    VisibleTokenFilter,
    extract_http_urls,
    host_fallback_memory,
    last_turn_block,
    memory_system_suffix,
    split_reply,
)
from ollama.conversation_store import ConversationStore
from ollama.prompts.shared import CURRENT_DATE_TOKEN, today_context
from ollama.inference_gate import INFERENCE_GATE
from ollama.modes import get_mode
from ollama.modes.base import READ_TOOLS, Mode
from ollama.router import suggest_mode

# on_tool(phase, name, detail) — phase is "start" | "done"
ToolCallback = Callable[[str, str, dict[str, Any]], None]
WaitCallback = Callable[[dict[str, Any]], None]
ContextCallback = Callable[[dict[str, Any]], None]


_MUTATING = {
    "write_json",
    "write_text",
    "create_json",
    "patch_json",
    "set_json_path",
    "create_folder",
    "create_cop",
    "create_project",
    "move_path",
    "archive_to_history",
    "append_changelog",
    "capture_inbox",
    "file_by_id",
    "file_note",
}

_PATH_ARG_KEYS = (
    "path",
    "src",
    "dest",
    "folder_or_path",
    "read_path",
    "history_dir",
    "source_path",
)


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
        self.project_root: str | None = None
        self._project_prev_mode: str = "companion"
        self.convo_memory: str = ""
        self.last_user_text: str = ""
        self.last_assistant_text: str = ""
        self.last_links: list[tuple[str, str]] = []
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
            self.convo_memory = self.store.load_memory(self.session_id)
            recent = self.store.recent_turns(self.session_id, limit=1)
            if recent:
                last = recent[-1]
                self.last_user_text = str(last.get("user") or "")
                self.last_assistant_text = str(last.get("assistant") or "")
                self.last_links = [("", u) for u in extract_http_urls(self.last_assistant_text)]
            self._rebuild_system()
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
        prompt = self.mode.prompt
        if self.mode.role == "oac":
            prompt = prompt.replace(CURRENT_DATE_TOKEN, today_context())
        if self.project_root:
            prompt = (
                f"{prompt}\n\nFocused project: {self.project_root}\n"
                "All paths are relative to this folder unless already under it."
            )
        if self.mode.role == "oac":
            prompt = f"{prompt}{memory_system_suffix(self.convo_memory)}"
            if self.last_user_text or self.last_assistant_text or self.last_links:
                prompt = f"{prompt}{last_turn_block(self.last_user_text, self.last_assistant_text, self.last_links)}"
        system: list[dict[str, Any]] = [{"role": "system", "content": prompt}]
        dialogue = [m for m in self.messages if m.get("role") != "system"]
        self.messages = system + dialogue

    def context_snapshot(self, *, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Labeled breakdown of what the model sees (for the Dev panel)."""
        system = ""
        for m in self.messages:
            if m.get("role") == "system":
                system = str(m.get("content") or "")
                break

        sections: list[dict[str, str]] = []
        date_line = today_context() if self.mode.role == "oac" else ""
        if date_line:
            sections.append({"id": "date", "label": "Today's date", "text": date_line})
        sections.append(
            {
                "id": "mode_rules",
                "label": f"Mode rules ({self.mode.id})",
                "text": (self.mode.prompt or "").strip() or "(none)",
            }
        )
        if self.project_root:
            sections.append(
                {
                    "id": "project",
                    "label": "Focused project",
                    "text": (
                        f"{self.project_root}\n"
                        "All paths are relative to this folder unless already under it."
                    ),
                }
            )
        if self.mode.role == "oac":
            mem = (self.convo_memory or "").strip() or "(empty — first turn)"
            sections.append({"id": "memory", "label": "Rolling memory", "text": mem})
            if self.last_user_text or self.last_assistant_text or self.last_links:
                sections.append(
                    {
                        "id": "previous_turn",
                        "label": "Previous turn",
                        "text": last_turn_block(
                            self.last_user_text,
                            self.last_assistant_text,
                            self.last_links,
                        ).strip(),
                    }
                )

        dialogue: list[dict[str, Any]] = []
        for m in self.messages:
            role = str(m.get("role") or "")
            if role == "system":
                continue
            entry: dict[str, Any] = {
                "role": role,
                "content": str(m.get("content") or ""),
            }
            tcalls = m.get("tool_calls")
            if tcalls:
                names: list[str] = []
                for tc in tcalls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    n = str((fn or {}).get("name") or "")
                    if n:
                        names.append(n)
                if names:
                    entry["tool_calls"] = names
            dialogue.append(entry)

        tool_names: list[str] = []
        if tools:
            for t in tools:
                if not isinstance(t, dict):
                    continue
                fn = t.get("function") if isinstance(t.get("function"), dict) else {}
                n = str((fn or {}).get("name") or "")
                if n:
                    tool_names.append(n)

        return {
            "mode": self.mode.id,
            "session_id": self.session_id,
            "project_root": self.project_root,
            "sections": sections,
            "system_chars": len(system),
            "system": system,
            "dialogue": dialogue,
            "tools": tool_names,
        }

    def reset(self) -> None:
        self.messages = []
        self.full_tools_unlocked = False
        self.project_root = None
        self._project_prev_mode = "companion"
        self.convo_memory = ""
        self.last_user_text = ""
        self.last_assistant_text = ""
        self.last_links = []
        if self.store and self.mode.role == "oac":
            self.session_id = self.store.new_session(mode_id=self.mode.id)
        self._rebuild_system()
        self.touch()

    def set_mode(self, mode_id: str, *, lock: bool = False) -> Mode:
        if mode_id != "project" and self.project_root:
            # Leaving project mode without close_project clears focus.
            self.project_root = None
        self.mode = get_mode(mode_id)
        self.mode_locked = lock
        self.full_tools_unlocked = False
        self._rebuild_system()
        self.touch()
        return self.mode

    def _maybe_autoroute(self, user_text: str) -> str | None:
        if self.project_root:
            return None
        if not self.auto_mode or self.mode.role != "oac":
            return None
        decision = suggest_mode(user_text, self.mode.id)
        explicit = decision.confidence >= 0.95
        if self.mode_locked and not explicit:
            return None
        if (
            decision.mode_id != self.mode.id
            and decision.confidence >= self.config.auto_mode_min_confidence
            and decision.mode_id != "soi"
        ):
            old = self.mode.id
            self.set_mode(decision.mode_id, lock=False)
            return f"(auto → {decision.mode_id}; was {old}; {decision.reason})"
        return None

    def ask(
        self,
        user_text: str,
        *,
        stream: bool = False,
        on_token: TokenCallback | None = None,
        on_thinking: ThinkingCallback | None = None,
        on_tool: ToolCallback | None = None,
        on_wait: WaitCallback | None = None,
        on_context: ContextCallback | None = None,
    ) -> str:
        # A prior Stop/abort can leave cancel set. A new user turn must start clean
        # or the first message after reload is immediately "(stopped)".
        self.clear_cancel()
        # OAC + SOI share one Ollama — never overlap inference (UI lockups / hung streams).
        holder = "soi" if self.mode.role == "soi" else "chat"
        waited = 0.0
        ticket = 0
        while True:
            if self.cancelled():
                raise OllamaCancelled("Cancelled")
            ticket = INFERENCE_GATE.acquire(timeout=1.0, holder=holder)
            if ticket:
                break
            waited += 1.0
            if on_wait is not None:
                try:
                    on_wait(dict(INFERENCE_GATE.snapshot()))
                except Exception:
                    pass
            if waited >= 180:
                snap = INFERENCE_GATE.snapshot()
                who = snap.get("holder") or "another job"
                raise OllamaError(
                    f"Ollama is busy ({who} for {snap.get('held_s')}s). "
                    "Press Reset AI, or Stop SOI if filing is running."
                )
        try:
            return self._ask_locked(
                user_text,
                stream=stream,
                on_token=on_token,
                on_thinking=on_thinking,
                on_tool=on_tool,
                on_context=on_context,
            )
        finally:
            INFERENCE_GATE.release(ticket)

    def _ask_locked(
        self,
        user_text: str,
        *,
        stream: bool = False,
        on_token: TokenCallback | None = None,
        on_thinking: ThinkingCallback | None = None,
        on_tool: ToolCallback | None = None,
        on_context: ContextCallback | None = None,
    ) -> str:
        self.touch()
        route_note = self._maybe_autoroute(user_text)
        if route_note and on_token is not None:
            on_token(f"{route_note}\n")

        self.messages.append({"role": "user", "content": user_text})
        self._turn_user_text = user_text
        self._trim_history()

        if self.mode.role == "oac":
            opened = self._maybe_host_open_links(
                user_text, on_token=on_token, on_tool=on_tool
            )
            if opened is not None:
                return opened

        tools = self._active_tools()
        if on_context is not None:
            try:
                on_context(self.context_snapshot(tools=tools))
            except Exception:
                pass
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
        hide_mem = self.mode.role == "oac"
        vis_filter = VisibleTokenFilter(on_token) if hide_mem and on_token else None
        turn_links: list[tuple[str, str]] = []

        def _token(delta: str) -> None:
            nonlocal streamed_any
            if not delta:
                return
            streamed_any = True
            streamed_parts.append(delta)
            if vis_filter is not None:
                vis_filter.feed(delta)
            elif on_token:
                on_token(delta)

        extra_mutating = {
            "mark_read_stale",
            "mark_read_refreshed",
            "save_research",
        }
        extra_opts = None
        if self.mode.role == "soi":
            extra_opts = {"temperature": 0, "num_predict": 900}
        elif self.mode.id == "deep_research":
            extra_opts = {"temperature": 0.2, "num_predict": 2800}
        max_rounds = self.config.max_tool_rounds
        if self.mode.id == "deep_research":
            max_rounds = max(max_rounds, 16)
            req_timeout = max(req_timeout, 360.0)

        try:
            for _round in range(max_rounds):
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
                        options=extra_opts,
                        should_cancel=self.cancelled,
                    )
                else:
                    response = self.client.chat(
                        self.messages,
                        tools=tools,
                        think=think,
                        timeout_s=req_timeout,
                        options=extra_opts,
                    )
                message = response.get("message") or {}
                self.messages.append(message)

                content = (message.get("content") or "").strip()
                tool_calls = message.get("tool_calls") or []
                parsed_from_text = False
                if not tool_calls:
                    if self.mode.role == "soi":
                        tool_calls = parse_content_tool_calls(content)
                        parsed_from_text = bool(tool_calls)
                    if not tool_calls:
                        piece = split_reply(content)[0] if hide_mem else content
                        if piece:
                            if final_text and piece not in final_text:
                                final_text = final_text.rstrip() + "\n\n" + piece
                            elif not final_text:
                                final_text = piece
                        break

                # Preamble before tool calls (math, explanation) must survive the
                # later "I found a video…" message — don't drop it from the turn.
                if content and tool_calls and not parsed_from_text:
                    piece = split_reply(content)[0] if hide_mem else content
                    if piece and piece not in (final_text or ""):
                        final_text = (
                            final_text.rstrip() + "\n\n" + piece if final_text else piece
                        )

                self.last_tool_rounds += 1
                if streamed_any and on_token:
                    if vis_filter is not None:
                        vis_filter.flush()
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
                        and self.mode.id != "deep_research"
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
                        done_detail: dict[str, Any] = {
                            "ok": bool(result.get("ok", True)),
                            "summary": self._tool_result_summary(name, result),
                        }
                        if name == "save_research" and isinstance(result, dict) and result.get("ok"):
                            done_detail["research"] = {
                                "id": result.get("id"),
                                "path": result.get("path"),
                                "title": result.get("title"),
                            }
                        if name == "image_search" and isinstance(result, dict) and result.get("ok"):
                            imgs = result.get("results")
                            if isinstance(imgs, list):
                                slim: list[dict[str, str]] = []
                                for item in imgs[:8]:
                                    if not isinstance(item, dict):
                                        continue
                                    slim.append(
                                        {
                                            "title": str(item.get("title") or "")[:120],
                                            "url": str(item.get("url") or ""),
                                            "page_url": str(item.get("page_url") or ""),
                                            "image_url": str(item.get("image_url") or ""),
                                            "thumbnail": str(item.get("thumbnail") or ""),
                                            "source": str(item.get("source") or ""),
                                        }
                                    )
                                done_detail["images"] = slim
                        if name == "web_search" and isinstance(result, dict) and result.get("ok"):
                            opened = result.get("auto_opened")
                            if isinstance(opened, list) and opened:
                                cards: list[dict[str, str]] = []
                                for item in opened[:3]:
                                    if not isinstance(item, dict):
                                        continue
                                    cards.append(
                                        {
                                            "title": str(item.get("title") or "")[:120],
                                            "url": str(item.get("url") or ""),
                                            "snippet": str(item.get("snippet") or "")[:220],
                                            "thumbnail": str(item.get("thumbnail") or ""),
                                            "source": str(item.get("source") or ""),
                                        }
                                    )
                                if cards:
                                    done_detail["articles"] = cards
                        on_tool("done", name, done_detail)
                    payload = self._truncate_tool_result(name, result)
                    self.messages.append(
                        {
                            "role": "tool",
                            "content": json.dumps(payload, ensure_ascii=False),
                        }
                    )
                    if hide_mem and isinstance(result, dict):
                        turn_links.extend(self._links_from_tool(name, result))
                if parsed_from_text:
                    break
                self._trim_history()
            else:
                final_text = "I hit the tool-call limit for this turn. Try again with a narrower ask."
        except OllamaCancelled:
            if vis_filter is not None:
                vis_filter.flush()
            partial = "".join(streamed_parts).strip()
            if hide_mem:
                partial, mem = split_reply(partial)
                if mem:
                    self.convo_memory = mem
                    self._rebuild_system()
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

        if vis_filter is not None:
            vis_filter.flush()
        if hide_mem:
            visible, mem = split_reply(final_text or "")
            final_text = visible
            if mem:
                self.convo_memory = mem
            else:
                self.convo_memory = host_fallback_memory(
                    user_text, final_text, self.convo_memory
                )
            self._strip_memory_from_stored_replies()
            self.last_user_text = user_text
            self.last_assistant_text = final_text
            self.last_links = turn_links[:8]
            self._rebuild_system()

        if self.store and self.mode.role == "oac":
            if not self.session_id or not self.store.session_exists(self.session_id):
                self.session_id = self.store.ensure_session(mode_id=self.mode.id)
            self.store.append_turn(
                self.session_id,
                user_text=user_text,
                assistant_text=final_text,
                mode_id=self.mode.id,
                memory=self.convo_memory,
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
        if name == "image_search":
            n = result.get("count")
            if n is None and isinstance(result.get("results"), list):
                n = len(result["results"])
            base = f"{n} photos" if n is not None else "ok"
            opened = result.get("auto_opened")
            if isinstance(opened, list) and opened:
                base += " · Google Images"
            return base
        if name == "web_fetch":
            text = result.get("text") or ""
            return f"{len(text)} chars"
        if name == "open_chrome":
            urls = result.get("urls")
            if isinstance(urls, list) and urls:
                n = len(urls)
                return f"{n} tabs" if n != 1 else str(urls[0])[:80]
            url = str(result.get("url") or "")
            if url:
                return url if len(url) <= 80 else url[:77] + "…"
            return "ok"
        if name == "save_research":
            path = str(result.get("path") or "")
            return path or "saved"
        if name == "inspect_research":
            if result.get("action") == "list":
                return f"{result.get('count', 0)} briefs"
            brief = result.get("brief") if isinstance(result.get("brief"), dict) else {}
            return str(brief.get("title") or brief.get("path") or "ok")
        if name in {"read_json", "read_text"}:
            path = result.get("path") or ""
            return path or "ok"
        if name in {"list_dir", "tree"}:
            kids = result.get("children") or result.get("entries")
            if isinstance(kids, list):
                return f"{len(kids)} entries"
        return "ok"

    def _wants_open_prior_links(self, user_text: str) -> bool:
        t = (user_text or "").lower()
        if self._user_opts_out_of_open(user_text):
            return False
        if "open" not in t and "pull up" not in t:
            return False
        return any(
            w in t
            for w in ("link", "tab", "page", "url", "those", "them", "these")
        )

    def _prior_urls(self) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        seen: set[str] = set()
        for title, url in self.last_links:
            href = (url or "").strip()
            if href.startswith("http") and href not in seen:
                seen.add(href)
                rows.append((title, href))
        if not rows and self.last_assistant_text:
            for href in extract_http_urls(self.last_assistant_text):
                rows.append(("", href))
        return rows[:8]

    def _maybe_host_open_links(
        self,
        user_text: str,
        *,
        on_token: TokenCallback | None,
        on_tool: ToolCallback | None,
    ) -> str | None:
        """Open last-turn URLs immediately — do not wait on the model (it hangs)."""
        if not self._wants_open_prior_links(user_text):
            return None
        rows = self._prior_urls()
        if not rows:
            return None
        from ainet.tools.browser import open_chrome

        urls = [url for _title, url in rows]
        if on_tool:
            on_tool("start", "open_chrome", {"arguments": {"urls": urls}})
        try:
            result = open_chrome(urls=urls, new_tab=True)
            ok = bool(result.get("ok", True))
            err = result.get("error")
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "error": str(exc)}
            ok = False
            err = str(exc)
        n = len(result.get("opened") or urls) if ok else 0
        summary = err or (f"{n} tabs" if n != 1 else urls[0])
        if on_tool:
            on_tool("done", "open_chrome", {"ok": ok, "summary": summary})
        if ok:
            reply = "Opened in Chrome." if n == 1 else f"Opened {n} tabs in Chrome."
        else:
            reply = f"Couldn't open Chrome: {err}"
        if on_token:
            on_token(reply)
        self.messages.append({"role": "assistant", "content": reply})
        self.last_tool_names = ["open_chrome"]
        self.last_user_text = user_text
        self.last_assistant_text = reply
        self.convo_memory = host_fallback_memory(user_text, reply, self.convo_memory)
        self._rebuild_system()
        if self.store and self.mode.role == "oac":
            if not self.session_id or not self.store.session_exists(self.session_id):
                self.session_id = self.store.ensure_session(mode_id=self.mode.id)
            self.store.append_turn(
                self.session_id,
                user_text=user_text,
                assistant_text=reply,
                mode_id=self.mode.id,
                memory=self.convo_memory,
            )
            current = self.store.current_session_id()
            if current:
                self.session_id = current
        self.touch()
        return reply

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
    def _pick_search_urls_to_open(query: str, results: list[Any], *, limit: int = 3) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url.startswith(("http://", "https://")):
                continue
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            thumb = str(item.get("thumbnail") or "").strip()
            source = str(item.get("source") or "").strip()
            rows.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "thumbnail": thumb,
                    "source": source,
                }
            )
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

    @staticmethod
    def _enrich_article_cards(picks: list[dict[str, str]]) -> list[dict[str, str]]:
        """Attach preview images to auto-opened search hits."""
        from ainet.tools.web import article_preview_images

        missing = [
            str(p.get("url") or "")
            for p in picks
            if not str(p.get("thumbnail") or "").startswith(("http://", "https://"))
        ]
        extras = article_preview_images(missing) if missing else {}
        cards: list[dict[str, str]] = []
        for pick in picks:
            url = str(pick.get("url") or "")
            thumb = str(pick.get("thumbnail") or extras.get(url) or "")
            if not thumb.startswith(("http://", "https://")):
                thumb = ""
            source = str(pick.get("source") or "")
            if not source and url:
                host = urllib.parse.urlparse(url).netloc.lower()
                source = host[4:] if host.startswith("www.") else host
            cards.append(
                {
                    "title": str(pick.get("title") or "")[:120],
                    "url": url,
                    "snippet": str(pick.get("snippet") or "")[:220],
                    "thumbnail": thumb,
                    "source": source,
                }
            )
        return cards

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
            limit=3,
        )
        if not picks:
            return search_result

        from ainet.tools.browser import open_chrome

        # Skip URLs already opened this turn
        fresh: list[dict[str, str]] = []
        for pick in picks:
            url = pick["url"]
            key_a = self._tool_call_key("open_chrome", {"url": url, "new_tab": True})
            key_b = self._tool_call_key("open_chrome", {"url": url})
            if key_a in seen_tool_keys or key_b in seen_tool_keys:
                continue
            fresh.append(pick)
        if not fresh:
            return search_result

        urls = [p["url"] for p in fresh]
        if on_tool:
            on_tool(
                "start",
                "open_chrome",
                {"arguments": {"urls": urls, "new_tab": True}},
            )
        try:
            chrome_result = open_chrome(urls=urls, new_tab=True)
            ok = bool(chrome_result.get("ok", True))
            err = None
        except Exception as exc:  # noqa: BLE001
            chrome_result = {"ok": False, "opened": False, "urls": urls, "error": str(exc)}
            ok = False
            err = str(exc)

        for url in urls:
            seen_tool_keys.add(self._tool_call_key("open_chrome", {"url": url, "new_tab": True}))
            seen_tool_keys.add(self._tool_call_key("open_chrome", {"url": url}))
        self.last_tool_names.append("open_chrome")
        cards = self._enrich_article_cards(fresh)
        if on_tool:
            summary = err
            if not summary:
                n = int(chrome_result.get("count") or len(urls))
                summary = f"{n} tabs" if n != 1 else urls[0]
            on_tool(
                "done",
                "open_chrome",
                {"ok": ok, "summary": summary},
            )

        if not ok:
            return search_result

        out = dict(search_result)
        out["auto_opened"] = cards
        out["hint"] = (
            f"Host already opened {len(cards)} URL(s) in Chrome (see auto_opened). "
            "Do not claim other tabs opened unless you call open_chrome."
        )
        return out

    def _trim_history(self) -> None:
        """Keep recent dialogue under a message/char budget without breaking tool sequences."""
        system = [m for m in self.messages if m.get("role") == "system"]
        rest = [self._compact_message(m) for m in self.messages if m.get("role") != "system"]
        rest = self._drop_orphan_tools(rest)

        # OAC: memory + previous turn (prose) + current turn. Drop old tool dumps.
        if self.mode.role == "oac":
            rest = self._keep_oac_recent(rest)
            self.messages = system + rest
            return

        msg_limit = max(4, self.config.max_history_messages)
        char_limit = max(2000, int(getattr(self.config, "max_history_chars", 12000) or 12000))
        if self.mode.id == "deep_research":
            msg_limit = max(msg_limit, 28)
            char_limit = max(char_limit, 36000)

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

    def _keep_oac_recent(self, rest: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Previous user+reply (text only) + current user turn (including in-flight tools)."""
        user_idxs = [i for i, m in enumerate(rest) if m.get("role") == "user"]
        if not user_idxs:
            return rest
        current = user_idxs[-1]
        current_turn = rest[current:]
        if len(user_idxs) < 2:
            return self._drop_orphan_tools(current_turn)
        prev = user_idxs[-2]
        prior = self._prose_only_turn(rest[prev:current])
        return self._drop_orphan_tools(prior + current_turn)

    @staticmethod
    def _prose_only_turn(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep last user + last answer in full; drop tool_calls. Preserve URLs from tools."""
        out: list[dict[str, Any]] = []
        last_assistant: dict[str, Any] | None = None
        link_lines: list[str] = []
        seen: set[str] = set()
        for message in messages:
            role = message.get("role")
            if role == "user":
                text = str(message.get("content") or "").strip()
                out.append({"role": "user", "content": text[:2000]})
            elif role == "assistant":
                text = str(message.get("content") or "").strip()
                if text:
                    last_assistant = {"role": "assistant", "content": text[:4000]}
            elif role == "tool":
                raw = message.get("content")
                try:
                    data = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    data = None
                if isinstance(data, dict):
                    for title, url in ChatSession._links_from_tool("", data):
                        if url in seen:
                            continue
                        seen.add(url)
                        label = title or url
                        link_lines.append(f"- {label}: {url}")
        if last_assistant:
            body = last_assistant["content"]
            if link_lines and "http" not in body:
                body = body.rstrip() + "\n\nLinks:\n" + "\n".join(link_lines[:8])
            last_assistant["content"] = body[:4000]
            out.append(last_assistant)
        return out

    @staticmethod
    def _links_from_tool(name: str, result: dict[str, Any]) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        results = result.get("results")
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                if url.startswith("http"):
                    rows.append((str(item.get("title") or ""), url))
        opened = result.get("auto_opened") or result.get("opened")
        if isinstance(opened, list):
            for item in opened:
                if isinstance(item, dict):
                    url = str(item.get("url") or "").strip()
                    if url.startswith("http"):
                        rows.append((str(item.get("title") or ""), url))
                elif isinstance(item, str) and item.startswith("http"):
                    rows.append(("", item))
        urls = result.get("urls")
        if isinstance(urls, list):
            for item in urls:
                if isinstance(item, str) and item.startswith("http"):
                    rows.append(("", item))
        url = str(result.get("url") or "").strip()
        if url.startswith("http"):
            rows.append((str(result.get("title") or ""), url))
        _ = name
        return rows

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

    def _strip_memory_from_stored_replies(self) -> None:
        for message in self.messages:
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content:
                continue
            visible, _mem = split_reply(content)
            message["content"] = visible

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

    def _truncate_tool_result(self, name: str, result: dict[str, Any]) -> dict[str, Any]:
        raw = json.dumps(result, ensure_ascii=False)
        limit = self.config.max_tool_result_chars
        if self.mode.id == "deep_research":
            if name == "web_fetch":
                limit = max(limit, 9000)
            elif name == "web_search":
                limit = max(limit, 4000)
            elif name == "image_search":
                limit = max(limit, 5000)
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

        name, args = self._prefer_create_project(name, args)

        if name in {"get_tools", "getTools"}:
            return catalog_tools(
                detail=bool(args.get("detail", True)),
                read_only=not self.mode.allow_mutations,
            )

        if name in {"open_project", "close_project"}:
            return self._handle_project_session_tool(name, args)

        if not self.mode.allow_mutations:
            allowed = set(READ_TOOLS) | {"get_tools", "getTools"} | set(PROJECT_SESSION_TOOLS)
            if self.mode.tool_names:
                allowed |= set(self.mode.tool_names)
            # create_project is mutating but explicitly allowed as a session tool.
            if name in PROJECT_SESSION_TOOLS:
                pass
            elif name not in allowed or name in _MUTATING:
                return {
                    "ok": False,
                    "error": (
                        f"OAC cannot use tool '{name}'. "
                        "Allowed: read/web + create/open/close project. "
                        "SOI files lasting DB writes from the changelog after idle."
                    ),
                }
        elif (
            not self.full_tools_unlocked
            and self.mode.tool_names is not None
            and name not in self.mode.tool_names
            and name not in {"get_tools", "getTools"}
            and name not in PROJECT_SESSION_TOOLS
        ):
            return {
                "ok": False,
                "error": (
                    f"Tool '{name}' not in the lean set for mode '{self.mode.id}'. "
                    "Call get_tools to unlock the full catalog."
                ),
            }

        if self.project_root and name == "create_project":
            return {
                "ok": False,
                "error": (
                    f"Already focused on {self.project_root}. "
                    "Call close_project before creating another project."
                ),
            }

        try:
            args = self._scope_tool_args(name, args)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        if name == "image_search" and self._user_opts_out_of_open(
            getattr(self, "_turn_user_text", "") or ""
        ):
            args = dict(args)
            args["open_google"] = False

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
        if name == "move_path" and self.project_root and result.get("ok"):
            self._maybe_update_project_root_after_move(result)
        if self.mode.role == "soi":
            result = _redact_assistant_fields(result)
        return result

    def _prefer_create_project(
        self, name: str, args: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Start a user project via create_project, never create_folder/create_cop."""
        if self.project_root or name == "create_project":
            return name, args
        if name not in {"create_folder", "create_cop"}:
            return name, args
        path = str(args.get("path") or args.get("folder_path") or args.get("name") or "")
        extracted = user_project_name_from_path(path)
        kind = str(args.get("kind") or args.get("cop_type") or "").strip().casefold()
        if (
            not extracted
            and name == "create_cop"
            and kind == "project"
            and not path.replace("\\", "/").startswith("Work/")
        ):
            extracted = user_project_name_from_path(path) or (path.strip("/\\") or "")
            if "/" in extracted.replace("\\", "/"):
                extracted = ""
        if not extracted:
            return name, args
        return "create_project", {
            "name": extracted,
            "summary": str(args.get("summary") or ""),
        }

    def _handle_project_session_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "close_project":
            closed = self.project_root
            if not closed:
                return {"ok": True, "closed": None, "hint": "No project was focused."}
            prev = self._project_prev_mode or "companion"
            self.project_root = None
            self.set_mode(prev, lock=False)
            return {
                "ok": True,
                "closed": closed,
                "mode": self.mode.id,
                "hint": f"Left {closed}. Full db/ access restored ({self.mode.id}).",
            }

        # open_project
        raw = str(args.get("name") or args.get("path") or "").strip()
        if not raw:
            return {"ok": False, "error": "open_project requires name (or path)"}
        found = find_project(self.db, raw)
        if not found:
            return {
                "ok": False,
                "error": (
                    f"Project not found: {raw!r}. "
                    "Use list_projects or create_project first."
                ),
            }
        if self.mode.id != "project":
            self._project_prev_mode = self.mode.id
        self.project_root = found
        self.set_mode("project", lock=True)
        # set_mode clears project_root when leaving project — re-apply after switch.
        self.project_root = found
        self._rebuild_system()
        return {
            "ok": True,
            "path": found,
            "name": found.rsplit("/", 1)[-1],
            "mode": "project",
            "hint": (
                f"Focused on {found}. Only this folder is accessible until close_project. "
                "Use '.' for list_dir/tree; bare filenames resolve inside the project."
            ),
        }

    def _scope_tool_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if not self.project_root:
            return args
        if name in PROJECT_SESSION_TOOLS | {
            "web_search",
            "web_fetch",
            "image_search",
            "open_chrome",
            "get_tools",
            "getTools",
            "save_research",
            "inspect_research",
        }:
            return args

        out = dict(args)
        root = self.project_root

        # Default list/tree to project root
        if name in {"list_dir", "tree"}:
            raw_path = out.get("path", ".")
            if raw_path is None or str(raw_path).strip() in {"", ".", "./"}:
                out["path"] = root
                return out

        # Renaming the project folder itself may target a Projects/<NewName> sibling.
        if name == "move_path":
            src_raw = str(out.get("src") or "").strip()
            dest_raw = str(out.get("dest") or "").strip()
            if src_raw:
                out["src"] = resolve_under_project(root, src_raw, self.db)
            if dest_raw:
                if out.get("src") == root:
                    sibling = resolve_project_rename_dest(root, dest_raw)
                    if sibling is None:
                        raise ValueError(
                            "To rename the project folder, dest must be a new name "
                            f"or Projects/<NewName> (still under Projects/). Got {dest_raw!r}."
                        )
                    out["dest"] = sibling
                else:
                    out["dest"] = resolve_under_project(root, dest_raw, self.db)
            return out

        for key in _PATH_ARG_KEYS:
            if key not in out or out[key] is None:
                continue
            raw = str(out[key]).strip()
            if not raw:
                continue
            if key == "path" and name in {"list_dir", "tree"} and raw in {".", "./"}:
                out[key] = root
                continue
            out[key] = resolve_under_project(root, raw, self.db)
        return out

    def _maybe_update_project_root_after_move(self, result: dict[str, Any]) -> None:
        """If the focused project folder itself was renamed, keep focus on the new path."""
        if not self.project_root:
            return
        src = str(result.get("from") or "")
        dest = str(result.get("to") or "")
        if not src or not dest:
            return
        root = self.project_root
        if src == root or root.startswith(src + "/"):
            # Renamed project root or an ancestor path segment
            if src == root:
                self.project_root = dest
            else:
                self.project_root = dest + root[len(src) :]
            self._rebuild_system()

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
