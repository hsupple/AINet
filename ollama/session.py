"""Chat session: OAC (read-only live) + optional SOI persistence hooks."""

from __future__ import annotations

import json
import re
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
from ainet.tools.registry import (
    CALENDAR_SESSION_TOOLS,
    PROJECT_SESSION_TOOLS,
    catalog_tools,
    dispatch,
    tools_subset,
)
from ollama.client import OllamaCancelled, OllamaClient, OllamaError, ThinkingCallback, TokenCallback
from ollama.db_query_hints import (
    compact_digest,
    extract_query_tokens,
    hayden_asking_about_self,
    is_hayden_db_dest,
    looks_like_bad_self_reply,
    looks_like_empty_therapy,
    looks_like_fresh_start,
    looks_like_generic_ignore,
    looks_like_profile_dump,
    personal_memory_question,
    query_result_includes_hayden,
    self_identity_context,
    should_prefetch,
    should_prefetch_calendar,
    strip_profile_dump,
    wants_open_web,
    wrong_dest_for_self_query,
)
from ollama.content_tools import normalize_soi_tool, parse_content_tool_calls
from ollama.config import OllamaConfig
from ollama.convo_memory import (
    VisibleTokenFilter,
    enrich_web_search_query,
    extract_http_urls,
    host_fallback_memory,
    is_action_confirm,
    is_short_ack,
    is_vague_pronoun_followup,
    is_vague_search_followup,
    known_context_block,
    last_user_wanted_videos,
    memory_system_suffix,
    named_search_topic,
    offered_to_calendar,
    offered_to_search,
    recent_turns_block,
    reply_looks_like_clarifier,
    split_reply,
    standing_request,
    strip_leading_clarifier,
    followup_topic,
    topic_for_search,
    wants_videos,
)

# Completed prose turns kept for OAC follow-ups (plus the in-flight current turn).
_OAC_PRIOR_TURNS = 5
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
ControlCallback = Callable[[str], None]

_YEAR_TOKEN = re.compile(r"\b(?:19|20)\d{2}\b")


def _clip_about(text: str, limit: int = 72) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip().strip("\"'")
    if len(s) <= limit:
        return s
    cut = s[: limit - 1].rsplit(" ", 1)[0] or s[: limit - 1]
    return cut.rstrip(" ·,-") + "…"


def _digest_lookup(text: str, *, tail: bool = False) -> str:
    s = _YEAR_TOKEN.sub(" ", str(text or ""))
    s = re.sub(r"\s+", " ", s).strip().strip("\"'")
    words = [w for w in s.split() if w]
    if len(words) > 8:
        words = words[-5:] if tail else words[:8]
    return _clip_about(" ".join(words))


def _host_about(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        host = (urllib.parse.urlparse(raw).netloc or "").removeprefix("www.")
    except Exception:
        host = ""
    return _clip_about(host or raw)


def default_tool_about(name: str, args: dict[str, Any] | None) -> str:
    """Short card phrase: model `about` if present, else a cleaned lookup string."""
    args = args or {}
    existing = _clip_about(str(args.get("about") or ""))
    if existing:
        return existing
    if name in {"web_search", "image_search"}:
        return _digest_lookup(str(args.get("query") or ""))
    if name == "query_db":
        dest = str(args.get("dest") or args.get("file") or "").casefold()
        q = str(args.get("q") or args.get("name") or "").strip()
        if dest.startswith("hayden") and not q:
            return "what we have on you"
        return _digest_lookup(q or dest, tail=True) or "stored notes"
    if name == "web_fetch":
        return _host_about(str(args.get("url") or "")) or "reading page"
    if name == "open_chrome":
        urls = args.get("urls")
        url = str(args.get("url") or "")
        if isinstance(urls, list) and urls:
            first = _host_about(str(urls[0] or ""))
            extra = len(urls) - 1
            if extra > 0 and first:
                return _clip_about(f"{first} +{extra}")
            return first or f"{len(urls)} pages"
        return _host_about(url) or "open tab"
    if name == "spotify":
        return _clip_about(str(args.get("query") or args.get("action") or ""))
    if name == "create_plot":
        return _clip_about(str(args.get("title") or args.get("equation") or ""))
    if name == "query_calendar":
        existing = _clip_about(str(args.get("about") or ""))
        if existing and not re.match(r"^-?\d{2}-\d{2}$", existing) and not re.match(
            r"^\d{4}-\d{2}-\d{2}", existing
        ):
            return existing
        q = str(args.get("q") or "").strip()
        if q:
            return _digest_lookup(q) or "calendar"
        start = str(args.get("start") or "")[:10]
        end = str(args.get("end") or "")[:10]
        try:
            from datetime import date as _date

            s = _date.fromisoformat(start) if len(start) == 10 else None
            e = _date.fromisoformat(end) if len(end) == 10 else None
        except ValueError:
            s = e = None
        if s and (not e or e == s):
            return f"{s.strftime('%a %b')} {s.day}"
        if s and e:
            return f"{s.strftime('%b')} {s.day}–{e.day}"
        return "calendar"
    if name == "add_calendar_event":
        return _clip_about(str(args.get("title") or "new event")) or "new event"
    if name == "update_calendar_event":
        return _clip_about(str(args.get("title") or args.get("id") or "update event"))
    if name == "cancel_calendar_event":
        return _clip_about(str(args.get("id") or "cancel event"))
    return ""


_MUTATING = {
    "write_json",
    "write_text",
    "create_json",
    "patch_json",
    "set_json_path",
    "create_folder",
    "create_project",
    "move_path",
    "append_changelog",
    "log_item",
    "file_note",
}

# Qwen3 sometimes answers with only the hidden %%mem%% block; resample instead.
_MAX_EMPTY_ROUNDS = 2

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
        # Phase 2 SOI: restrict patch_json / refresh_read to this folder only.
        self.soi_folder_scope: str | None = None
        self._project_prev_mode: str = "companion"
        self.convo_memory: str = ""
        self._injected_context: str = ""
        self.last_host_retry: str | None = None
        self._replace_stream = None
        self.last_user_text: str = ""
        self.last_assistant_text: str = ""
        self.last_links: list[tuple[str, str]] = []
        # Last 2–3 completed (user, assistant) pairs for pronoun / follow-up context.
        self.recent_turns: list[tuple[str, str]] = []
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
            if resume_session:
                self.session_id = self.store.ensure_session(mode_id=mode.id)
            else:
                self.session_id = self.store.new_session(mode_id=mode.id)
        self._rebuild_system()
        if resume_session and self.store and self.session_id:
            self.convo_memory = self.store.load_memory(self.session_id)
            self._load_recent_turns_from_store(limit=_OAC_PRIOR_TURNS + 1)
            self._rebuild_system()
        self.cancel_event = threading.Event()

    def _load_recent_turns_from_store(self, *, limit: int = 3) -> None:
        if not self.store or not self.session_id:
            self.recent_turns = []
            self.last_user_text = ""
            self.last_assistant_text = ""
            self.last_links = []
            return
        recent = self.store.recent_turns(self.session_id, limit=max(1, limit))
        self.recent_turns = [
            (str(t.get("user") or ""), str(t.get("assistant") or ""))
            for t in recent
            if isinstance(t, dict)
        ]
        if self.recent_turns:
            self.last_user_text, self.last_assistant_text = self.recent_turns[-1]
            self.last_links = [("", u) for u in extract_http_urls(self.last_assistant_text)]
        else:
            self.last_user_text = ""
            self.last_assistant_text = ""
            self.last_links = []

    def _record_completed_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        links: list[tuple[str, str]] | None = None,
    ) -> None:
        self.last_user_text = user_text
        self.last_assistant_text = assistant_text
        if links is not None:
            self.last_links = links
        pair = (user_text, assistant_text)
        if self.recent_turns and self.recent_turns[-1] == pair:
            return
        self.recent_turns.append(pair)
        self.recent_turns = self.recent_turns[-(_OAC_PRIOR_TURNS + 1) :]

    def open_stored_session(self, session_id: str) -> dict[str, Any]:
        """Switch to a logged chat. UI gets full turns; the model still gets recent-turn memory."""
        if not self.store:
            return {"ok": False, "error": "Chat log is not enabled"}
        if not self.store.session_exists(session_id):
            return {"ok": False, "error": "Chat not found"}
        payload = self.store.session_payload(session_id)
        self.session_id = str(payload.get("id") or session_id)
        self.store.set_current(self.session_id)
        self.convo_memory = str(payload.get("memory") or "")
        turns = payload.get("turns") if isinstance(payload.get("turns"), list) else []
        self.recent_turns = [
            (str(t.get("user") or ""), str(t.get("assistant") or ""))
            for t in turns[-(_OAC_PRIOR_TURNS + 1) :]
            if isinstance(t, dict)
        ]
        if self.recent_turns:
            self.last_user_text, self.last_assistant_text = self.recent_turns[-1]
            self.last_links = [("", u) for u in extract_http_urls(self.last_assistant_text)]
        else:
            self.last_user_text = ""
            self.last_assistant_text = ""
            self.last_links = []
        self.messages = []
        self.full_tools_unlocked = False
        self.project_root = None
        mode_id = str(payload.get("mode_id") or "").strip()
        if mode_id:
            try:
                mode = get_mode(mode_id)
            except KeyError:
                mode = None
            if mode is not None and mode.role == "oac":
                self.mode = mode
                self.mode_locked = False
        self._rebuild_system()
        self.touch()
        return {"ok": True, "chat": payload}

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
                names = READ_TOOLS + ("get_tools",) + tuple(CALENDAR_SESSION_TOOLS)
            else:
                names = self.mode.tool_names or READ_TOOLS + ("get_tools",)
            # Calendar writes stay available even when OAC is otherwise read-only.
            merged = list(names)
            for n in CALENDAR_SESSION_TOOLS:
                if n not in merged:
                    merged.append(n)
            tools = tools_subset(merged)
            ask = getattr(self, "_turn_user_text", "") or ""
            if (
                self.mode.role == "oac"
                and hayden_asking_about_self(ask)
                and not wants_open_web(ask)
            ):
                blocked = {"web_search", "web_fetch"}
                tools = [
                    t
                    for t in tools
                    if (t.get("function") or {}).get("name") not in blocked
                ]
            return tools
        if self.full_tools_unlocked or self.mode.tool_names is None:
            return tools_subset(None)
        merged = list(self.mode.tool_names)
        for n in CALENDAR_SESSION_TOOLS:
            if n not in merged:
                merged.append(n)
        return tools_subset(merged)

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
            injected = (self._injected_context or "").strip()
            if injected:
                prompt = f"{prompt}{known_context_block(injected)}"
            turns = list(self.recent_turns)
            if not turns and (self.last_user_text or self.last_assistant_text):
                turns = [(self.last_user_text, self.last_assistant_text)]
            if turns:
                prompt = f"{prompt}{recent_turns_block(turns, self.last_links)}"
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
                "label": "OAC rules" if self.mode.role == "oac" else f"Mode rules ({self.mode.id})",
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
            injected = (self._injected_context or "").strip()
            if injected:
                sections.append(
                    {
                        "id": "known_context",
                        "label": "Known context (auto)",
                        "text": injected,
                    }
                )
            if self.recent_turns or self.last_user_text or self.last_assistant_text or self.last_links:
                turns = list(self.recent_turns)
                if not turns and (self.last_user_text or self.last_assistant_text):
                    turns = [(self.last_user_text, self.last_assistant_text)]
                sections.append(
                    {
                        "id": "recent_turns",
                        "label": "Recent turns",
                        "text": recent_turns_block(turns, self.last_links).strip(),
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
        self._injected_context = ""
        self.last_host_retry = None
        self.last_user_text = ""
        self.last_assistant_text = ""
        self.last_links = []
        self.recent_turns = []
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
        """Silent capability switch (deep research / explicit). Never announce flavor changes."""
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
            self.set_mode(decision.mode_id, lock=False)
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
        on_control: ControlCallback | None = None,
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
                on_control=on_control,
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
        on_control: ControlCallback | None = None,
    ) -> str:
        self.touch()
        self._maybe_autoroute(user_text)
        self._turn_user_text = user_text
        self.last_tool_names: list[str] = []
        self.last_mutating_calls: list[dict[str, Any]] = []
        self.last_tool_rounds = 0
        self._turn_hayden_db_ok = False
        self._force_prose_after_hayden_db = False
        self._hayden_db_digest = ""
        self._injected_context = ""
        self.last_host_retry = None
        self._replace_stream = None
        self._turn_calendar_rows: list[dict[str, Any]] = []
        self._turn_calendar_about = ""

        if self.mode.role == "oac":
            self._prefetch_turn_context(user_text, on_tool=on_tool)
            self._rebuild_system()

        ask_content = user_text
        if self.mode.role == "oac":
            bind = self._pronoun_bind_note(user_text)
            if bind:
                ask_content = f"{user_text}\n\n{bind}"
        self.messages.append({"role": "user", "content": ask_content})
        self._trim_history()

        if self.mode.role == "oac":
            cal_add = self._maybe_host_calendar_add(
                user_text, on_token=on_token, on_tool=on_tool
            )
            if cal_add is not None:
                return cal_add
            followed = self._maybe_host_followup_action(
                user_text, on_token=on_token, on_tool=on_tool
            )
            if followed is not None:
                return followed
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
        model_mem: str | None = None
        empty_rounds = 0
        streamed_any = False
        streamed_parts: list[str] = []
        seen_tool_keys: set[str] = set()
        think = self.config.soi_think if self.mode.role == "soi" else self.config.oac_think
        req_timeout = (
            self.config.soi_timeout_s if self.mode.role == "soi" else self.config.timeout_s
        )
        hide_mem = self.mode.role == "oac"
        vis_filter = VisibleTokenFilter(on_token) if hide_mem and on_token else None
        turn_links: list[tuple[str, str]] = []
        hold_ui = hide_mem and self.mode.role == "oac" and on_token is not None
        held_tokens: list[str] = []
        released_hold = not hold_ui

        def _forward(delta: str) -> None:
            if vis_filter is not None:
                vis_filter.feed(delta)
            elif on_token:
                on_token(delta)

        def _release_hold() -> None:
            nonlocal released_hold
            if released_hold:
                return
            released_hold = True
            blob = "".join(held_tokens)
            held_tokens.clear()
            if blob:
                _forward(blob)

        def _discard_and_replace() -> None:
            nonlocal released_hold
            held_tokens.clear()
            released_hold = True
            if vis_filter is not None:
                vis_filter.reset()
            if on_control:
                on_control("replace_reply")

        self._replace_stream = _discard_and_replace

        def _token(delta: str) -> None:
            nonlocal streamed_any
            if not delta:
                return
            streamed_any = True
            streamed_parts.append(delta)
            if not released_hold:
                held_tokens.append(delta)
                preview = "".join(held_tokens)
                if reply_looks_like_clarifier(preview) and self._will_retry_clarifier(
                    user_text
                ):
                    return
                if len(preview) >= 80:
                    _release_hold()
                return
            _forward(delta)

        extra_mutating = {
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
        elif self.mode.id == "soi_test_p2":
            max_rounds = max(max_rounds, 10)

        try:
            for _round in range(max_rounds):
                if self.cancelled():
                    raise OllamaCancelled("Cancelled")
                self._trim_history()
                round_tools = tools
                if getattr(self, "_force_prose_after_hayden_db", False):
                    round_tools = None
                if stream or on_token or on_thinking:
                    response = self.client.chat_stream(
                        self.messages,
                        tools=round_tools,
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
                        tools=round_tools,
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
                        piece, mem_piece = (
                            split_reply(content) if hide_mem else (content, None)
                        )
                        if mem_piece:
                            model_mem = mem_piece
                        if piece:
                            if final_text and piece not in final_text:
                                final_text = final_text.rstrip() + "\n\n" + piece
                            elif not final_text:
                                final_text = piece
                            if not (
                                hide_mem
                                and self._looks_like_lost_context_clarifier(
                                    final_text, user_text
                                )
                            ):
                                _release_hold()
                            break
                        if not final_text and empty_rounds < _MAX_EMPTY_ROUNDS:
                            # Nothing but the hidden memory block (or silence). Drop the
                            # dead turn and resample rather than showing an empty reply.
                            empty_rounds += 1
                            if self.messages and self.messages[-1] is message:
                                self.messages.pop()
                            continue
                        break

                # Preamble before tool calls (math, explanation) must survive the
                # later "I found a video…" message — don't drop it from the turn.
                # A "please clarify" preamble is a failed first generation; hide it.
                if content and tool_calls and not parsed_from_text:
                    piece, mem_piece = (
                        split_reply(content) if hide_mem else (content, None)
                    )
                    if mem_piece:
                        model_mem = mem_piece
                    if (
                        piece
                        and reply_looks_like_clarifier(piece)
                        and self._will_retry_clarifier(user_text)
                    ):
                        _discard_and_replace()
                    elif piece and piece not in (final_text or ""):
                        _release_hold()
                        final_text = (
                            final_text.rstrip() + "\n\n" + piece if final_text else piece
                        )
                    else:
                        _release_hold()

                self.last_tool_rounds += 1
                if streamed_any and on_token:
                    if not released_hold:
                        if (
                            content
                            and reply_looks_like_clarifier(content)
                            and self._will_retry_clarifier(user_text)
                        ):
                            _discard_and_replace()
                        else:
                            _release_hold()
                    if released_hold:
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
                    elif name == "web_search" and self.mode.role == "oac":
                        args, enriched = self._maybe_enrich_web_search(args, user_text)
                        if enriched:
                            call = {"function": {"name": name, "arguments": args}}
                    elif name == "query_calendar" and self.mode.role == "oac":
                        args = self._shape_calendar_query_args(args)
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
                            self._emit_tool_start(on_tool, name, args)
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
                        self._emit_tool_start(on_tool, name, args)
                    result = self._run_tool_call(call)
                    self.last_tool_names.append(name)
                    if name == "query_calendar" and isinstance(result, dict):
                        self._note_calendar_result(result, args)
                    if (
                        name == "query_db"
                        and isinstance(result, dict)
                        and result.get("ok")
                    ):
                        dest = str(args.get("dest") or args.get("file") or "")
                        if is_hayden_db_dest(dest) or query_result_includes_hayden(result):
                            self._turn_hayden_db_ok = True
                            digest = str(result.get("digest") or "").strip()
                            if digest:
                                self._hayden_db_digest = digest
                            if self._self_identity_turn(user_text):
                                self._force_prose_after_hayden_db = True
                    if (
                        name == "web_search"
                        and self.mode.role == "oac"
                        and self.mode.id != "deep_research"
                        and isinstance(result, dict)
                        and result.get("ok")
                        and not result.get("duplicate")
                        and not result.get("blocked")
                        and not self._user_opts_out_of_open(user_text)
                        and not self._self_identity_turn(user_text)
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
                        if name == "create_plot" and isinstance(result, dict) and result.get("ok"):
                            fig = result.get("figure")
                            if isinstance(fig, dict):
                                done_detail["plot"] = fig
                                done_detail["plot_meta"] = {
                                    "chart": str(result.get("chart") or ""),
                                    "title": str(result.get("title") or ""),
                                    "source": str(result.get("source") or ""),
                                    "animate": bool(result.get("animate", True)),
                                }
                        on_tool("done", name, done_detail)
                    payload = self._truncate_tool_result(name, result)
                    tool_msg: dict[str, Any] = {
                        "role": "tool",
                        "content": json.dumps(payload, ensure_ascii=False),
                    }
                    if name:
                        tool_msg["tool_name"] = name
                    call_id = call.get("id") if isinstance(call, dict) else None
                    if call_id:
                        tool_msg["tool_call_id"] = call_id
                    self.messages.append(tool_msg)
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

        if (
            hide_mem
            and self.mode.role == "oac"
            and self._personal_db_turn(user_text)
            and not (self._injected_context or "").strip()
            and (
                "query_db" not in set(self.last_tool_names or [])
                or looks_like_bad_self_reply(final_text or "")
            )
        ):
            nudged = self._retry_personal_with_query_db(
                user_text,
                stream=stream,
                on_token=on_token,
                on_thinking=on_thinking,
                on_tool=on_tool,
                think=think,
                req_timeout=req_timeout,
                vis_filter=vis_filter,
                hide_mem=hide_mem,
            )
            if nudged:
                final_text = nudged
                streamed_parts = [nudged]

        if (
            hide_mem
            and self.mode.role == "oac"
            and not (final_text or "").strip()
            and (
                self._personal_db_turn(user_text)
                or (self._injected_context or "").strip()
            )
        ):
            if not (self._injected_context or getattr(self, "_hayden_db_digest", "") or "").strip():
                nudged = self._retry_personal_with_query_db(
                    user_text,
                    stream=stream,
                    on_token=on_token,
                    on_thinking=on_thinking,
                    on_tool=on_tool,
                    think=think,
                    req_timeout=req_timeout,
                    vis_filter=vis_filter,
                    hide_mem=hide_mem,
                )
                if nudged:
                    final_text = nudged
                    streamed_parts = [nudged]
            if not (final_text or "").strip():
                retry = self._retry_answer_from_hayden_db(
                    user_text,
                    stream=stream,
                    on_token=on_token,
                    on_thinking=on_thinking,
                    think=think,
                    req_timeout=req_timeout,
                    vis_filter=vis_filter,
                )
                if retry:
                    final_text = retry
                    streamed_parts = [retry]

        if (
            hide_mem
            and self.mode.role == "oac"
            and self._self_identity_turn(user_text)
            and looks_like_bad_self_reply(final_text or "")
        ):
            retry = self._retry_answer_from_hayden_db(
                user_text,
                stream=stream,
                on_token=on_token,
                on_thinking=on_thinking,
                think=think,
                req_timeout=req_timeout,
                vis_filter=vis_filter,
            )
            if retry and not looks_like_bad_self_reply(retry):
                final_text = retry
                streamed_parts = [retry]

        if (
            hide_mem
            and self.mode.role == "oac"
            and self._needs_model_calendar_tool(user_text)
        ):
            retry = self._nudge_query_calendar(
                user_text,
                stream=stream,
                on_token=on_token,
                on_thinking=on_thinking,
                on_tool=on_tool,
                think=think,
                req_timeout=req_timeout,
                vis_filter=vis_filter,
                tools=tools,
            )
            if retry:
                final_text = retry
                streamed_parts = [retry]

        if hide_mem and self.mode.role == "oac" and self._calendar_turn(user_text):
            from ainet.calendar_store import looks_like_calendar_write

            if not looks_like_calendar_write(user_text):
                spoken = self._spoken_calendar_reply()
                if spoken:
                    self._begin_host_retry("calendar_format")
                    final_text = spoken
                    streamed_parts = [spoken]
                    if self.messages and self.messages[-1].get("role") == "assistant":
                        self.messages[-1]["content"] = spoken
                    else:
                        self.messages.append({"role": "assistant", "content": spoken})
                    if on_token is not None:
                        on_token(spoken)

        if hide_mem and self.mode.role == "oac":
            forced_cal = self._force_calendar_write_if_needed(
                user_text,
                final_text or "",
                on_tool=on_tool,
            )
            if forced_cal:
                final_text = forced_cal
                streamed_parts = [forced_cal]
                if on_token is not None:
                    on_token(forced_cal)

        if (
            hide_mem
            and self.mode.role == "oac"
            and self._turn_had_web_evidence()
            and self._looks_like_stale_clarifier(final_text or "", user_text)
        ):
            retry = self._retry_answer_after_web(
                user_text,
                stream=stream,
                on_token=on_token,
                on_thinking=on_thinking,
                think=think,
                req_timeout=req_timeout,
                vis_filter=vis_filter,
            )
            if retry:
                final_text = retry
                streamed_parts = [retry]

        if hide_mem and self.mode.role == "oac":
            cleaned = strip_leading_clarifier(final_text or "")
            if cleaned and cleaned != (final_text or "").strip():
                final_text = cleaned

        if (
            hide_mem
            and self.mode.role == "oac"
            and not self._turn_had_web_evidence()
            and self._looks_like_lost_context_clarifier(final_text or "", user_text)
        ):
            retry = self._retry_answer_with_recent_context(
                user_text,
                stream=stream,
                on_token=on_token,
                on_thinking=on_thinking,
                think=think,
                req_timeout=req_timeout,
                vis_filter=vis_filter,
            )
            if retry:
                final_text = retry
                streamed_parts = [retry]

        if hide_mem and self.mode.role == "oac" and not self._self_identity_turn(user_text):
            stripped = strip_profile_dump(final_text or "", self._injected_context)
            if stripped and stripped != (final_text or "").strip():
                final_text = stripped
            elif (
                looks_like_empty_therapy(final_text or "")
                or looks_like_generic_ignore(final_text or "", user_text)
                or looks_like_profile_dump(final_text or "")
                or looks_like_fresh_start(final_text or "")
            ):
                retry = self._retry_answer_in_thread(
                    user_text,
                    stream=stream,
                    on_token=on_token,
                    on_thinking=on_thinking,
                    think=think,
                    req_timeout=req_timeout,
                    vis_filter=vis_filter,
                )
                if retry:
                    final_text = strip_profile_dump(retry, self._injected_context) or retry
                    streamed_parts = [final_text]

        if not released_hold:
            _release_hold()
        if vis_filter is not None:
            vis_filter.flush()
        self._replace_stream = None
        if hide_mem:
            visible, mem = split_reply(final_text or "")
            final_text = visible
            mem = mem or model_mem
            if mem:
                self.convo_memory = mem
            else:
                self.convo_memory = host_fallback_memory(
                    user_text,
                    final_text,
                    self.convo_memory,
                    recent_turns=self.recent_turns,
                )
            self._strip_memory_from_stored_replies()
            self._record_completed_turn(
                user_text, final_text, links=turn_links[:8]
            )
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

    def _turn_had_web_evidence(self) -> bool:
        names = set(getattr(self, "last_tool_names", []) or [])
        return bool(names & {"web_search", "web_fetch"})

    @staticmethod
    def _looks_like_stale_clarifier(reply: str, user_text: str) -> bool:
        if not reply_looks_like_clarifier(reply):
            return False
        ask = (user_text or "").strip()
        return len(ask.split()) >= 3 and (
            "?" in ask
            or ask.casefold().startswith(
                ("who ", "what ", "how ", "when ", "where ", "why ", "which ")
            )
        )

    def _will_retry_clarifier(self, user_text: str) -> bool:
        has_context = bool(
            standing_request(self.convo_memory).strip()
            or self.recent_turns
            or (self.last_user_text and self.last_assistant_text)
        )
        if not has_context:
            return False
        return is_vague_pronoun_followup(user_text) or len((user_text or "").split()) <= 10

    def _looks_like_lost_context_clarifier(self, reply: str, user_text: str) -> bool:
        """First generation asked to clarify despite recent turns already naming the topic."""
        if not reply_looks_like_clarifier(reply):
            return False
        return self._will_retry_clarifier(user_text)

    def _begin_host_retry(self, reason: str) -> None:
        self.last_host_retry = reason
        replace = getattr(self, "_replace_stream", None)
        if callable(replace):
            replace()

    def _pronoun_bind_note(self, user_text: str) -> str:
        """Tell the model what 'it' is *before* it asks."""
        has_pronoun = bool(
            re.search(r"\b(?:it|that|this|them|those)\b", user_text or "", re.I)
        )
        if not has_pronoun and not is_vague_pronoun_followup(user_text):
            return ""
        topic = followup_topic(self.convo_memory, self.recent_turns)
        if not topic:
            return ""
        if not is_vague_pronoun_followup(user_text) and len((user_text or "").split()) > 18:
            return ""
        return (
            f"(Host: it/that/this in this message refers to: {topic}. "
            "Answer that. Do not ask what 'it' means or say the question is ambiguous.)"
        )

    def _retry_answer_with_recent_context(
        self,
        user_text: str,
        *,
        stream: bool,
        on_token: TokenCallback | None,
        on_thinking: ThinkingCallback | None,
        think: bool,
        req_timeout: float,
        vis_filter: VisibleTokenFilter | None,
    ) -> str:
        """Force an answer that binds pronouns to recent turns / standing request."""
        self._begin_host_retry("lost_context_clarifier")
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages.pop()
        topic = standing_request(self.convo_memory).strip()
        prior = ""
        if self.recent_turns:
            u, a = self.recent_turns[-1]
            prior = f"Latest prior ask was {u!r}; you answered about that topic."
        elif self.last_user_text:
            prior = f"Latest prior ask was {self.last_user_text!r}."
        topic_bit = f" Standing request: {topic}." if topic else ""
        self.messages.append(
            {
                "role": "user",
                "content": (
                    "Host note: Hayden's follow-up depends on recent conversation context. "
                    f"Message: {user_text!r}. {prior}{topic_bit} "
                    "Resolve pronouns (it/that/this) from Recent turns / Standing request and "
                    "answer now. Do not ask what 'it' means or ask clarifying questions."
                ),
            }
        )
        try:
            if stream and (on_token is not None or on_thinking is not None):

                def _tok(delta: str) -> None:
                    if vis_filter is not None:
                        vis_filter.feed(delta)
                    elif on_token is not None:
                        on_token(delta)

                response = self.client.chat_stream(
                    self.messages,
                    tools=None,
                    think=think,
                    timeout_s=req_timeout,
                    on_token=_tok if on_token is not None else None,
                    on_thinking=on_thinking,
                )
            else:
                response = self.client.chat(
                    self.messages,
                    tools=None,
                    think=think,
                    timeout_s=req_timeout,
                )
        except Exception:
            return ""
        message = response.get("message") or {}
        self.messages.append(message)
        content = str(message.get("content") or "")
        visible, _mem = split_reply(content)
        return (visible or content).strip()

    def _retry_answer_after_web(
        self,
        user_text: str,
        *,
        stream: bool,
        on_token: TokenCallback | None,
        on_thinking: ThinkingCallback | None,
        think: bool,
        req_timeout: float,
        vis_filter: VisibleTokenFilter | None,
    ) -> str:
        """One forced answer pass when the model ignored successful web evidence."""
        self._begin_host_retry("stale_clarifier_after_web")
        # Drop the bad assistant turn if it is still the last message.
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages.pop()
        self.messages.append(
            {
                "role": "user",
                "content": (
                    "Host note: web_search already returned results/fetched page text for "
                    f"this question: {user_text!r}. "
                    "Answer that question now from the tool results already in this turn. "
                    "Do not ask clarifying questions. Do not say the topic is vague."
                ),
            }
        )
        try:
            if stream and (on_token is not None or on_thinking is not None):

                def _tok(delta: str) -> None:
                    if vis_filter is not None:
                        vis_filter.feed(delta)
                    elif on_token is not None:
                        on_token(delta)

                response = self.client.chat_stream(
                    self.messages,
                    tools=None,
                    think=think,
                    timeout_s=req_timeout,
                    on_token=_tok if on_token is not None else None,
                    on_thinking=on_thinking,
                )
            else:
                response = self.client.chat(
                    self.messages,
                    tools=None,
                    think=think,
                    timeout_s=req_timeout,
                )
        except Exception:
            return ""
        message = response.get("message") or {}
        self.messages.append(message)
        content = str(message.get("content") or "")
        visible, _mem = split_reply(content)
        return (visible or content).strip()

    def _retry_personal_with_query_db(
        self,
        user_text: str,
        *,
        stream: bool,
        on_token: TokenCallback | None,
        on_thinking: ThinkingCallback | None,
        on_tool: ToolCallback | None,
        think: bool,
        req_timeout: float,
        vis_filter: VisibleTokenFilter | None,
        hide_mem: bool,
    ) -> str:
        """If the model skipped query_db on a personal ask, host-lookup then answer from digest."""
        self._begin_host_retry("personal_query_db")
        _ = on_tool, hide_mem
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages.pop()

        tokens = [
            t
            for t in re.split(r"\W+", user_text.casefold())
            if len(t) > 2
            and t
            not in {
                "what",
                "about",
                "have",
                "been",
                "with",
                "does",
                "did",
                "the",
                "and",
                "you",
                "any",
                "for",
                "how",
                "do",
                "who",
                "where",
                "when",
                "like",
                "know",
                "tell",
                "something",
            }
        ][:8]
        # Broad multi-word search first (omit dest). Self-identity also tries hayden.
        attempts: list[dict[str, Any]] = []
        if tokens:
            attempts.append({"q": " ".join(tokens)})
        if self._self_identity_turn(user_text):
            attempts.append({"dest": "hayden"})
        attempts.append({})  # full scan fallback

        result: dict[str, Any] = {"ok": False, "digest": "", "matches": []}
        args: dict[str, Any] = {}
        for args in attempts:
            call = {"function": {"name": "query_db", "arguments": args}}
            candidate = self._run_tool_call(call)
            if isinstance(candidate, dict) and (
                candidate.get("digest") or candidate.get("matches")
            ):
                result = candidate
                break
            if isinstance(candidate, dict):
                result = candidate

        self.last_tool_names = list(self.last_tool_names or []) + ["query_db"]

        digest = ""
        if isinstance(result, dict):
            digest = str(result.get("digest") or "").strip()
            if not digest:
                parts: list[str] = []
                for row in result.get("matches") or []:
                    if not isinstance(row, dict):
                        continue
                    for entry in row.get("entries") or []:
                        if isinstance(entry, dict) and entry.get("text"):
                            parts.append(str(entry["text"]))
                digest = "\n".join(parts[:12])

        self._turn_hayden_db_ok = True
        self._hayden_db_digest = digest

        self.messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "query_db", "arguments": args},
                    }
                ],
            }
        )
        self.messages.append(
            {
                "role": "tool",
                "content": json.dumps(
                    result if isinstance(result, dict) else {"ok": False},
                    ensure_ascii=False,
                ),
            }
        )

        return self._retry_answer_from_hayden_db(
            user_text,
            stream=stream,
            on_token=on_token,
            on_thinking=on_thinking,
            think=think,
            req_timeout=req_timeout,
            vis_filter=vis_filter,
        )

    def _retry_answer_from_hayden_db(
        self,
        user_text: str,
        *,
        stream: bool,
        on_token: TokenCallback | None,
        on_thinking: ThinkingCallback | None,
        think: bool,
        req_timeout: float,
        vis_filter: VisibleTokenFilter | None,
    ) -> str:
        """Force a plain spoken answer from stored digest — not meta, not role-play."""
        self._begin_host_retry("hayden_db_answer")
        digest = (getattr(self, "_hayden_db_digest", "") or "").strip()
        if not digest:
            digest = self._hayden_digest_from_messages()
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages.pop()
        body = digest or "(no stored observations)"
        self.messages.append(
            {
                "role": "user",
                "content": (
                    f"Hayden asked: {user_text!r}\n\n"
                    "Stored facts from query_db:\n"
                    f"{body}\n\n"
                "Answer NOW in plain speech, second person (you/your). "
                "You are AI1, not Hayden — never say I'm Hayden. "
                "Use only the facts that answer the question. "
                "No meta talk, no biography dump, no offers to summarize, no web search."
                ),
            }
        )
        try:
            if stream and (on_token is not None or on_thinking is not None):

                def _tok(delta: str) -> None:
                    if vis_filter is not None:
                        vis_filter.feed(delta)
                    elif on_token is not None:
                        on_token(delta)

                response = self.client.chat_stream(
                    self.messages,
                    tools=None,
                    think=think,
                    timeout_s=req_timeout,
                    on_token=_tok if on_token is not None else None,
                    on_thinking=on_thinking,
                    should_cancel=self.cancelled,
                )
            else:
                response = self.client.chat(
                    self.messages,
                    tools=None,
                    think=think,
                    timeout_s=req_timeout,
                )
        except Exception:
            return ""
        message = response.get("message") or {}
        self.messages.append(message)
        content = str(message.get("content") or "")
        visible, _mem = split_reply(content)
        return (visible or content).strip()

    def _hayden_digest_from_messages(self) -> str:
        for message in reversed(self.messages):
            if message.get("role") != "tool":
                continue
            raw = message.get("content")
            try:
                data = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(data, dict):
                continue
            digest = str(data.get("digest") or "").strip()
            if digest:
                return digest
        return ""

    def _maybe_enrich_web_search(
        self, args: dict[str, Any], user_text: str
    ) -> tuple[dict[str, Any], bool]:
        raw_q = args.get("query")
        query = str(raw_q or "").strip()
        topic = topic_for_search(self.convo_memory, self.recent_turns) or standing_request(
            self.convo_memory
        )
        new_q, changed = enrich_web_search_query(query, user_text, topic)
        if not changed:
            return args, False
        out = dict(args)
        out["query"] = new_q
        return out, True

    def _emit_tool_start(
        self,
        on_tool: ToolCallback | None,
        name: str,
        args: dict[str, Any] | None,
    ) -> None:
        if on_tool is None:
            return
        payload = dict(args or {})
        about = _clip_about(str(payload.get("about") or ""))
        if not about:
            about = default_tool_about(name, payload)
        if about:
            payload["about"] = about
        on_tool("start", name, {"arguments": payload})

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
            fetched = result.get("fetched")
            if isinstance(fetched, list) and fetched:
                ok_n = sum(1 for row in fetched if isinstance(row, dict) and row.get("text"))
                base += f" · fetched {ok_n}"
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
        if name == "create_plot":
            return str(result.get("summary") or result.get("chart") or "plot")
        if name == "spotify":
            return str(result.get("summary") or result.get("action") or "spotify")
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
        if name == "query_db":
            return f"{result.get('count', 0)} matches"
        if name == "query_calendar":
            return f"{result.get('count', 0)} events"
        if name == "add_calendar_event":
            ev = result.get("event") if isinstance(result.get("event"), dict) else {}
            return str(ev.get("title") or "added")
        if name == "update_calendar_event":
            ev = result.get("event") if isinstance(result.get("event"), dict) else {}
            return str(ev.get("title") or ev.get("id") or "updated")
        if name == "cancel_calendar_event":
            return "deleted" if result.get("deleted") else "cancelled"
        if name in {"list_dir", "tree"}:
            kids = result.get("children") or result.get("entries")
            if isinstance(kids, list):
                return f"{len(kids)} entries"
        return "ok"

    def _commit_host_reply(
        self,
        user_text: str,
        reply: str,
        *,
        on_token: TokenCallback | None,
        tool_names: list[str] | None = None,
        links: list[tuple[str, str]] | None = None,
    ) -> str:
        if on_token:
            on_token(reply)
        self.messages.append({"role": "assistant", "content": reply})
        if tool_names is not None:
            self.last_tool_names = list(tool_names)
        self._record_completed_turn(user_text, reply, links=links)
        self.convo_memory = host_fallback_memory(
            user_text, reply, self.convo_memory, recent_turns=self.recent_turns
        )
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

    def _host_followup_intent(self, user_text: str) -> str | None:
        """videos / search / ack — host handles these so the model cannot re-lecture."""
        last_a = self.last_assistant_text or (
            self.recent_turns[-1][1] if self.recent_turns else ""
        )
        pending_video = last_user_wanted_videos(self.recent_turns) or offered_to_search(
            last_a
        )
        if wants_videos(user_text):
            return "videos"
        if is_vague_search_followup(user_text) and not wants_videos(user_text):
            return "videos" if pending_video else "search"
        if is_action_confirm(user_text):
            # Calendar offers must never fall through to web_search.
            if offered_to_calendar(last_a):
                return None
            from ainet.calendar_store import looks_like_calendar_write

            prior = self.last_user_text or ""
            if looks_like_calendar_write(prior):
                return None
            if pending_video or wants_videos(self.last_user_text or ""):
                return "videos"
            if offered_to_search(last_a) or is_vague_search_followup(
                self.last_user_text or ""
            ):
                return "search"
            return None
        if is_short_ack(user_text):
            if pending_video:
                return "videos"
            return "ack"
        return None

    def _calendar_source_text(self, user_text: str) -> str:
        """For 'yes' after a calendar offer, use the prior ask that named the event."""
        from ainet.calendar_store import looks_like_calendar_write

        if looks_like_calendar_write(user_text):
            return user_text
        last_a = self.last_assistant_text or (
            self.recent_turns[-1][1] if self.recent_turns else ""
        )
        if is_action_confirm(user_text) and (
            offered_to_calendar(last_a)
            or looks_like_calendar_write(self.last_user_text or "")
        ):
            return self.last_user_text or standing_request(self.convo_memory) or ""
        return ""

    def _maybe_host_calendar_add(
        self,
        user_text: str,
        *,
        on_token: TokenCallback | None,
        on_tool: ToolCallback | None,
        force: bool = False,
        commit: bool = True,
    ) -> str | None:
        """Add a calendar event from clear wording (or a yes after offering to add)."""
        from ainet.calendar_store import (
            looks_like_calendar_write,
            parse_event_from_text,
        )

        source = self._calendar_source_text(user_text)
        if not source:
            if looks_like_calendar_write(user_text) or force:
                source = user_text
            else:
                return None
        payload = parse_event_from_text(source)
        if not payload:
            return None
        # Only auto-add when we have a concrete start (timed or dated).
        if not payload.get("start"):
            return None
        about = str(payload.get("title") or "new event")
        if on_tool:
            self._emit_tool_start(
                on_tool,
                "add_calendar_event",
                {**payload, "about": about},
            )
        try:
            result = self._run_tool_call(
                {"function": {"name": "add_calendar_event", "arguments": dict(payload)}}
            )
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "error": str(exc)}
        self.last_tool_names = list(getattr(self, "last_tool_names", None) or []) + [
            "add_calendar_event"
        ]
        ok = bool(isinstance(result, dict) and result.get("ok"))
        summary = self._tool_result_summary(
            "add_calendar_event", result if isinstance(result, dict) else {}
        )
        if on_tool:
            on_tool(
                "done",
                "add_calendar_event",
                {"ok": ok, "summary": summary or ("saved" if ok else "failed")},
            )
        if not ok:
            err = ""
            if isinstance(result, dict):
                err = str(result.get("error") or "")
            reply = f"Couldn't add that to your calendar{': ' + err if err else '.'}"
            if not commit:
                return reply
            return self._commit_host_reply(
                user_text,
                reply,
                on_token=on_token,
                tool_names=list(getattr(self, "last_tool_names", None) or []),
            )
        event = result.get("event") if isinstance(result, dict) else None
        title = ""
        start = ""
        end = ""
        loc = ""
        if isinstance(event, dict):
            title = str(event.get("title") or payload.get("title") or "Event")
            start = str(event.get("start") or payload.get("start") or "")
            end = str(event.get("end") or payload.get("end") or "")
            loc = str(event.get("location") or payload.get("location") or "")
        else:
            title = str(payload.get("title") or "Event")
            start = str(payload.get("start") or "")
            end = str(payload.get("end") or "")
            loc = str(payload.get("location") or "")
        when = start
        if end and end != start:
            when = f"{start} to {end}"
        bits = [f"Added {title}", when]
        if loc:
            bits.append(loc)
        reply = " · ".join(b for b in bits if b)
        if not commit:
            return reply
        return self._commit_host_reply(
            user_text,
            reply,
            on_token=on_token,
            tool_names=list(getattr(self, "last_tool_names", None) or []),
        )

    def _force_calendar_write_if_needed(
        self,
        user_text: str,
        final_text: str,
        *,
        on_tool: ToolCallback | None,
    ) -> str | None:
        """If Hayden asked to set the calendar and nothing was written, write it now."""
        from ainet.calendar_store import looks_like_calendar_write, parse_event_from_text

        _ = final_text
        if not looks_like_calendar_write(user_text):
            return None
        names = set(getattr(self, "last_tool_names", None) or [])
        if "add_calendar_event" in names:
            return None
        if not parse_event_from_text(self._calendar_source_text(user_text) or user_text):
            return None
        self._begin_host_retry("calendar_write")
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages.pop()
        reply = self._maybe_host_calendar_add(
            user_text,
            on_token=None,
            on_tool=on_tool,
            force=True,
            commit=False,
        )
        if not reply:
            return None
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def _shape_calendar_query_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Pin the lookup to the date Hayden asked for; treat 'today's stuff' as today."""
        from ainet.calendar_store import infer_schedule_query, looks_like_calendar_write

        ask = str(getattr(self, "_turn_user_text", "") or "")
        if looks_like_calendar_write(ask):
            return args
        blob = " ".join(
            part
            for part in (
                ask,
                str((args or {}).get("q") or ""),
                str((args or {}).get("about") or ""),
                str((args or {}).get("query") or ""),
            )
            if str(part or "").strip()
        )
        if not blob.strip():
            return args
        inferred = infer_schedule_query(blob)
        out = dict(args or {})
        out["start"] = str(inferred.get("start") or out.get("start") or "")
        out["end"] = str(inferred.get("end") or out.get("end") or "")
        q = str(inferred.get("q") or "").strip()
        if q:
            out["q"] = q
        else:
            out.pop("q", None)
            out.pop("query", None)
        if inferred.get("date_explicit"):
            out.pop("upcoming", None)
        elif inferred.get("upcoming"):
            out["upcoming"] = inferred.get("upcoming")
        else:
            out.pop("upcoming", None)
        out["about"] = str(inferred.get("about") or out.get("about") or "calendar")
        return out

    def _note_calendar_result(self, result: dict[str, Any], args: dict[str, Any] | None = None) -> None:
        rows = [e for e in (result.get("events") or []) if isinstance(e, dict)]
        self._turn_calendar_rows = rows
        about = ""
        if isinstance(args, dict):
            about = str(args.get("about") or "").strip()
        if not about:
            about = default_tool_about(
                "query_calendar",
                {
                    "start": result.get("start"),
                    "end": result.get("end"),
                    "about": "",
                },
            )
        self._turn_calendar_about = about

    def _spoken_calendar_reply(self) -> str:
        from ainet.calendar_store import spoken_schedule

        rows = list(getattr(self, "_turn_calendar_rows", None) or [])
        about = str(getattr(self, "_turn_calendar_about", "") or "")
        if not rows and not about:
            return ""
        return spoken_schedule(rows, about=about)

    def _calendar_turn(self, user_text: str) -> bool:
        from ainet.calendar_store import looks_like_calendar_write, looks_like_schedule_ask

        if looks_like_calendar_write(user_text):
            return True
        return looks_like_schedule_ask(user_text) or should_prefetch_calendar(
            user_text, standing_request(self.convo_memory), self.recent_turns
        )

    @staticmethod
    def _looks_like_missing_calendar_reply(reply: str) -> bool:
        low = (reply or "").casefold()
        if not low.strip():
            return False
        markers = (
            "don't have access",
            "do not have access",
            "no access to your",
            "don't have your",
            "do not have your",
            "can't access your schedule",
            "cannot access your schedule",
            "check your course syllabus",
            "contact your academic advisor",
            "university portal",
            "i don't have your personal",
            "i do not have your personal",
            "specific details about your enrollment",
        )
        return any(m in low for m in markers)

    def _needs_model_calendar_tool(self, user_text: str) -> bool:
        from ainet.calendar_store import looks_like_calendar_write

        if looks_like_calendar_write(user_text):
            return False
        if not self._calendar_turn(user_text):
            return False
        names = set(getattr(self, "last_tool_names", None) or [])
        return "query_calendar" not in names

    def _nudge_query_calendar(
        self,
        user_text: str,
        *,
        stream: bool,
        on_token: TokenCallback | None,
        on_thinking: ThinkingCallback | None,
        on_tool: ToolCallback | None,
        think: bool,
        req_timeout: float,
        vis_filter: VisibleTokenFilter | None,
        tools: list[dict[str, Any]] | None,
    ) -> str:
        """If the model skipped query_calendar on a schedule ask, tell it to call it."""
        from ainet.calendar_store import infer_schedule_query

        self._begin_host_retry("calendar_tool")
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages.pop()
        inferred = infer_schedule_query(user_text)
        start = str(inferred.get("start") or "")
        end = str(inferred.get("end") or "")
        self.messages.append(
            {
                "role": "user",
                "content": (
                    f"Host note: Hayden asked about his schedule: {user_text!r}. "
                    "You have not called query_calendar yet. Call query_calendar now "
                    f"with start={start!r} and end={end!r}. Leave q empty. "
                    "Do not invent events. Do not answer until the tool returns."
                ),
            }
        )
        try:
            response = self.client.chat(
                self.messages,
                tools=tools,
                think=think,
                timeout_s=req_timeout,
            )
        except Exception:
            return ""
        message = response.get("message") or {}
        self.messages.append(message)
        tool_calls = message.get("tool_calls") or []
        for call in tool_calls:
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            raw_args = fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    args = {}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}
            if name != "query_calendar":
                continue
            args = self._shape_calendar_query_args(args)
            if on_tool:
                self._emit_tool_start(on_tool, name, args)
            result = self._run_tool_call(
                {"function": {"name": name, "arguments": args}}
            )
            self.last_tool_names = list(self.last_tool_names or []) + [name]
            if name == "query_calendar" and isinstance(result, dict):
                self._note_calendar_result(result, args)
            if on_tool:
                summary = self._tool_result_summary(
                    name, result if isinstance(result, dict) else {}
                )
                on_tool(
                    "done",
                    name,
                    {
                        "ok": bool(isinstance(result, dict) and result.get("ok")),
                        "summary": summary,
                    },
                )
            self.messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        result if isinstance(result, dict) else {"ok": False},
                        ensure_ascii=False,
                    ),
                }
            )
        if "query_calendar" not in set(self.last_tool_names or []):
            # Model still skipped the tool — run the real lookup (not auto-context).
            from ainet.calendar_store import infer_schedule_query

            inferred = infer_schedule_query(user_text)
            args = {
                "start": str(inferred.get("start") or ""),
                "end": str(inferred.get("end") or ""),
                "about": str(inferred.get("about") or "schedule"),
            }
            if on_tool:
                self._emit_tool_start(on_tool, "query_calendar", args)
            result = self._run_tool_call(
                {"function": {"name": "query_calendar", "arguments": args}}
            )
            self.last_tool_names = list(self.last_tool_names or []) + ["query_calendar"]
            if isinstance(result, dict):
                self._note_calendar_result(result, args)
            if on_tool:
                summary = self._tool_result_summary(
                    "query_calendar", result if isinstance(result, dict) else {}
                )
                on_tool(
                    "done",
                    "query_calendar",
                    {
                        "ok": bool(isinstance(result, dict) and result.get("ok")),
                        "summary": summary,
                    },
                )
            self.messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        result if isinstance(result, dict) else {"ok": False},
                        ensure_ascii=False,
                    ),
                }
            )

        if "query_calendar" not in set(self.last_tool_names or []):
            return ""
        return self._spoken_calendar_reply() or ""

    def _retry_answer_from_calendar(
        self,
        user_text: str,
        *,
        stream: bool,
        on_token: TokenCallback | None,
        on_thinking: ThinkingCallback | None,
        on_tool: ToolCallback | None,
        think: bool,
        req_timeout: float,
        vis_filter: VisibleTokenFilter | None,
    ) -> str:
        """Kept for callers; schedule asks now go through _nudge_query_calendar."""
        return self._nudge_query_calendar(
            user_text,
            stream=stream,
            on_token=on_token,
            on_thinking=on_thinking,
            on_tool=on_tool,
            think=think,
            req_timeout=req_timeout,
            vis_filter=vis_filter,
            tools=self._active_tools(),
        )

    def _maybe_host_followup_action(
        self,
        user_text: str,
        *,
        on_token: TokenCallback | None,
        on_tool: ToolCallback | None,
    ) -> str | None:
        """Run video/search follow-ups (and short acks) without waiting on the model."""
        intent = self._host_followup_intent(user_text)
        if intent is None:
            return None
        if intent == "ack":
            return self._commit_host_reply(
                user_text, "Yep.", on_token=on_token, tool_names=[]
            )
        topic = named_search_topic(user_text) or topic_for_search(
            self.convo_memory, self.recent_turns
        )
        if not topic:
            return None
        if intent == "videos":
            query = topic if re.search(r"\byoutube\b", topic, re.I) else f"{topic} youtube"
        else:
            query = topic
        if on_tool:
            card = topic if intent == "videos" else query
            self._emit_tool_start(
                on_tool,
                "web_search",
                {"query": query, "auto": True, "about": card},
            )
        try:
            result = self._run_tool_call(
                {"function": {"name": "web_search", "arguments": {"query": query}}}
            )
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "error": str(exc), "query": query, "results": []}
        self.last_tool_names = list(self.last_tool_names or []) + ["web_search"]
        if not isinstance(result, dict):
            result = {"ok": False, "query": query, "results": []}
        result["query"] = str(result.get("query") or query)
        if not self._user_opts_out_of_open(user_text):
            result = self._auto_open_from_search(
                result,
                on_tool=on_tool,
                seen_tool_keys=set(),
                fetch=False,
            )
        opened = result.get("auto_opened") if isinstance(result.get("auto_opened"), list) else []
        if on_tool:
            summary = self._tool_result_summary("web_search", result)
            done: dict[str, Any] = {
                "ok": bool(result.get("ok", True)),
                "summary": summary,
            }
            if opened:
                done["articles"] = opened[:3]
            on_tool("done", "web_search", done)
        titles = [
            str(row.get("title") or row.get("url") or "").strip()
            for row in opened
            if isinstance(row, dict)
        ]
        titles = [t for t in titles if t][:3]
        if titles:
            lines = "\n".join(f"- {t}" for t in titles)
            n = len(titles)
            reply = (
                f"Opened {n} video{'s' if n != 1 else ''} in Chrome:\n{lines}"
                if intent == "videos"
                else f"Opened {n} tab{'s' if n != 1 else ''} in Chrome:\n{lines}"
            )
        elif result.get("ok") and result.get("results"):
            reply = "Searched, but nothing good to open."
        else:
            err = str(result.get("error") or "search failed")
            reply = f"Couldn't search: {err}"
        links = [
            (str(row.get("title") or ""), str(row.get("url") or ""))
            for row in opened
            if isinstance(row, dict) and str(row.get("url") or "").startswith("http")
        ]
        return self._commit_host_reply(
            user_text,
            reply,
            on_token=on_token,
            tool_names=list(self.last_tool_names or []),
            links=links,
        )

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
            self._emit_tool_start(on_tool, "open_chrome", {"urls": urls})
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
        return self._commit_host_reply(
            user_text,
            reply,
            on_token=on_token,
            tool_names=["open_chrome"],
            links=rows,
        )

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
        fetch: bool = True,
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
            self._emit_tool_start(
                on_tool, "open_chrome", {"urls": urls, "new_tab": True}
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

        q = str(search_result.get("query") or "").lower()
        want_video = any(
            w in q for w in ("youtube", "youtu.be", "video", "watch", "tutorial")
        )
        if not fetch or want_video:
            out = dict(search_result)
            out["auto_opened"] = cards
            out["hint"] = (
                f"Host already opened {len(cards)} URL(s) in Chrome (see auto_opened). "
                "Confirm that briefly. Do not re-explain the topic."
            )
            return out

        from ainet.tools.web import web_fetch

        fetched: list[dict[str, str]] = []
        for pick in cards[:2]:
            url = str(pick.get("url") or "")
            if not url.startswith("http"):
                continue
            try:
                page = web_fetch(url, max_chars=2800)
            except Exception as exc:  # noqa: BLE001
                fetched.append(
                    {
                        "title": str(pick.get("title") or ""),
                        "url": url,
                        "error": str(exc)[:200],
                    }
                )
                continue
            if not isinstance(page, dict) or not page.get("ok"):
                fetched.append(
                    {
                        "title": str(pick.get("title") or ""),
                        "url": url,
                        "error": str((page or {}).get("error") or "fetch failed")[:200],
                    }
                )
                continue
            fetched.append(
                {
                    "title": str(pick.get("title") or page.get("title") or ""),
                    "url": url,
                    "text": str(page.get("text") or "")[:2800],
                }
            )

        out = dict(search_result)
        out["auto_opened"] = cards
        if fetched:
            out["fetched"] = fetched
        out["hint"] = (
            f"Host already opened {len(cards)} URL(s) in Chrome (see auto_opened) "
            f"and fetched page text (see fetched). "
            "Answer Hayden NOW from snippets + fetched. "
            "Do not ask clarifying questions. Do not ask whether to search again. "
            "Do not reuse an earlier 'too vague / please specify' reply."
        )
        return out

    def _trim_history(self) -> None:
        """Keep recent dialogue under a message/char budget without breaking tool sequences."""
        system = [m for m in self.messages if m.get("role") == "system"]
        rest = [self._compact_message(m) for m in self.messages if m.get("role") != "system"]
        rest = self._drop_orphan_tools(rest)

        # OAC: memory + last 2 prior prose turns + current turn. Drop older tool dumps.
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
        """Last N prior user+reply (text only) + current user turn (including in-flight tools)."""
        user_idxs = [i for i, m in enumerate(rest) if m.get("role") == "user"]
        if not user_idxs:
            return rest
        current = user_idxs[-1]
        keep_start_idx = max(0, len(user_idxs) - 1 - _OAC_PRIOR_TURNS)
        out: list[dict[str, Any]] = []
        for pos in range(keep_start_idx, len(user_idxs)):
            start = user_idxs[pos]
            end = user_idxs[pos + 1] if pos + 1 < len(user_idxs) else len(rest)
            chunk = rest[start:end]
            if start == current:
                out.extend(chunk)
            else:
                out.extend(self._prose_only_turn(chunk))
        return self._drop_orphan_tools(out)

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
        if name == "create_plot" and isinstance(result, dict):
            ok = bool(result.get("ok", True)) and not result.get("error")
            slim: dict[str, Any] = {"ok": ok}
            if result.get("error"):
                slim["error"] = str(result["error"])[:400]
                slim["note"] = "Plot failed; tell Hayden it failed. Do not claim a chart was shown."
                return slim
            slim.update(
                {
                    "chart": result.get("chart"),
                    "title": result.get("title"),
                    "series_count": result.get("series_count"),
                    "summary": result.get("summary"),
                    "source": result.get("source"),
                    "note": "Chart rendered in the chat UI.",
                }
            )
            return slim
        # Keep search + auto-fetched page text intact for OAC answers.
        if name == "web_search" and self.mode.role == "oac":
            limit = max(limit, 9000)
        if name == "web_fetch" and self.mode.role == "oac":
            limit = max(limit, 5000)
        if name == "query_db" and isinstance(result, dict) and self.mode.role == "oac":
            slim = dict(result)
            note = (
                "Answer from the facts that matter to THIS message. "
                "Do not recap Hayden's whole biography or list unrelated traits."
            )
            existing = str(slim.get("hint") or "").strip()
            slim["hint"] = f"{existing} {note}".strip() if existing else note
            return slim
        if len(raw) <= limit:
            return result
        if name == "web_search" and isinstance(result, dict):
            slim = {
                "ok": result.get("ok", True),
                "query": result.get("query"),
                "count": result.get("count"),
                "provider": result.get("provider"),
                "hint": result.get("hint"),
                "results": (result.get("results") or [])[:5],
                "auto_opened": result.get("auto_opened"),
                "fetched": result.get("fetched"),
                "truncated": True,
            }
            slim_raw = json.dumps(slim, ensure_ascii=False)
            if len(slim_raw) <= limit:
                return slim
            # Drop fetched bodies last if still too large.
            fetched = slim.get("fetched")
            if isinstance(fetched, list):
                slim["fetched"] = [
                    {
                        "title": row.get("title"),
                        "url": row.get("url"),
                        "text": str(row.get("text") or "")[:1200],
                        "error": row.get("error"),
                    }
                    if isinstance(row, dict)
                    else row
                    for row in fetched[:2]
                ]
            return slim
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
            allowed = (
                set(READ_TOOLS)
                | {"get_tools", "getTools"}
                | set(PROJECT_SESSION_TOOLS)
                | set(CALENDAR_SESSION_TOOLS)
            )
            if self.mode.tool_names:
                allowed |= set(self.mode.tool_names)
            # create_project / calendar writes are mutating but allowed as session tools.
            if name in PROJECT_SESSION_TOOLS or name in CALENDAR_SESSION_TOOLS:
                pass
            elif name not in allowed or name in _MUTATING:
                return {
                    "ok": False,
                    "error": (
                        f"OAC cannot use tool '{name}'. "
                        "Allowed: read/web + calendar + create/open/close project. "
                        "SOI files lasting DB writes from the changelog after idle."
                    ),
                }
        elif (
            not self.full_tools_unlocked
            and self.mode.tool_names is not None
            and name not in self.mode.tool_names
            and name not in {"get_tools", "getTools"}
            and name not in PROJECT_SESSION_TOOLS
            and name not in CALENDAR_SESSION_TOOLS
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

        if name == "open_chrome":
            guard = self._reject_non_web_chrome(args)
            if guard is not None:
                return guard

        if name == "web_fetch":
            guard = self._reject_db_path_as_web_url(args)
            if guard is not None:
                return guard

        if name == "query_calendar" and self.mode.role == "oac":
            args = self._shape_calendar_query_args(args)

        if name in {"web_search", "web_fetch"} and self.mode.role == "oac":
            guard = self._reject_open_web_for_self(
                str(getattr(self, "_turn_user_text", "") or "")
            )
            if guard is not None:
                return guard
            ask = str(getattr(self, "_turn_user_text", "") or "")
            from ainet.calendar_store import looks_like_calendar_write

            if looks_like_calendar_write(ask) or (
                is_action_confirm(ask)
                and (
                    offered_to_calendar(
                        self.last_assistant_text
                        or (self.recent_turns[-1][1] if self.recent_turns else "")
                    )
                    or looks_like_calendar_write(self.last_user_text or "")
                )
            ):
                return {
                    "ok": False,
                    "blocked": True,
                    "error": "web_search is blocked while adding a calendar event.",
                    "hint": (
                        "Call add_calendar_event with title, start, end (or duration_minutes), "
                        "and optional location. Do not search the web."
                    ),
                }

        if name == "query_db" and self.mode.role == "oac":
            dest = str(args.get("dest") or args.get("file") or "")
            ask = str(getattr(self, "_turn_user_text", "") or "")
            if hayden_asking_about_self(ask) and wrong_dest_for_self_query(dest):
                return {
                    "ok": False,
                    "error": (
                        "Hayden asked about himself, not someone else. "
                        "people.json is for other people only."
                    ),
                    "hint": (
                        "Call query_db again with dest=hayden (or name= for one trait like "
                        "curiosity or personality). Then answer from entries[].text in plain speech."
                    ),
                }

        if self.soi_folder_scope and name in {"patch_json", "refresh_read"}:
            guard = self._reject_soi_folder_scope(name, args)
            if guard is not None:
                return guard

        result = dispatch(self.db, name, args)
        if name == "spotify":
            try:
                from ainet.tools.spotify_log import append_spotify_use

                append_spotify_use(
                    self.db,
                    ask=str(getattr(self, "_turn_user_text", "") or ""),
                    args=args,
                    result=result if isinstance(result, dict) else {"result": result},
                    session_id=str(self.session_id or ""),
                )
            except Exception:
                pass
        if name == "move_path" and self.project_root and result.get("ok"):
            self._maybe_update_project_root_after_move(result)
        if self.mode.role == "soi":
            result = _redact_assistant_fields(result)
        return result

    def _self_identity_turn(self, user_text: str) -> bool:
        return self_identity_context(user_text, standing_request(self.convo_memory))

    def _personal_db_turn(self, user_text: str) -> bool:
        """Personal-life asks should use the DB, not the open web for biography."""
        if hayden_asking_about_self(user_text):
            return True
        if wants_open_web(user_text):
            return False
        if personal_memory_question(user_text):
            return True
        if personal_memory_question(standing_request(self.convo_memory)):
            return True
        return False

    def _prefetch_calendar_context(
        self,
        user_text: str,
        *,
        on_tool: ToolCallback | None = None,
    ) -> str:
        """Calendar is model-driven only — no host auto query / auto tool card."""
        _ = user_text, on_tool
        return ""

    def _prefetch_turn_context(
        self,
        user_text: str,
        *,
        on_tool: ToolCallback | None = None,
    ) -> None:
        """Pull relevant stored facts into the system prompt before the model answers."""
        self._injected_context = ""
        if (
            wants_videos(user_text)
            or is_action_confirm(user_text)
            or is_short_ack(user_text)
            or is_vague_search_followup(user_text)
        ):
            return
        parts: list[str] = []
        # Schedule asks: do not auto-run query_calendar or inject a fake digest.
        # OAC must call query_calendar itself. Still skip query_db so it doesn't steal.
        if should_prefetch_calendar(
            user_text, standing_request(self.convo_memory), self.recent_turns
        ) and not hayden_asking_about_self(user_text):
            self._injected_context = ""
            return
        if not should_prefetch(user_text, standing_request(self.convo_memory), self.recent_turns):
            self._injected_context = "\n\n".join(parts)
            return
        prior_user = " ".join(u for u, _a in (self.recent_turns or [])[-4:])
        tokens = extract_query_tokens(
            user_text, standing_request(self.convo_memory), prior_user
        )
        if hayden_asking_about_self(user_text):
            args: dict[str, Any] = {"dest": "hayden"}
        else:
            if not tokens:
                self._injected_context = "\n\n".join(parts)
                return
            args = {"q": " ".join(tokens[:10])}
        low = user_text.casefold()
        if any(w in low for w in (" pin", "pins", "password", "secret")):
            args["include_secrets"] = True
            if not args.get("dest"):
                args["dest"] = "secrets"
        if hayden_asking_about_self(user_text):
            card = "what we have on you"
        else:
            bits = tokens[-3:] if tokens else []
            card = " ".join(bits) if bits else "stored notes"
        if on_tool:
            try:
                self._emit_tool_start(
                    on_tool, "query_db", {**args, "auto": True, "about": card}
                )
            except Exception:
                pass
        try:
            result = self._run_tool_call(
                {"function": {"name": "query_db", "arguments": args}}
            )
        except Exception:
            if on_tool:
                try:
                    on_tool("done", "query_db", {"ok": False, "summary": "auto context failed"})
                except Exception:
                    pass
            self._injected_context = "\n\n".join(parts)
            return
        digest = ""
        count = 0
        if isinstance(result, dict):
            digest = str(result.get("digest") or "").strip()
            count = int(result.get("count") or 0)
        digest = compact_digest(digest, tokens)
        if on_tool:
            try:
                on_tool(
                    "done",
                    "query_db",
                    {
                        "ok": bool(digest),
                        "summary": f"auto context · {count} matches",
                    },
                )
            except Exception:
                pass
        if digest:
            parts.append(digest)
            self._hayden_db_digest = digest
            self.last_tool_names = list(getattr(self, "last_tool_names", None) or []) + [
                "query_db"
            ]
            if hayden_asking_about_self(user_text):
                self._turn_hayden_db_ok = True
                self._force_prose_after_hayden_db = True
        self._injected_context = "\n\n".join(parts)

    def _retry_answer_in_thread(
        self,
        user_text: str,
        *,
        stream: bool,
        on_token: TokenCallback | None,
        on_thinking: ThinkingCallback | None,
        think: bool,
        req_timeout: float,
        vis_filter: VisibleTokenFilter | None,
    ) -> str:
        """Force a reply that uses the live thread instead of therapy templates / bios."""
        self._begin_host_retry("empty_therapy_or_dump")
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages.pop()
        prior = ""
        if self.recent_turns:
            chunks = []
            for user, _ans in self.recent_turns[-4:]:
                bit = " ".join((user or "").split())[:160]
                if bit:
                    chunks.append(bit)
            prior = " Thread so far: " + " → ".join(chunks) if chunks else ""
        known = (self._injected_context or "").strip()
        known_bit = f" Relevant stored facts (use, do not recap): {known[:800]}" if known else ""
        self.messages.append(
            {
                "role": "user",
                "content": (
                    "Host note: your last draft ignored the live conversation or dumped a biography. "
                    f"Hayden just said: {user_text!r}.{prior}{known_bit} "
                    "Answer THIS message. Use the specifics he named (classes, gym, clubs, constraints). "
                    "If he pushed back, engage the pushback. "
                    "No numbered generic lists. No 'you're not alone' / 'I'm here' closer. "
                    "No 'You're a mechanical engineering student…' recap."
                ),
            }
        )
        try:
            if stream and (on_token is not None or on_thinking is not None):

                def _tok(delta: str) -> None:
                    if vis_filter is not None:
                        vis_filter.feed(delta)
                    elif on_token is not None:
                        on_token(delta)

                response = self.client.chat_stream(
                    self.messages,
                    tools=None,
                    think=think,
                    timeout_s=req_timeout,
                    on_token=_tok if on_token is not None else None,
                    on_thinking=on_thinking,
                )
            else:
                response = self.client.chat(
                    self.messages,
                    tools=None,
                    think=think,
                    timeout_s=req_timeout,
                )
        except Exception:
            return ""
        message = response.get("message") or {}
        self.messages.append(message)
        content = str(message.get("content") or "")
        visible, _mem = split_reply(content)
        return (visible or content).strip()

    def _reject_open_web_for_self(self, user_text: str) -> dict[str, Any] | None:
        if wants_open_web(user_text):
            return None
        if not hayden_asking_about_self(user_text):
            return None
        return {
            "ok": False,
            "blocked": True,
            "error": (
                "web_search and web_fetch are blocked while answering questions about "
                "Hayden's stored identity."
            ),
            "hint": (
                "Use query_db dest=hayden (or the injected KNOWN CONTEXT) "
                "and answer from digest or matches[].entries[].text. "
                "Do not search the open web for who Hayden is."
            ),
        }

    @staticmethod
    def _reject_db_path_as_web_url(args: dict[str, Any]) -> dict[str, Any] | None:
        """Block web_fetch on db filenames mistaken for URLs (e.g. https://hayden.json)."""
        url = str(args.get("url") or "").strip()
        if not url:
            return None
        if not url.lower().startswith(("http://", "https://")):
            if url.replace("\\", "/").endswith(".json"):
                return {
                    "ok": False,
                    "error": (
                        f"{url!r} is a local database path, not a web URL. "
                        "Use query_db or read_json on that file and answer from the matches."
                    ),
                }
            return None
        parsed = urllib.parse.urlparse(url)
        host = (parsed.netloc or "").casefold()
        if host.endswith(".json") and host.count(".") == 1:
            return {
                "ok": False,
                "error": (
                    f"{url!r} is not a real website; it looks like a database filename. "
                    "query_db already returns observation text in matches[].entries; answer from that."
                ),
            }
        return None

    @staticmethod
    def _reject_non_web_chrome(args: dict[str, Any]) -> dict[str, Any] | None:
        """Chrome opens web pages. Database paths mean the model should just answer."""
        targets = [str(args.get("url") or "")]
        raw_urls = args.get("urls")
        if isinstance(raw_urls, list):
            targets.extend(str(item or "") for item in raw_urls)
        targets = [t.strip() for t in targets if t.strip()]
        if not targets:
            return None
        if any(t.lower().startswith(("http://", "https://")) for t in targets):
            return None
        return {
            "ok": False,
            "error": (
                "open_chrome is for http(s) web pages only, and this is not one. "
                "Database files are already readable with query_db, read_json, or read_text — "
                "query or read the file and answer Hayden directly instead of opening anything."
            ),
        }

    def _reject_soi_folder_scope(
        self, name: str, args: dict[str, Any]
    ) -> dict[str, Any] | None:
        _ = name, args
        return None

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
        if name in PROJECT_SESSION_TOOLS | CALENDAR_SESSION_TOOLS | {
            "web_search",
            "web_fetch",
            "image_search",
            "create_plot",
            "open_chrome",
            "spotify",
            "get_tools",
            "getTools",
            "save_research",
            "inspect_research",
            "query_db",
            "log_item",
            "file_note",
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
