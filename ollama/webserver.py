"""LAN web chat for AINet OAC (stdlib only).

Binds to all interfaces by default so you can open http://<your-ip>:1111
from this PC or another device on the network.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse

from ollama.client import OllamaCancelled, OllamaClient, OllamaError
from ollama.config import OllamaConfig
from ollama.idle import IdleSOIWatcher
from ollama.inference_gate import INFERENCE_GATE
from ollama.modes import DEFAULT_MODE_ID, get_mode, list_modes
from ollama.session import ChatSession

_STATIC = Path(__file__).resolve().parent / "static"


class ChatApp:
    def __init__(self, config: OllamaConfig, mode_id: str = DEFAULT_MODE_ID) -> None:
        self.config = config
        self.lock = threading.RLock()
        self._ask_lock = threading.Lock()
        self._soi_seq = 0
        self.soi_log: deque[dict[str, Any]] = deque(maxlen=500)
        self._raw_seq = 0
        self.raw_log: deque[dict[str, Any]] = deque(maxlen=300)
        self._last_error: str | None = None
        self.session = ChatSession(
            mode=get_mode(mode_id),
            config=config,
            auto_mode=config.auto_mode,
        )
        self.watcher = IdleSOIWatcher(
            self.session,
            config,
            on_status=self._append_soi_log,
        )
        if config.soi_enabled:
            self.watcher.start()

    def _append_raw(self, *, source: str, **fields: Any) -> None:
        row = {k: v for k, v in fields.items() if v is not None}
        row["source"] = source
        row["ts"] = row.get("ts") or time.strftime("%Y-%m-%dT%H:%M:%S")
        with self.lock:
            self._raw_seq += 1
            row["id"] = self._raw_seq
            self.raw_log.append(row)

    def raw_lines_after(self, after_id: int = 0) -> list[dict[str, Any]]:
        with self.lock:
            return [row for row in self.raw_log if int(row.get("id") or 0) > after_id]

    def _append_soi_log(self, msg: str | dict[str, Any]) -> None:
        if isinstance(msg, dict):
            row = {k: v for k, v in msg.items() if v is not None}
            text = str(row.get("text") or "").rstrip()
            row["text"] = text
        else:
            text = str(msg or "").rstrip()
            row = {"text": text}
        if not text and not row.get("event"):
            return
        with self.lock:
            self._soi_seq += 1
            row["id"] = self._soi_seq
            self.soi_log.append(row)
        event = str(row.get("event") or "")
        if event in {"model_ask", "model_reply", "model_error"}:
            self._append_raw(
                source="soi",
                event=event,
                phase=row.get("phase"),
                text=row.get("text") or row.get("preview") or row.get("reply") or "",
                preview=row.get("preview"),
                reply=row.get("reply"),
                error=row.get("error"),
            )

    def soi_lines_after(self, after_id: int = 0) -> list[dict[str, Any]]:
        with self.lock:
            return [row for row in self.soi_log if int(row.get("id") or 0) > after_id]

    def _tts_client(self) -> None:
        return None

    def tts_status(self) -> dict[str, Any]:
        return {
            "tts_enabled": False,
            "tts_url": self.config.tts_url,
            "tts_language": self.config.tts_language,
            "tts_healthy": False,
        }

    def status(self) -> dict[str, Any]:
        import os

        with self.lock:
            base = {
                "ok": True,
                "model": self.config.model,
                "mode": self.session.mode.id,
                "auto_mode": self.session.auto_mode,
                "mode_locked": self.session.mode_locked,
                "project_root": self.session.project_root,
                "project_name": (
                    self.session.project_root.rsplit("/", 1)[-1]
                    if self.session.project_root
                    else None
                ),
                "session_id": self.session.session_id,
                "db_root": str(self.config.db_root),
                "public_url": (os.environ.get("AINET_PUBLIC_URL") or "").strip().rstrip("/")
                or None,
                "soi_enabled": self.config.soi_enabled,
                "soi_idle_seconds": self.config.soi_idle_seconds,
                "soi_running": self.watcher.running,
                "modes": [
                    {"id": m.id, "name": m.name, "description": m.description}
                    for m in list_modes()
                    if m.role == "oac"
                ],
                "soi_log": list(self.soi_log),
                "soi_log_seq": self._soi_seq,
                "raw_log_seq": self._raw_seq,
                "last_raw": list(self.raw_log)[-1] if self.raw_log else None,
                "last_error": self._last_error,
                "hang": {
                    "ask_lock": self._ask_lock.locked(),
                    "chat_cancel": self.session.cancelled(),
                    "inference": INFERENCE_GATE.snapshot(),
                },
            }
        base.update(self.tts_status())
        try:
            from ainet.tools import spotify as spotify_mod

            base["spotify"] = spotify_mod.connection_status()
        except Exception as exc:  # noqa: BLE001
            base["spotify"] = {"ok": False, "configured": False, "connected": False, "error": str(exc)}
        return base

    def ask(self, text: str) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "Empty message"}
        try:
            # Do not hold self.lock across Ollama inference — SOI status/logs need it.
            with self._ask_lock:
                reply = self.session.ask(text)
            self._append_raw(source="oac", user=text, reply=reply, model=self.config.model)
            with self.lock:
                payload = {
                    "ok": True,
                    "reply": reply,
                    "mode": self.session.mode.id,
                    "project_root": self.session.project_root,
                    "project_name": (
                        self.session.project_root.rsplit("/", 1)[-1]
                        if self.session.project_root
                        else None
                    ),
                    "session_id": self.session.session_id,
                    "soi_running": self.watcher.running,
                }
            payload.update(self.tts_status())
            return payload
        except OllamaError as exc:
            return {"ok": False, "error": str(exc)}

    def ask_stream(self, text: str, *, voice: bool = False) -> Iterator[dict[str, Any]]:
        """Yield SSE payloads: token / tool_* / done / error. Voice/TTS is disabled."""
        text = (text or "").strip()
        if not text:
            yield {"type": "error", "error": "Empty message"}
            return

        events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        _ = voice  # API compat; TTS removed from chat

        def _on_token(delta: str) -> None:
            if not delta:
                return
            events.put({"type": "token", "text": delta})

        def _on_tool(phase: str, name: str, detail: dict[str, Any]) -> None:
            detail = detail or {}
            if phase == "start":
                args = detail.get("arguments") or {}
                # Keep SSE small — huge save_research bodies stall / blank the tool card.
                slim: dict[str, Any] = {}
                if isinstance(args, dict):
                    for key in (
                        "title",
                        "query",
                        "url",
                        "urls",
                        "path",
                        "dest",
                        "entry_id",
                        "summary",
                        "question",
                        "chart",
                        "equation",
                        "action",
                        "xlab",
                        "ylab",
                    ):
                        if key in args and args[key] not in (None, ""):
                            slim[key] = args[key]
                    if not slim and args:
                        # Fallback: first short scalar
                        for key, val in args.items():
                            if key in {"body", "sources", "key_findings", "data", "text"}:
                                continue
                            if isinstance(val, (str, int, float, bool)):
                                slim[key] = val
                                break
                events.put(
                    {
                        "type": "tool_start",
                        "name": name,
                        "arguments": slim,
                    }
                )
            elif phase == "done":
                payload = {
                    "type": "tool_done",
                    "name": name,
                    "ok": bool(detail.get("ok", True)),
                    "summary": detail.get("summary") or "",
                }
                if isinstance(detail.get("research"), dict):
                    payload["research"] = detail["research"]
                imgs = detail.get("images")
                if isinstance(imgs, list) and imgs:
                    payload["images"] = imgs[:8]
                articles = detail.get("articles")
                if isinstance(articles, list) and articles:
                    payload["articles"] = articles[:3]
                plot = detail.get("plot")
                if isinstance(plot, dict) and plot.get("data") is not None:
                    payload["plot"] = plot
                    meta = detail.get("plot_meta")
                    if isinstance(meta, dict):
                        payload["plot_meta"] = meta
                events.put(payload)

        def _on_context(snap: dict[str, Any]) -> None:
            events.put({"type": "context", **(snap or {})})

        def _run() -> None:
            reply = ""
            cancelled = False
            acquired = False
            try:
                # Serialize asks, but don't wait forever if a prior turn is wedged.
                acquired = self._ask_lock.acquire(timeout=2.0)
                if not acquired:
                    # Prior ask likely stuck; force-cancel so the gate can free.
                    self.session.request_cancel()
                    acquired = self._ask_lock.acquire(timeout=8.0)
                    if not acquired:
                        events.put(
                            {
                                "type": "error",
                                "error": "Chat is still busy — press Stop, then try again",
                            }
                        )
                        return
                try:
                    reply = self.session.ask(
                        text,
                        stream=True,
                        on_token=_on_token,
                        on_tool=_on_tool,
                        on_context=_on_context,
                        on_wait=lambda snap: events.put(
                            {"type": "status", "phase": "queued", **(snap or {})}
                        ),
                    )
                    cancelled = self.session.cancelled()
                finally:
                    if acquired:
                        self._ask_lock.release()
                        acquired = False
                self._append_raw(
                    source="oac",
                    user=text,
                    reply=reply,
                    model=self.config.model,
                )
                with self.lock:
                    done_payload = {
                        "type": "done",
                        "ok": True,
                        "reply": reply,
                        "cancelled": cancelled,
                        "mode": self.session.mode.id,
                        "project_root": self.session.project_root,
                        "project_name": (
                            self.session.project_root.rsplit("/", 1)[-1]
                            if self.session.project_root
                            else None
                        ),
                        "session_id": self.session.session_id,
                        "soi_running": self.watcher.running,
                        "voice": False,
                    }
                done_payload.update(self.tts_status())
                events.put(done_payload)
            except OllamaCancelled:
                events.put(
                    {
                        "type": "done",
                        "ok": True,
                        "reply": "(stopped)",
                        "cancelled": True,
                        "mode": self.session.mode.id,
                        "session_id": self.session.session_id,
                        "soi_running": self.watcher.running,
                        "voice": False,
                    }
                )
            except OllamaError as exc:
                self._last_error = str(exc)
                self._append_raw(source="oac", event="error", error=str(exc))
                events.put({"type": "error", "error": str(exc)})
            except Exception as exc:
                self._last_error = str(exc)
                self._append_raw(source="oac", event="error", error=str(exc))
                events.put({"type": "error", "error": str(exc)})
            finally:
                if acquired:
                    try:
                        self._ask_lock.release()
                    except RuntimeError:
                        pass
                events.put(None)

        threading.Thread(target=_run, name="ainet-chat-stream", daemon=True).start()
        stall_limit_s = 60.0
        last_progress = time.monotonic()
        while True:
            try:
                item = events.get(timeout=1.0)
            except queue.Empty:
                item = None
                empty = True
            else:
                empty = False
                if item is None:
                    break

            # Wall clock, not a tick count: "queued" status heartbeats arrive every
            # second and must not disguise a model that is producing nothing.
            if time.monotonic() - last_progress >= stall_limit_s:
                self.session.request_cancel()
                self.watcher.cancel_job()
                INFERENCE_GATE.force_reset()
                err = "Model stalled with no output. Press Reset AI if Stop does nothing."
                self._last_error = err
                self._append_raw(source="oac", event="stall", error=err)
                yield {"type": "error", "error": err}
                break

            if empty:
                # Keeps the socket provably alive so a dead client is detected here
                # instead of leaking a connection until the turn ends.
                yield {"type": "ping"}
                continue
            if item.get("type") != "status":
                last_progress = time.monotonic()
            yield item

    def cancel(self, *, chat: bool = True, soi: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": True, "chat": False, "soi": False}
        if chat:
            self.session.request_cancel()
            out["chat"] = True
        if soi:
            out["soi"] = bool(self.watcher.cancel_job().get("stopped"))
        return {**out, **self.status()}

    def reset(self, *, hard: bool = False) -> dict[str, Any]:
        old = self.session
        old.request_cancel()
        if hard:
            self.watcher.cancel_job()
            INFERENCE_GATE.force_reset()
            try:
                old.client.cancel_active()
            except Exception:
                pass
        with self.lock:
            if hard:
                mode_id = old.mode.id
                auto = old.auto_mode
                self.session = ChatSession(
                    mode=get_mode(mode_id),
                    config=self.config,
                    auto_mode=auto,
                    resume_session=False,
                )
                self.session.reset()
                self.watcher.session = self.session
                # Zombie ask threads may still hold the old lock; new turns use this one.
                self._ask_lock = threading.Lock()
            else:
                self.session.reset()
            self.session.clear_cancel()
            self._last_error = None
            self._append_raw(
                source="sys",
                event="hard_reset" if hard else "reset",
                text="Reset AI" if hard else "New session",
            )
            return {"ok": True, "hard": hard, **self.status()}

    def list_chats(self) -> dict[str, Any]:
        store = self.session.store
        if store is None:
            return {"ok": True, "chats": [], "session_id": self.session.session_id}
        return {
            "ok": True,
            "chats": store.list_sessions(current_id=self.session.session_id),
            "session_id": self.session.session_id,
        }

    def get_chat(self, session_id: str) -> dict[str, Any]:
        store = self.session.store
        if store is None:
            return {"ok": False, "error": "Chat log is not enabled"}
        try:
            payload = store.session_payload(session_id)
        except (OSError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "chat": payload}

    def open_chat(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            result = self.session.open_stored_session(session_id)
            if not result.get("ok"):
                return result
            return {**result, **self.status()}

    def set_mode(self, mode_id: str) -> dict[str, Any]:
        with self.lock:
            mid = (mode_id or "").strip().lower().replace(" ", "_").replace("-", "_")
            if mid == "project" and not self.session.project_root:
                return {
                    "ok": False,
                    "error": "Open a project with the open_project tool first (or ask the AI to open one).",
                }
            try:
                mode = self.session.set_mode(mode_id, lock=True)
            except KeyError as exc:
                return {"ok": False, "error": str(exc)}
            if mode.role != "oac":
                self.session.set_mode(DEFAULT_MODE_ID, lock=False)
                return {"ok": False, "error": "Web chat only supports OAC modes"}
            return {"ok": True, **self.status()}

    def run_soi(self) -> dict[str, Any]:
        from ainet.tools import changelog as changelog_mod
        from ainet.tools.paths import DbPaths

        pending = len(changelog_mod.pending_oac_entries(DbPaths(self.config.db_root)))
        kicked = self.watcher.request_run()
        with self.lock:
            return {
                "ok": True,
                "started": bool(kicked.get("started")),
                "reason": kicked.get("reason") or "",
                "pending_changelog": pending,
                **self.status(),
            }

    def research_current(self) -> dict[str, Any]:
        from ainet.tools.research import get_current_brief

        brief = get_current_brief(self.session.db)
        if not brief:
            return {"ok": True, "brief": None}
        return {"ok": True, "brief": brief}

    def research_list(self) -> dict[str, Any]:
        from ainet.tools.research import list_briefs

        return {"ok": True, "briefs": list_briefs(self.session.db, limit=40)}

    def research_get(self, *, brief_id: str = "", path: str = "") -> dict[str, Any]:
        from ainet.tools.research import load_brief

        brief = load_brief(self.session.db, brief_id=brief_id, path=path)
        if not brief:
            return {"ok": False, "error": "Brief not found"}
        return {"ok": True, "brief": brief}

    def open_chrome_url(self, url: str = "", urls: list[str] | None = None) -> dict[str, Any]:
        from ainet.tools.browser import open_chrome

        try:
            return open_chrome(url or "", urls=urls, new_tab=True)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def shutdown(self) -> None:
        self.watcher.stop()


def _json_bytes(payload: dict[str, Any], status: int = 200) -> tuple[int, bytes, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return status, body, "application/json; charset=utf-8"


def make_handler(app: ChatApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AINetWeb/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep console readable: method path status
            sys_stderr = __import__("sys").stderr
            print(f"{self.address_string()} {fmt % args}", file=sys_stderr, flush=True)

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, CF-Access-Client-Id, CF-Access-Client-Secret")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _fail(self, exc: BaseException) -> None:
            try:
                status, body, ctype = _json_bytes(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500
                )
                self._send(status, body, ctype)
            except Exception:
                try:
                    self.wfile.write(b"HTTP/1.1 500 Internal Server Error\r\n\r\n")
                except Exception:
                    pass

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            try:
                self._do_GET()
            except Exception as exc:
                print(f"GET {self.path} crashed: {exc}", file=__import__("sys").stderr, flush=True)
                self._fail(exc)

        def _do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)
            if path in {"/", "/index.html"}:
                html = (_STATIC / "index.html").read_bytes()
                self._send(200, html, "text/html; charset=utf-8")
                return
            if path in {"/favicon.ico", "/favicon.svg"}:
                icon = _STATIC / "favicon.svg"
                if icon.is_file():
                    self._send(200, icon.read_bytes(), "image/svg+xml")
                    return
            if path == "/api/status":
                status, body, ctype = _json_bytes(app.status())
                self._send(status, body, ctype)
                return
            if path == "/api/chats":
                sid = (qs.get("id") or [""])[0]
                if sid:
                    payload = app.get_chat(str(sid))
                    code = 200 if payload.get("ok") else 404
                    status, body, ctype = _json_bytes(payload, code)
                    self._send(status, body, ctype)
                    return
                status, body, ctype = _json_bytes(app.list_chats())
                self._send(status, body, ctype)
                return
            if path == "/api/spotify/status":
                from ainet.tools import spotify as spotify_mod

                status, body, ctype = _json_bytes(spotify_mod.connection_status())
                self._send(status, body, ctype)
                return
            if path == "/auth/spotify":
                from ainet.tools import spotify as spotify_mod

                started = spotify_mod.begin_auth()
                if not started.get("ok"):
                    self._send(
                        400,
                        spotify_mod.auth_error_html(str(started.get("error") or "Not configured")),
                        "text/html; charset=utf-8",
                    )
                    return
                loc = str(started["url"])
                self.send_response(302)
                self._cors()
                self.send_header("Location", loc)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            if path == "/auth/spotify/callback":
                from ainet.tools import spotify as spotify_mod

                code = (qs.get("code") or [""])[0]
                state = (qs.get("state") or [""])[0]
                err = (qs.get("error") or [""])[0]
                if err:
                    self._send(
                        400,
                        spotify_mod.auth_error_html(str(err)),
                        "text/html; charset=utf-8",
                    )
                    return
                result = spotify_mod.finish_auth(code=str(code or ""), state=str(state or ""))
                if not result.get("ok"):
                    self._send(
                        400,
                        spotify_mod.auth_error_html(str(result.get("error") or "Auth failed")),
                        "text/html; charset=utf-8",
                    )
                    return
                self._send(200, spotify_mod.auth_success_html(), "text/html; charset=utf-8")
                return
            if path == "/api/research/current":
                status, body, ctype = _json_bytes(app.research_current())
                self._send(status, body, ctype)
                return
            if path == "/api/research/list":
                status, body, ctype = _json_bytes(app.research_list())
                self._send(status, body, ctype)
                return
            if path == "/api/research/get":
                brief_id = (qs.get("id") or qs.get("brief_id") or [""])[0]
                rel = (qs.get("path") or [""])[0]
                status, body, ctype = _json_bytes(
                    app.research_get(brief_id=str(brief_id or ""), path=str(rel or ""))
                )
                self._send(status, body, ctype)
                return
            if path == "/api/soi-log":
                try:
                    after = int((qs.get("after") or ["0"])[0] or 0)
                except ValueError:
                    after = 0
                payload = {
                    "ok": True,
                    "soi_running": app.watcher.running,
                    "lines": app.soi_lines_after(after),
                    "events": app.soi_lines_after(after),
                    "event_seq": app._soi_seq,
                }
                status, body, ctype = _json_bytes(payload)
                self._send(status, body, ctype)
                return
            if path == "/api/ollama-raw":
                try:
                    after = int((qs.get("after") or ["0"])[0] or 0)
                except ValueError:
                    after = 0
                payload = {
                    "ok": True,
                    "model": app.config.model,
                    "lines": app.raw_lines_after(after),
                    "seq": app._raw_seq,
                }
                status, body, ctype = _json_bytes(payload)
                self._send(status, body, ctype)
                return
            if path == "/api/soi-stream":
                try:
                    after = int((qs.get("after") or ["0"])[0] or 0)
                except ValueError:
                    after = 0
                # Must close after each burst — keep-alive SSE holds a browser
                # connection forever and starves Reset / chat when Ollama wedges.
                self.close_connection = True
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                deadline = time.monotonic() + 25.0
                try:
                    while time.monotonic() < deadline:
                        rows = app.soi_lines_after(after)
                        if rows:
                            after = int(rows[-1].get("id") or after)
                        blob = json.dumps(
                            {
                                "lines": rows,
                                "soi_running": app.watcher.running,
                                "seq": after,
                            },
                            ensure_ascii=False,
                        )
                        self.wfile.write(f"data: {blob}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        time.sleep(0.4)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass
                return
            self._send(404, b'{"ok":false,"error":"not found"}', "application/json")

        def do_POST(self) -> None:  # noqa: N802
            try:
                self._do_POST()
            except Exception as exc:
                print(f"POST {self.path} crashed: {exc}", file=__import__("sys").stderr, flush=True)
                self._fail(exc)

        def _do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._send(400, b'{"ok":false,"error":"invalid JSON"}', "application/json")
                return
            if not isinstance(data, dict):
                self._send(400, b'{"ok":false,"error":"JSON object required"}', "application/json")
                return

            if path == "/api/chat":
                payload = app.ask(str(data.get("message") or ""))
                code = 200 if payload.get("ok") else 400
                status, body, ctype = _json_bytes(payload, code)
                self._send(status, body, ctype)
                return
            if path == "/api/chat-stream":
                self.close_connection = True
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                try:
                    for event in app.ask_stream(
                        str(data.get("message") or ""),
                        voice=False,
                    ):
                        blob = json.dumps(event, ensure_ascii=False)
                        self.wfile.write(f"data: {blob}\n\n".encode("utf-8"))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    app.cancel(chat=True, soi=False)
                return
            if path == "/api/cancel":
                chat = data.get("chat", True)
                soi = data.get("soi", True)
                status, body, ctype = _json_bytes(
                    app.cancel(chat=bool(chat), soi=bool(soi))
                )
                self._send(status, body, ctype)
                return
            if path == "/api/reset":
                status, body, ctype = _json_bytes(
                    app.reset(hard=bool(data.get("hard")))
                )
                self._send(status, body, ctype)
                return
            if path == "/api/chats/open":
                payload = app.open_chat(str(data.get("id") or data.get("session_id") or ""))
                code = 200 if payload.get("ok") else 404
                status, body, ctype = _json_bytes(payload, code)
                self._send(status, body, ctype)
                return
            if path == "/api/mode":
                payload = app.set_mode(str(data.get("mode") or ""))
                code = 200 if payload.get("ok") else 400
                status, body, ctype = _json_bytes(payload, code)
                self._send(status, body, ctype)
                return
            if path == "/api/soi":
                status, body, ctype = _json_bytes(app.run_soi())
                self._send(status, body, ctype)
                return
            if path == "/api/open-chrome":
                raw_urls = data.get("urls")
                urls = [str(u) for u in raw_urls] if isinstance(raw_urls, list) else None
                status, body, ctype = _json_bytes(
                    app.open_chrome_url(str(data.get("url") or ""), urls=urls)
                )
                self._send(status, body, ctype)
                return
            if path == "/api/spotify/credentials":
                from ainet.tools import spotify as spotify_mod

                payload = spotify_mod.save_app_config(
                    client_id=str(data.get("client_id") or ""),
                    client_secret=str(data.get("client_secret") or ""),
                    redirect_uri=str(data.get("redirect_uri") or ""),
                )
                code = 200 if payload.get("ok") else 400
                status, body, ctype = _json_bytes(payload, code)
                self._send(status, body, ctype)
                return
            self._send(404, b'{"ok":false,"error":"not found"}', "application/json")

    return Handler


def serve(
    host: str = "0.0.0.0",
    port: int = 1111,
    config: OllamaConfig | None = None,
    mode_id: str = DEFAULT_MODE_ID,
) -> None:
    import os

    config = config or OllamaConfig.from_env()
    app = ChatApp(config, mode_id=mode_id)
    handler = make_handler(app)
    httpd = ThreadingHTTPServer((host, port), handler)
    public = (os.environ.get("AINET_PUBLIC_URL") or "").strip().rstrip("/")
    print(
        f"AINet web  http://{host}:{port}/  "
        f"(LAN: use this machine's IP, e.g. http://192.168.x.x:{port}/)\n"
        f"model={config.model}  mode={mode_id}  db={config.db_root}  "
        f"soi_file={config.soi_idle_seconds:.0f}s",
        flush=True,
    )
    if public:
        print(f"Cloudflare public URL: {public}/  (same API as local)", flush=True)
    else:
        print(
            "Cloudflare: set Public Hostname → http://127.0.0.1:1111 "
            "(optional AINET_PUBLIC_URL=https://… for status)",
            flush=True,
        )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down…", flush=True)
    finally:
        app.shutdown()
        httpd.server_close()


def build_config(
    *,
    host: str | None = None,
    model: str | None = None,
    db: Path | None = None,
    soi_enabled: bool | None = None,
) -> OllamaConfig:
    config = OllamaConfig.from_env()
    updates: dict[str, Any] = {}
    if host:
        updates["host"] = host.rstrip("/")
    if model:
        updates["model"] = model
    if db:
        updates["db_root"] = db
    if soi_enabled is not None:
        updates["soi_enabled"] = soi_enabled
    return replace(config, **updates) if updates else config
