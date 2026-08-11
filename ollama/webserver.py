"""LAN web chat for AINet OAC (stdlib only).

Binds to all interfaces by default so you can open http://<your-ip>:1111
from this PC or another device on the network.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ollama.client import OllamaError
from ollama.config import OllamaConfig
from ollama.idle import IdleSOIWatcher
from ollama.modes import DEFAULT_MODE_ID, get_mode, list_modes
from ollama.session import ChatSession

_STATIC = Path(__file__).resolve().parent / "static"


class ChatApp:
    def __init__(self, config: OllamaConfig, mode_id: str = DEFAULT_MODE_ID) -> None:
        self.config = config
        self.lock = threading.RLock()
        self.soi_log: deque[str] = deque(maxlen=20)
        self.session = ChatSession(
            mode=get_mode(mode_id),
            config=config,
            auto_mode=config.auto_mode,
        )
        self.watcher = IdleSOIWatcher(
            self.session,
            config,
            on_status=lambda msg: self.soi_log.append(msg),
        )
        if config.soi_enabled:
            self.watcher.start()

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "ok": True,
                "model": self.config.model,
                "mode": self.session.mode.id,
                "auto_mode": self.session.auto_mode,
                "mode_locked": self.session.mode_locked,
                "session_id": self.session.session_id,
                "topic": self.session.topic["title"] if self.session.topic else None,
                "db_root": str(self.config.db_root),
                "soi_enabled": self.config.soi_enabled,
                "soi_idle_seconds": self.config.soi_idle_seconds,
                "soi_running": self.watcher.running,
                "modes": [
                    {"id": m.id, "name": m.name, "description": m.description}
                    for m in list_modes()
                    if m.role == "oac"
                ],
                "soi_log": list(self.soi_log),
            }

    def ask(self, text: str) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "Empty message"}
        with self.lock:
            try:
                reply = self.session.ask(text)
            except OllamaError as exc:
                return {"ok": False, "error": str(exc)}
            return {
                "ok": True,
                "reply": reply,
                "mode": self.session.mode.id,
                "session_id": self.session.session_id,
                "topic": self.session.topic["title"] if self.session.topic else None,
                "soi_running": self.watcher.running,
            }

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.session.reset()
            return {"ok": True, **self.status()}

    def set_mode(self, mode_id: str) -> dict[str, Any]:
        with self.lock:
            try:
                mode = self.session.set_mode(mode_id, lock=True)
            except KeyError as exc:
                return {"ok": False, "error": str(exc)}
            if mode.role != "oac":
                self.session.set_mode(DEFAULT_MODE_ID, lock=False)
                return {"ok": False, "error": "Web chat only supports OAC modes"}
            return {"ok": True, **self.status()}

    def bind_topic(self, title: str) -> dict[str, Any]:
        title = (title or "").strip()
        if not title:
            return {"ok": False, "error": "Topic title required"}
        with self.lock:
            info = self.session.bind_topic(title)
            return {"ok": True, "topic": info, **self.status()}

    def run_soi(self) -> dict[str, Any]:
        kicked = self.watcher.request_run()
        with self.lock:
            return {
                "ok": True,
                "started": bool(kicked.get("started")),
                "reason": kicked.get("reason") or "",
                **self.status(),
            }

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
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
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
            self._send(404, b'{"ok":false,"error":"not found"}', "application/json")

        def do_POST(self) -> None:  # noqa: N802
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
            if path == "/api/reset":
                status, body, ctype = _json_bytes(app.reset())
                self._send(status, body, ctype)
                return
            if path == "/api/mode":
                payload = app.set_mode(str(data.get("mode") or ""))
                code = 200 if payload.get("ok") else 400
                status, body, ctype = _json_bytes(payload, code)
                self._send(status, body, ctype)
                return
            if path == "/api/topic":
                payload = app.bind_topic(str(data.get("topic") or data.get("title") or ""))
                code = 200 if payload.get("ok") else 400
                status, body, ctype = _json_bytes(payload, code)
                self._send(status, body, ctype)
                return
            if path == "/api/soi":
                status, body, ctype = _json_bytes(app.run_soi())
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
    config = config or OllamaConfig.from_env()
    app = ChatApp(config, mode_id=mode_id)
    handler = make_handler(app)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(
        f"AINet web  http://{host}:{port}/  "
        f"(LAN: use this machine's IP, e.g. http://192.168.x.x:{port}/)\n"
        f"model={config.model}  mode={mode_id}  db={config.db_root}  "
        f"soi_file={config.soi_idle_seconds:.0f}s",
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
