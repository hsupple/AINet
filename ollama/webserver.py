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

from ollama.client import OllamaError
from ollama.config import OllamaConfig
from ollama.idle import IdleSOIWatcher
from ollama.modes import DEFAULT_MODE_ID, get_mode, list_modes
from ollama.session import ChatSession
from ollama.tts import SpeechPipeline, TtsClient, TtsConfig

_STATIC = Path(__file__).resolve().parent / "static"


class ChatApp:
    def __init__(self, config: OllamaConfig, mode_id: str = DEFAULT_MODE_ID) -> None:
        self.config = config
        self.lock = threading.RLock()
        self._soi_seq = 0
        self.soi_log: deque[dict[str, Any]] = deque(maxlen=500)
        self._raw_seq = 0
        self.raw_log: deque[dict[str, Any]] = deque(maxlen=300)
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

    def _tts_client(self) -> TtsClient:
        return TtsClient(
            TtsConfig(
                url=self.config.tts_url,
                enabled=self.config.tts_enabled,
                language=self.config.tts_language,
                timeout_s=self.config.tts_timeout_s,
            )
        )

    def tts_status(self) -> dict[str, Any]:
        client = self._tts_client()
        healthy = False
        if self.config.tts_enabled:
            # Short probe; cache briefly so /api/status stays snappy.
            now = time.monotonic()
            cached = getattr(self, "_tts_health_cache", None)
            if cached and now - cached[0] < 5.0:
                healthy = cached[1]
            else:
                healthy = client.healthy()
                self._tts_health_cache = (now, healthy)
        return {
            "tts_enabled": self.config.tts_enabled,
            "tts_url": self.config.tts_url,
            "tts_language": self.config.tts_language,
            "tts_healthy": healthy,
        }

    def status(self) -> dict[str, Any]:
        with self.lock:
            base = {
                "ok": True,
                "model": self.config.model,
                "mode": self.session.mode.id,
                "auto_mode": self.session.auto_mode,
                "mode_locked": self.session.mode_locked,
                "session_id": self.session.session_id,
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
                "soi_log_seq": self._soi_seq,
                "raw_log_seq": self._raw_seq,
                "last_raw": list(self.raw_log)[-1] if self.raw_log else None,
            }
        base.update(self.tts_status())
        return base

    def ask(self, text: str) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "Empty message"}
        with self.lock:
            try:
                reply = self.session.ask(text)
            except OllamaError as exc:
                return {"ok": False, "error": str(exc)}
            self._append_raw(source="oac", user=text, reply=reply, model=self.config.model)
            return {
                "ok": True,
                "reply": reply,
                "mode": self.session.mode.id,
                "session_id": self.session.session_id,
                "soi_running": self.watcher.running,
                **self.tts_status(),
            }

    def ask_stream(self, text: str, *, voice: bool = True) -> Iterator[dict[str, Any]]:
        """Yield SSE payloads: token / audio / tts_error / done / error."""
        text = (text or "").strip()
        if not text:
            yield {"type": "error", "error": "Empty message"}
            return

        events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        want_voice = bool(voice) and self.config.tts_enabled
        client = self._tts_client()
        pipe: SpeechPipeline | None = None

        if want_voice:
            if not client.healthy():
                events.put(
                    {
                        "type": "tts_error",
                        "error": f"TTS server not reachable at {self.config.tts_url}",
                    }
                )
                want_voice = False
            else:
                def _on_audio(b64: str, sr: int, seq: int, spoken: str) -> None:
                    events.put(
                        {
                            "type": "audio",
                            "audio_base64": b64,
                            "sample_rate": sr,
                            "seq": seq,
                            "text": spoken,
                            "mime": "audio/wav",
                        }
                    )

                def _on_tts_err(err: str) -> None:
                    events.put({"type": "tts_error", "error": err})

                pipe = SpeechPipeline(
                    client,
                    on_audio_b64=_on_audio,
                    on_error=_on_tts_err,
                    enabled=True,
                )

        def _on_token(delta: str) -> None:
            if not delta:
                return
            events.put({"type": "token", "text": delta})
            if pipe is not None:
                pipe.feed(delta)

        def _run() -> None:
            reply = ""
            try:
                with self.lock:
                    reply = self.session.ask(text, stream=True, on_token=_on_token)
                    self._append_raw(
                        source="oac",
                        user=text,
                        reply=reply,
                        model=self.config.model,
                    )
                    done_payload = {
                        "type": "done",
                        "ok": True,
                        "reply": reply,
                        "mode": self.session.mode.id,
                        "session_id": self.session.session_id,
                        "soi_running": self.watcher.running,
                        "voice": bool(pipe is not None),
                        **self.tts_status(),
                    }
                # Unlock the UI as soon as the model finishes — don't wait on TTS.
                events.put(done_payload)
            except OllamaError as exc:
                events.put({"type": "error", "error": str(exc)})
            except Exception as exc:
                events.put({"type": "error", "error": str(exc)})
            finally:
                if pipe is not None:
                    try:
                        # Keep this short so a hung TTS server cannot pin the stream open.
                        pipe.close(timeout=20.0)
                    except Exception:
                        pass
                events.put(None)

        threading.Thread(target=_run, name="ainet-chat-stream", daemon=True).start()
        while True:
            item = events.get()
            if item is None:
                break
            yield item

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
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                try:
                    while True:
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
                        time.sleep(0.35)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
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
            if path == "/api/chat-stream":
                voice = data.get("voice")
                if voice is None:
                    voice = True
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                try:
                    for event in app.ask_stream(
                        str(data.get("message") or ""),
                        voice=bool(voice),
                    ):
                        blob = json.dumps(event, ensure_ascii=False)
                        self.wfile.write(f"data: {blob}\n\n".encode("utf-8"))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    return
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
        f"soi_file={config.soi_idle_seconds:.0f}s  "
        f"tts={'on' if config.tts_enabled else 'off'}@{config.tts_url}",
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
