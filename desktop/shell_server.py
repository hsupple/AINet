"""Local HTTP shell for the AINet desktop app."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

_REPO = Path(__file__).resolve().parent.parent
_STATIC = Path(__file__).resolve().parent / "static"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ainet.tools.ops import DatabaseTools
from ollama.config import OllamaConfig

SHELL_PORT = 1100
CHAT_PORT = 1111
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_DIR = Path(r"D:\AINet-Tools\Ollama")
OLLAMA_MODELS = Path(r"D:\AINet-Tools\ollama-models")


def _json_bytes(payload: dict[str, Any], status: int = 200) -> tuple[int, bytes, str]:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    return status, body, "application/json; charset=utf-8"


def _probe(url: str, timeout: float = 1.5) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            try:
                data = json.loads(text) if text.strip() else None
            except json.JSONDecodeError:
                data = text
            return {"ok": True, "status": getattr(resp, "status", 200), "url": url, "data": data}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "url": url}


def _force_qwen() -> None:
    os.environ["AINET_OLLAMA_MODEL"] = "qwen3:8b"
    if OLLAMA_DIR.is_dir():
        os.environ["PATH"] = str(OLLAMA_DIR) + os.pathsep + os.environ.get("PATH", "")
    if OLLAMA_MODELS.is_dir() and not os.environ.get("OLLAMA_MODELS"):
        os.environ["OLLAMA_MODELS"] = str(OLLAMA_MODELS)


def _get_json(url: str, timeout: float = 3.0) -> dict[str, Any]:
    probe = _probe(url, timeout=timeout)
    if not probe.get("ok"):
        return probe
    data = probe.get("data")
    return data if isinstance(data, dict) else {"ok": True, "data": data}


class ShellApp:
    def __init__(self, config: OllamaConfig | None = None) -> None:
        _force_qwen()
        self.config = config or OllamaConfig.from_env()
        if self.config.model != "qwen3:8b":
            from dataclasses import replace as dc_replace

            self.config = dc_replace(self.config, model="qwen3:8b")
        self.db = DatabaseTools(self.config.db_root)
        self._procs: dict[str, Any] = {}
        self._lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        chat = _probe(f"http://127.0.0.1:{CHAT_PORT}/api/status")
        ollama = _probe(f"{OLLAMA_HOST}/api/tags")
        chat_model = None
        if isinstance(chat.get("data"), dict):
            chat_model = chat["data"].get("model")
        return {
            "ok": True,
            "db_root": str(self.config.db_root),
            "model": self.config.model,
            "chat_model": chat_model,
            "services": {
                "chat": {"ok": bool(chat.get("ok")), "port": CHAT_PORT, "error": chat.get("error")},
                "ollama": {"ok": bool(ollama.get("ok")), "host": OLLAMA_HOST, "error": ollama.get("error")},
            },
        }

    def tree(self, path: str = ".", max_depth: int = 4) -> dict[str, Any]:
        try:
            return self.db.tree(path, max_depth=max_depth)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def read_file(self, path: str) -> dict[str, Any]:
        try:
            if path.endswith(".json"):
                return self.db.read_json(path)
            return self.db.read_text(path)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def soi_events(self, after: int = 0, limit: int = 200) -> dict[str, Any]:
        """Live SOI activity from the chat server, with disk log fallback."""
        live = _get_json(f"http://127.0.0.1:{CHAT_PORT}/api/soi-log?after={after}")
        rows = live.get("events") if isinstance(live.get("events"), list) else live.get("lines")
        if live.get("ok") and isinstance(rows, list):
            return {
                "ok": True,
                "source": "chat",
                "events": rows,
                "event_seq": live.get("event_seq") or live.get("soi_log_seq"),
                "soi_running": live.get("soi_running"),
            }

        path = Path(self.config.db_root) / "runtime" / "soi" / "events.jsonl"
        events: list[dict[str, Any]] = []
        if path.is_file():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
                for i, line in enumerate(lines[-limit:], start=max(0, len(lines) - limit)):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        row = {**row, "id": i + 1}
                        if int(row.get("id") or 0) > after:
                            events.append(row)
            except OSError as exc:
                return {"ok": False, "error": str(exc), "events": []}
        return {"ok": True, "source": "disk", "events": events}

    def ollama_raw(self, after: int = 0) -> dict[str, Any]:
        tags = _probe(f"{OLLAMA_HOST}/api/tags", timeout=3)
        ps = _probe(f"{OLLAMA_HOST}/api/ps", timeout=3)
        live = _get_json(f"http://127.0.0.1:{CHAT_PORT}/api/ollama-raw?after={after}")
        lines = live.get("lines") if isinstance(live.get("lines"), list) else []
        chat = _get_json(f"http://127.0.0.1:{CHAT_PORT}/api/status")
        return {
            "ok": True,
            "model": self.config.model,
            "tags": tags.get("data") if tags.get("ok") else {"error": tags.get("error")},
            "ps": ps.get("data") if ps.get("ok") else {"error": ps.get("error")},
            "lines": lines,
            "seq": live.get("seq") if isinstance(live, dict) else 0,
            "chat_ok": bool(isinstance(chat, dict) and chat.get("ok")),
            "chat_model": chat.get("model") if isinstance(chat, dict) else None,
            "soi_running": chat.get("soi_running") if isinstance(chat, dict) else None,
        }

    def start_ollama(self) -> dict[str, Any]:
        if _probe(f"{OLLAMA_HOST}/api/tags").get("ok"):
            return {"ok": True, "already": True}

        candidates = [
            OLLAMA_DIR / "ollama app.exe",
            OLLAMA_DIR / "ollama.exe",
        ]
        for path in candidates:
            if not path.is_file():
                continue
            try:
                subprocess.Popen(
                    [str(path)],
                    cwd=str(path.parent),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}
            for _ in range(20):
                time.sleep(0.5)
                if _probe(f"{OLLAMA_HOST}/api/tags").get("ok"):
                    return {"ok": True, "started": True, "path": str(path)}
            return {"ok": True, "started": True, "waiting": True, "path": str(path)}
        return {"ok": False, "error": f"Ollama not found under {OLLAMA_DIR}"}

    def start_chat(self) -> dict[str, Any]:
        probe = _probe(f"http://127.0.0.1:{CHAT_PORT}/api/status")
        if probe.get("ok"):
            data = probe.get("data") if isinstance(probe.get("data"), dict) else {}
            model = str(data.get("model") or "")
            raw_ok = _probe(f"http://127.0.0.1:{CHAT_PORT}/api/ollama-raw").get("ok")
            wrong_model = bool(model and "llama" in model.lower())
            if not wrong_model and raw_ok:
                return {
                    "ok": True,
                    "already": True,
                    "port": CHAT_PORT,
                    "model": model or self.config.model,
                }
            self._kill_chat()

        if getattr(sys, "frozen", False):
            return self._start_chat_inprocess()

        env = {**os.environ, "AINET_OLLAMA_MODEL": "qwen3:8b"}
        if OLLAMA_DIR.is_dir():
            env["PATH"] = str(OLLAMA_DIR) + os.pathsep + env.get("PATH", "")
        if OLLAMA_MODELS.is_dir():
            env["OLLAMA_MODELS"] = str(OLLAMA_MODELS)
        cmd = [
            sys.executable,
            "-m",
            "ollama",
            "web",
            "--bind",
            "127.0.0.1",
            "--port",
            str(CHAT_PORT),
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(_REPO),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        self._procs["chat"] = proc
        return {"ok": True, "started": True, "pid": proc.pid}

    def _kill_chat(self) -> None:
        existing = self._procs.get("chat")
        if existing and hasattr(existing, "poll") and existing.poll() is None:
            try:
                existing.kill()
                existing.wait(timeout=3)
            except Exception:
                pass
            self._procs.pop("chat", None)
            time.sleep(0.4)
            return
        # Windows: stop whatever owns :1111 (stale python -m ollama web).
        if os.name == "nt":
            try:
                subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        (
                            "$p = Get-NetTCPConnection -LocalPort 1111 -State Listen "
                            "-ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique; "
                            "if ($p) { $p | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue } }"
                        ),
                    ],
                    capture_output=True,
                    timeout=15,
                )
                time.sleep(0.6)
            except Exception:
                pass

    def _start_chat_inprocess(self) -> dict[str, Any]:
        key = "thread:chat"
        if key in self._procs:
            return {"ok": True, "already": True}

        def _run() -> None:
            from ollama.webserver import serve as serve_chat

            serve_chat(host="127.0.0.1", port=CHAT_PORT, config=self.config)

        thread = threading.Thread(target=_run, name="ainet-chat", daemon=True)
        thread.start()
        self._procs[key] = thread
        return {"ok": True, "started": True, "inprocess": True}

    def ensure_backends(self) -> dict[str, Any]:
        _force_qwen()
        ollama = self.start_ollama()
        # Wait briefly for Ollama before starting chat.
        for _ in range(15):
            if _probe(f"{OLLAMA_HOST}/api/tags").get("ok"):
                break
            time.sleep(0.4)
        chat = self.start_chat()
        return {"ok": True, "ollama": ollama, "chat": chat}


def make_handler(app: ShellApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AINetDesktop/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[shell] {fmt % args}", flush=True)

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

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return {}
            return data if isinstance(data, dict) else {}

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            if path in {"/", "/index.html"}:
                html = _STATIC / "shell.html"
                self._send(200, html.read_bytes(), "text/html; charset=utf-8")
                return

            if path == "/api/status":
                status, body, ctype = _json_bytes(app.status())
                self._send(status, body, ctype)
                return

            if path == "/api/tree":
                rel = (qs.get("path") or ["."])[0]
                try:
                    depth = int((qs.get("depth") or ["5"])[0])
                except ValueError:
                    depth = 5
                status, body, ctype = _json_bytes(app.tree(rel, max_depth=depth))
                self._send(status, body, ctype)
                return

            if path == "/api/file":
                rel = (qs.get("path") or [""])[0]
                payload = app.read_file(rel) if rel else {"ok": False, "error": "path required"}
                status, body, ctype = _json_bytes(payload, 200 if payload.get("ok", True) else 400)
                self._send(status, body, ctype)
                return

            if path == "/api/soi/events":
                try:
                    after = int((qs.get("after") or ["0"])[0] or 0)
                except ValueError:
                    after = 0
                status, body, ctype = _json_bytes(app.soi_events(after=after))
                self._send(status, body, ctype)
                return

            if path == "/api/ollama/raw":
                try:
                    after = int((qs.get("after") or ["0"])[0] or 0)
                except ValueError:
                    after = 0
                status, body, ctype = _json_bytes(app.ollama_raw(after=after))
                self._send(status, body, ctype)
                return

            self._send(404, b'{"ok":false,"error":"not found"}', "application/json")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            _ = self._read_json()

            if path == "/api/boot":
                status, body, ctype = _json_bytes(app.ensure_backends())
                self._send(status, body, ctype)
                return

            self._send(404, b'{"ok":false,"error":"not found"}', "application/json")

    return Handler


def serve(host: str = "127.0.0.1", port: int = SHELL_PORT, *, ensure_backends: bool = True) -> None:
    app = ShellApp()
    if ensure_backends:
        threading.Thread(target=app.ensure_backends, name="ainet-boot", daemon=True).start()
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"AINet desktop  http://{host}:{port}/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down…", flush=True)
    finally:
        httpd.server_close()
