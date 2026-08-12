"""SOI test harness web UI — visualize pending changelog, tool calls, and in-memory diffs."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ollama.config import OllamaConfig
from ollama.soi_test_app import SOITestApp

_STATIC = Path(__file__).resolve().parent / "static"


def _json_bytes(payload: dict[str, Any], status: int = 200) -> tuple[int, bytes, str]:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    return status, body, "application/json; charset=utf-8"


def make_handler(app: SOITestApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AINetSOITest/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} {fmt % args}", file=sys.stderr, flush=True)

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
                html_path = _STATIC / "soi_test.html"
                if html_path.is_file():
                    self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
                    return
                self._send(404, b"soi_test.html missing", "text/plain")
                return

            if path == "/api/status":
                status, body, ctype = _json_bytes(app.status())
                self._send(status, body, ctype)
                return

            if path == "/api/pending":
                st = app.status()
                status, body, ctype = _json_bytes(
                    {"ok": True, "pending": st.get("pending") or [], "seed": st.get("seed")}
                )
                self._send(status, body, ctype)
                return

            if path == "/api/tree":
                rel = (qs.get("path") or ["."])[0]
                try:
                    depth = int((qs.get("depth") or ["3"])[0])
                except ValueError:
                    depth = 3
                status, body, ctype = _json_bytes(app.tree(rel, max_depth=depth))
                self._send(status, body, ctype)
                return

            if path == "/api/file":
                rel = (qs.get("path") or [""])[0]
                view = (qs.get("view") or ["current"])[0]
                if not rel:
                    status, body, ctype = _json_bytes({"ok": False, "error": "path required"}, 400)
                else:
                    status, body, ctype = _json_bytes(app.read_file(rel, view=view))
                self._send(status, body, ctype)
                return

            if path == "/api/changes":
                status, body, ctype = _json_bytes(app.changes_snapshot())
                self._send(status, body, ctype)
                return

            if path == "/api/tools":
                status, body, ctype = _json_bytes(app.tool_trace())
                self._send(status, body, ctype)
                return

            if path == "/api/events":
                try:
                    after = int((qs.get("after") or ["0"])[0] or 0)
                except ValueError:
                    after = 0
                payload = {
                    "ok": True,
                    "events": app.events_after(after),
                    "event_seq": app.status().get("event_seq"),
                    "running": app.status().get("running"),
                }
                status, body, ctype = _json_bytes(payload)
                self._send(status, body, ctype)
                return

            self._send(404, b'{"ok":false,"error":"not found"}', "application/json")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            data = self._read_json()

            if path == "/api/reset":
                extra = bool(data.get("extra_turns", True))
                try:
                    payload = app.reset_session(extra_turns=extra)
                    status, body, ctype = _json_bytes(payload)
                    self._send(status, body, ctype)
                except Exception as exc:
                    status, body, ctype = _json_bytes({"ok": False, "error": str(exc)}, 500)
                    self._send(status, body, ctype)
                return

            if path == "/api/run":
                status, body, ctype = _json_bytes(app.run_filing())
                code = 200 if body and b'"ok":true' in body.replace(b" ", b"") else 400
                self._send(code, body, ctype)
                return

            if path == "/api/run_p2":
                status, body, ctype = _json_bytes(app.run_read_refresh())
                code = 200 if body and b'"ok":true' in body.replace(b" ", b"") else 400
                self._send(code, body, ctype)
                return

            self._send(404, b'{"ok":false,"error":"not found"}', "application/json")

    return Handler


def serve(
    host: str = "0.0.0.0",
    port: int = 1112,
    config: OllamaConfig | None = None,
) -> None:
    config = config or OllamaConfig.from_env()
    app = SOITestApp(config)
    handler = make_handler(app)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(
        f"SOI test harness  http://{host}:{port}/\n"
        f"  source db (read-only): {config.db_root}\n"
        f"  model={config.model}  mutations stay in ephemeral sandbox — never saved to db/",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down…", flush=True)
    finally:
        app._teardown_sandbox()
        httpd.server_close()


def build_config(
    *,
    host: str | None = None,
    model: str | None = None,
    db: Path | None = None,
) -> OllamaConfig:
    config = OllamaConfig.from_env()
    updates: dict[str, Any] = {}
    if host:
        updates["host"] = host.rstrip("/")
    if model:
        updates["model"] = model
    if db:
        updates["db_root"] = db
    return replace(config, **updates) if updates else config


if __name__ == "__main__":
    serve()
