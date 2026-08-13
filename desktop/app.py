"""Launch the AINet desktop shell window (and backends).

Usage:
  python -m desktop.app
  python desktop/app.py
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from desktop.shell_server import SHELL_PORT, ShellApp, make_handler
from http.server import ThreadingHTTPServer


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def _start_shell_server(port: int, ensure_backends: bool) -> ThreadingHTTPServer:
    app = ShellApp()
    if ensure_backends:
        threading.Thread(target=app.ensure_backends, name="ainet-backends", daemon=True).start()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    threading.Thread(target=httpd.serve_forever, name="ainet-shell", daemon=True).start()
    return httpd


def _open_window(url: str) -> None:
    try:
        import webview  # type: ignore

        window = webview.create_window(
            "AINet",
            url,
            width=1280,
            height=840,
            min_size=(960, 640),
            background_color="#ffffff",
        )
        webview.start()
        _ = window
        return
    except Exception:
        pass

    # Fallback: system browser (still usable as a desktop entrypoint)
    webbrowser.open(url)
    print(f"Opened {url} in your browser (install pywebview for a native window).", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AINet desktop shell")
    parser.add_argument("--port", type=int, default=SHELL_PORT)
    parser.add_argument("--no-backends", action="store_true", help="Do not auto-start Ollama/chat")
    parser.add_argument("--browser-only", action="store_true", help="Force system browser instead of pywebview")
    args = parser.parse_args(argv)

    # Pin model before any backend config is loaded.
    import os

    os.environ["AINET_OLLAMA_MODEL"] = "qwen3:8b"

    if _port_open(args.port):
        print(f"Shell already running on :{args.port}", flush=True)
        httpd = None
    else:
        httpd = _start_shell_server(args.port, ensure_backends=not args.no_backends)
        # Give backends a moment before the UI iframe hits them
        time.sleep(0.8)
        print(f"AINet desktop shell  http://127.0.0.1:{args.port}/", flush=True)

    url = f"http://127.0.0.1:{args.port}/"
    try:
        if args.browser_only:
            webbrowser.open(url)
            print("Press Ctrl+C to stop.", flush=True)
            while True:
                time.sleep(1)
        else:
            _open_window(url)
    finally:
        if httpd is not None:
            httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
