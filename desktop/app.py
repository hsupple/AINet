"""Launch the AINet desktop shell window (and backends).

Usage:
  python -m desktop.app
  python desktop/app.py
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import threading
import time
import urllib.request
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


def _http_ok(url: str, timeout: float = 1.2) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= int(resp.status) < 500
    except Exception:
        return False


def _kill_listeners(port: int) -> None:
    """Drop a wedged listener so we can bind again (Windows)."""
    if sys.platform != "win32":
        return
    try:
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    f"$p = Get-NetTCPConnection -LocalPort {int(port)} -State Listen "
                    "-ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique; "
                    "if ($p) { $p | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue } }"
                ),
            ],
            capture_output=True,
            timeout=15,
        )
    except Exception:
        pass
    time.sleep(0.5)


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

    url = f"http://127.0.0.1:{args.port}/"
    if _port_open(args.port):
        if _http_ok(url):
            print(f"Shell already running on :{args.port}", flush=True)
            httpd = None
        else:
            print(
                f"Shell on :{args.port} is not responding — restarting it.",
                flush=True,
            )
            _kill_listeners(args.port)
            httpd = _start_shell_server(args.port, ensure_backends=not args.no_backends)
            time.sleep(0.8)
            print(f"AINet desktop shell  {url}", flush=True)
    else:
        httpd = _start_shell_server(args.port, ensure_backends=not args.no_backends)
        # Give backends a moment before the UI iframe hits them
        time.sleep(0.8)
        print(f"AINet desktop shell  {url}", flush=True)
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
