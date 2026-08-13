"""Open URLs in Google Chrome on the host machine (Windows-first)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


_CHROME_CANDIDATES = (
    Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Google" / "Chrome" / "Application" / "chrome.exe",
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Google" / "Chrome" / "Application" / "chrome.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
)


def _normalize_url(url: str) -> str:
    text = (url or "").strip()
    if not text:
        raise ValueError("url is required")
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http(s) URLs can be opened in Chrome")
    if not parsed.netloc:
        raise ValueError("URL must include a host")
    return text


def _find_chrome() -> str | None:
    for path in _CHROME_CANDIDATES:
        if path.is_file():
            return str(path)
    which = shutil.which("chrome") or shutil.which("google-chrome") or shutil.which("chromium")
    return which


def open_chrome(url: str, *, new_tab: bool = True) -> dict[str, Any]:
    """Open an http(s) URL in Google Chrome. Returns a JSON-serializable result."""
    target = _normalize_url(url)
    chrome = _find_chrome()
    if not chrome:
        raise ValueError("Google Chrome not found on this machine")

    args = [chrome]
    if new_tab:
        args.append("--new-tab")
    args.append(target)

    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )

    subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
    )
    return {
        "ok": True,
        "opened": True,
        "browser": "chrome",
        "url": target,
        "chrome": chrome,
    }
