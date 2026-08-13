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

# Windows: break out of the parent job object so Chrome is not deferred until Python exits.
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_DETACHED_PROCESS = 0x00000008
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000
_CREATE_NO_WINDOW = 0x08000000


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

    if sys.platform == "win32":
        # `cmd /c start` returns immediately and detaches Chrome from this process tree.
        # Direct chrome.exe Popen often stays tied to the Python job, so tabs only appear
        # when the chat server exits.
        start_args = ["cmd", "/c", "start", "", chrome]
        if new_tab:
            start_args.append("--new-tab")
        start_args.append(target)
        subprocess.Popen(
            start_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=_CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP | _CREATE_BREAKAWAY_FROM_JOB,
            close_fds=False,
        )
    else:
        args = [chrome]
        if new_tab:
            args.append("--new-tab")
        args.append(target)
        subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    return {
        "ok": True,
        "opened": True,
        "browser": "chrome",
        "url": target,
        "chrome": chrome,
    }
