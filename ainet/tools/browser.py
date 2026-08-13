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


def open_chrome(
    url: str = "",
    *,
    urls: list[str] | None = None,
    new_tab: bool = True,
) -> dict[str, Any]:
    """Open one or more http(s) URLs in Google Chrome."""
    targets: list[str] = []
    if urls:
        for item in urls:
            text = str(item or "").strip()
            if text:
                targets.append(_normalize_url(text))
    single = (url or "").strip()
    if single:
        targets.insert(0, _normalize_url(single))
    # Dedupe while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for t in targets:
        if t in seen:
            continue
        seen.add(t)
        ordered.append(t)
    if not ordered:
        raise ValueError("url or urls is required")
    if len(ordered) > 8:
        ordered = ordered[:8]

    chrome = _find_chrome()
    if not chrome:
        raise ValueError("Google Chrome not found on this machine")

    opened: list[str] = []
    for target in ordered:
        if sys.platform == "win32":
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
        opened.append(target)

    return {
        "ok": True,
        "opened": True,
        "browser": "chrome",
        "url": opened[0],
        "urls": opened,
        "count": len(opened),
        "chrome": chrome,
    }

