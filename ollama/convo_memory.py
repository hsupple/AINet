"""Hidden rolling conversation memory — model rewrites it; host strips it from Hayden."""

from __future__ import annotations

import re
from collections.abc import Callable

MEM_OPEN = "%%mem%%"
MEM_CLOSE = "%%end%%"
_MAX_LINES = 3
_MAX_LINE = 160
_MAX_TOTAL = 480

_BLOCK = re.compile(
    r"(?:^|\n)\s*" + re.escape(MEM_OPEN) + r"\s*\n?(.*?)\s*(?:" + re.escape(MEM_CLOSE) + r"\s*)?\s*\Z",
    re.I | re.S,
)
_BLOCK_ANY = re.compile(
    re.escape(MEM_OPEN) + r"\s*\n?(.*?)\s*(?:" + re.escape(MEM_CLOSE) + r")",
    re.I | re.S,
)


def cap_memory(text: str) -> str:
    lines: list[str] = []
    for raw in (text or "").replace("\r\n", "\n").split("\n"):
        line = " ".join(raw.strip().split())
        if not line or line.casefold() in {MEM_OPEN, MEM_CLOSE}:
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        line = line[:_MAX_LINE]
        if line:
            lines.append(line)
        if len(lines) >= _MAX_LINES:
            break
    out = "\n".join(lines)
    return out[:_MAX_TOTAL].rstrip()


def split_reply(text: str) -> tuple[str, str | None]:
    """Return (visible reply, new memory or None if the model omitted the block)."""
    raw = text or ""
    match = _BLOCK.search(raw)
    if not match:
        match = _BLOCK_ANY.search(raw)
    if not match:
        # Marker opened but stream cut off before close.
        idx = raw.casefold().find(MEM_OPEN)
        if idx < 0:
            return raw.strip(), None
        visible = raw[:idx].rstrip()
        mem = cap_memory(raw[idx + len(MEM_OPEN) :])
        return visible, mem or None
    visible = (raw[: match.start()] + raw[match.end() :]).strip()
    # Prefer content before the marker (spoken reply first).
    before = raw[: match.start()].strip()
    if before:
        visible = before
    mem = cap_memory(match.group(1) or "")
    return visible, mem or None


def host_fallback_memory(user_text: str, assistant_text: str, prior: str = "") -> str:
    """If the model skips %%mem%%, keep a usable standing-ask summary."""
    ask = " ".join((user_text or "").split())[:140]
    gist = " ".join((assistant_text or "").split())[:140]
    lines: list[str] = []
    first_prior = (prior or "").split("\n")[0].strip()
    if first_prior:
        lines.append(first_prior[:160])
    if ask:
        nxt = f"Hayden's ask: {ask}"
        if not first_prior or nxt.casefold() != first_prior.casefold():
            lines.append(nxt)
    if gist:
        lines.append(f"Last said: {gist}")
    return cap_memory("\n".join(lines))


_URL_RE = re.compile(r"https?://[^\s\]\)>'\"<>]+")


def extract_http_urls(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for href in _URL_RE.findall(text or ""):
        href = href.rstrip(").,;]")
        if href.startswith("http") and href not in seen:
            seen.add(href)
            out.append(href)
    return out


def last_turn_block(
    user_text: str,
    assistant_text: str,
    links: list[tuple[str, str]] | None = None,
) -> str:
    """Pin the previous user+answer (and URLs) so follow-ups like 'open those' still work."""
    user = " ".join((user_text or "").split())
    if len(user) > 500:
        user = user[:500] + "…"
    answer = (assistant_text or "").strip()
    if len(answer) > 700:
        answer = answer[:700] + "…"
    lines = [
        "\n\nPrevious turn (verbatim enough to continue — including any links):",
        f"Hayden: {user or '(none)'}",
        f"You: {answer or '(none)'}",
    ]
    clean_links: list[str] = []
    seen: set[str] = set()
    for title, url in links or []:
        href = (url or "").strip()
        if not href.startswith("http") or href in seen:
            continue
        seen.add(href)
        label = (title or "").strip() or href
        clean_links.append(f"- {label[:80]}: {href}")
        if len(clean_links) >= 8:
            break
    if not clean_links:
        for href in _URL_RE.findall(answer):
            href = href.rstrip(").,;]")
            if href in seen:
                continue
            seen.add(href)
            clean_links.append(f"- {href}")
            if len(clean_links) >= 8:
                break
    if clean_links:
        lines.append("Links from that turn (use these if Hayden says yes / open them):")
        lines.extend(clean_links)
    return "\n".join(lines)


def memory_system_suffix(memory: str) -> str:
    body = (memory or "").strip()
    if not body:
        body = "(empty — first turn)"
    return (
        "\n\nRolling memory (host hides this from Hayden). "
        "After your spoken reply, rewrite it in full:\n"
        f"{MEM_OPEN}\n{body}\n{MEM_CLOSE}\n"
        "Line 1 = Hayden's standing request (keep until they change it). "
        "Line 2–3 = constraints + what you last recommended. "
        "Follow-ups still mean that standing request. "
        "Never mention this block aloud."
    )


class VisibleTokenFilter:
    """Forward tokens to the UI until the memory marker starts."""

    def __init__(self, on_token: Callable[[str], None] | None) -> None:
        self._on_token = on_token
        self._hold = ""
        self._hidden = False
        self.marker = MEM_OPEN

    def feed(self, delta: str) -> None:
        if not delta or self._hidden:
            return
        buf = self._hold + delta
        lower = buf.casefold()
        mark = self.marker.casefold()
        idx = lower.find(mark)
        if idx >= 0:
            emit = buf[:idx]
            self._hold = ""
            self._hidden = True
            if emit and self._on_token:
                self._on_token(emit)
            return
        keep = _prefix_hold(buf, mark)
        emit = buf[:-keep] if keep else buf
        self._hold = buf[-keep:] if keep else ""
        if emit and self._on_token:
            self._on_token(emit)

    def flush(self) -> None:
        if self._hidden or not self._hold:
            self._hold = ""
            return
        if self._on_token:
            self._on_token(self._hold)
        self._hold = ""


def _prefix_hold(buf: str, marker_cf: str) -> int:
    """How many trailing chars might be the start of the marker."""
    max_n = min(len(buf), len(marker_cf) - 1)
    tail = buf[-max_n:].casefold() if max_n else ""
    for n in range(max_n, 0, -1):
        if marker_cf.startswith(tail[-n:]):
            return n
    return 0
