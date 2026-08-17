"""Hidden rolling conversation memory — model rewrites it; host strips it from Hayden."""

from __future__ import annotations

import re
from collections.abc import Callable

MEM_OPEN = "%%mem%%"
MEM_CLOSE = "%%end%%"
_MAX_LINES = 3
_MAX_LINE = 160
_MAX_TOTAL = 480

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
    match = _BLOCK_ANY.search(raw)
    if not match:
        # Marker opened but stream cut off before close.
        idx = raw.casefold().find(MEM_OPEN)
        if idx < 0:
            return raw.strip(), None
        visible = raw[:idx].rstrip()
        mem = cap_memory(raw[idx + len(MEM_OPEN) :])
        return visible.strip(), mem or None
    # Models put the block first about as often as last — keep speech from both sides.
    before = raw[: match.start()].strip()
    after = raw[match.end() :].strip()
    visible = "\n\n".join(part for part in (before, after) if part)
    mem = cap_memory(match.group(1) or "")
    return visible, mem or None


def host_fallback_memory(user_text: str, assistant_text: str, prior: str = "") -> str:
    """If the model skips %%mem%%, retain a labeled rolling-memory block."""
    ask = " ".join((user_text or "").split())[:140]
    gist = " ".join((assistant_text or "").split())[:140]
    fields = _memory_fields(prior)
    if _looks_like_new_question(user_text):
        standing = ask
    else:
        standing = fields.get("standing request") or ask
    context = fields.get("context") or ""
    return cap_memory(
        "\n".join(
            (
                f"Standing request: {standing}",
                f"Context: {context}",
                f"Last answer: {gist}",
            )
        )
    )


_VAGUE_SEARCH_RE = re.compile(
    r"\b(?:"
    r"search\s+(?:it\s+)?up|look\s+(?:it\s+)?up|just\s+search|go\s+ahead\s+and\s+search|"
    r"yeah[\s,]+search|search\s+for\s+(?:it|that)|find\s+(?:me\s+)?(?:a\s+)?vids?\b|"
    r"find\s+(?:me\s+)?(?:a\s+)?videos?\b|get\s+(?:me\s+)?(?:a\s+)?vids?\b|"
    r"get\s+(?:me\s+)?(?:a\s+)?videos?\b"
    r")\b",
    re.I,
)
_VIDEO_HINT_RE = re.compile(r"\b(?:vids?\b|videos?\b|youtube|youtu\.be|watch\b|clip\b|tutorial\b)\b", re.I)


def standing_request(memory: str) -> str:
    return _memory_fields(memory).get("standing request", "").strip()


def is_vague_search_followup(user_text: str) -> bool:
    raw = " ".join((user_text or "").split())
    if not raw:
        return False
    if _VAGUE_SEARCH_RE.search(raw):
        return True
    if _VIDEO_HINT_RE.search(raw) and len(raw.split()) <= 8 and not _looks_like_new_question(user_text):
        return True
    return False


def enrich_web_search_query(query: str, user_text: str, standing: str) -> tuple[str, bool]:
    """When Hayden's follow-up is vague, search the standing request instead of a bad guess."""
    topic = " ".join((standing or "").split()).strip()
    if not topic or not is_vague_search_followup(user_text):
        return " ".join((query or "").split()).strip(), False

    user = " ".join((user_text or "").split())
    if _VIDEO_HINT_RE.search(user):
        new_q = f"{topic} tutorial video"
    else:
        new_q = topic

    new_q = new_q[:200].strip()
    old_q = " ".join((query or "").split()).strip()
    return new_q, new_q.casefold() != old_q.casefold()


def _looks_like_new_question(text: str) -> bool:
    raw = " ".join((text or "").split())
    if len(raw.split()) < 3:
        return False
    low = f" {raw.casefold()} "
    starters = (
        " who ",
        " what ",
        " how ",
        " when ",
        " where ",
        " why ",
        " which ",
        "whom ",
    )
    if any(low.startswith(s.strip()) or s in low for s in starters):
        return True
    return "?" in raw


def _memory_fields(memory: str) -> dict[str, str]:
    """Read labeled memory and map legacy positional lines for old sessions."""
    lines = [line.strip() for line in (memory or "").splitlines() if line.strip()]
    fields: dict[str, str] = {}
    labels = ("standing request", "context", "last answer")
    for line in lines[:3]:
        key, sep, value = line.partition(":")
        normalized = key.strip().casefold()
        if sep and normalized in labels:
            fields[normalized] = value.strip()
    if fields:
        return fields
    for label, line in zip(labels, lines[:3]):
        fields[label] = line.removeprefix("- ").strip()
    return fields


def _labeled_memory(memory: str) -> str:
    fields = _memory_fields(memory)
    return cap_memory(
        "\n".join(
            (
                f"Standing request: {fields.get('standing request', '')}",
                f"Context: {fields.get('context', '')}",
                f"Last answer: {fields.get('last answer', '')}",
            )
        )
    )


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
    body = _labeled_memory(memory)
    return (
        "\n\nROLLING MEMORY STATE\n"
        "This is hidden host context, not spoken text:\n"
        f"{MEM_OPEN}\n{body}\n{MEM_CLOSE}\n"
        "Answer Hayden in plain text first, then replace this block at the very end "
        "with exactly three concise labeled lines: "
        "Standing request, Context, and Last answer. "
        "Keep the standing request until Hayden changes it. "
        "If Hayden asks a new clear question, replace Standing request with that question. "
        "End with %%end%%. Never mention this block aloud."
    )


class VisibleTokenFilter:
    """Forward tokens to the UI, skipping the hidden memory block."""

    def __init__(self, on_token: Callable[[str], None] | None) -> None:
        self._on_token = on_token
        self._hold = ""
        self._hidden = False
        self.marker = MEM_OPEN

    def feed(self, delta: str) -> None:
        if not delta:
            return
        buf = self._hold + delta
        self._hold = ""
        while buf:
            marker = MEM_CLOSE if self._hidden else MEM_OPEN
            mark = marker.casefold()
            idx = buf.casefold().find(mark)
            if idx >= 0:
                if not self._hidden:
                    self._emit(buf[:idx])
                buf = buf[idx + len(marker) :]
                # Speech can follow the block, so resume instead of hiding for good.
                self._hidden = not self._hidden
                continue
            keep = _prefix_hold(buf, mark)
            if not self._hidden:
                self._emit(buf[:-keep] if keep else buf)
            self._hold = buf[-keep:] if keep else ""
            return

    def _emit(self, text: str) -> None:
        if text and self._on_token:
            self._on_token(text)

    def flush(self) -> None:
        hold, self._hold = self._hold, ""
        if self._hidden:
            return
        self._emit(hold)


def _prefix_hold(buf: str, marker_cf: str) -> int:
    """How many trailing chars might be the start of the marker."""
    max_n = min(len(buf), len(marker_cf) - 1)
    tail = buf[-max_n:].casefold() if max_n else ""
    for n in range(max_n, 0, -1):
        if marker_cf.startswith(tail[-n:]):
            return n
    return 0
