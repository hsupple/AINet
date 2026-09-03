"""Hidden rolling conversation memory — model rewrites it; host strips it from Hayden."""

from __future__ import annotations

import re
from collections.abc import Callable

MEM_OPEN = "%%mem%%"
MEM_CLOSE = "%%end%%"
_MAX_LINES = 3
_MAX_LINE = 280
_MAX_TOTAL = 840

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


def host_fallback_memory(
    user_text: str,
    assistant_text: str,
    prior: str = "",
    recent_turns: list[tuple[str, str]] | None = None,
) -> str:
    """If the model skips %%mem%%, retain a labeled rolling-memory block with the thread."""
    ask = " ".join((user_text or "").split())[:200]
    gist = " ".join((assistant_text or "").split())[:200]
    fields = _memory_fields(prior)
    if named_search_topic(user_text) or (
        _looks_like_new_question(user_text) and not is_vague_pronoun_followup(user_text)
    ):
        standing = ask
    else:
        standing = fields.get("standing request") or ask
    thread_bits: list[str] = []
    for user, _answer in (recent_turns or [])[-4:]:
        bit = " ".join((user or "").split())[:90]
        if bit:
            thread_bits.append(bit)
    if ask and (not thread_bits or thread_bits[-1] != ask[:90]):
        thread_bits.append(ask[:90])
    thread = " → ".join(thread_bits)[:280]
    context = thread or fields.get("context") or ""
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
    r"get\s+(?:me\s+)?(?:a\s+)?videos?\b|pull\s+up|show\s+me|bring\s+(?:me\s+)?up"
    r")\b",
    re.I,
)
_VIDEO_HINT_RE = re.compile(r"\b(?:vids?\b|videos?\b|youtube|youtu\.be|watch\b|clip\b|tutorial\b)\b", re.I)
_ACTION_CONFIRM_RE = re.compile(
    r"^\s*(?:do\s+it|do\s+that|go\s+ahead|yes|yeah|yep|sure|please|"
    r"yes\s+please|please\s+do|ok\s+do\s+it)\s*[.!]?\s*$",
    re.I,
)
_SHORT_ACK_RE = re.compile(
    r"^\s*(?:ok|okay|k|kk|cool|thanks|thank\s+you|thx|got\s+it|alright|np|nice)\s*[.!]?\s*$",
    re.I,
)
_OFFERED_SEARCH_MARKERS = (
    "if you would like",
    "i can help you find",
    "let me search",
    "let me find",
    "one moment",
    "would you like me to search",
    "would you like me to look",
    "would you like me to find",
    "i'll find",
    "i will find",
    "searching for",
    "i can look",
)
_OFFERED_CALENDAR_MARKERS = (
    "create this event",
    "add this event",
    "add it to your calendar",
    "add it to the calendar",
    "put it on your calendar",
    "put that on your calendar",
    "save this to your calendar",
    "add that to your calendar",
    "would you like me to create",
    "would you like me to add",
    "want me to add",
    "want me to create",
    "i can add that",
    "i can create that",
    "shall i add",
    "should i add",
)


def standing_request(memory: str) -> str:
    return _memory_fields(memory).get("standing request", "").strip()


def is_vague_search_followup(user_text: str) -> bool:
    raw = " ".join((user_text or "").split())
    if not raw:
        return False
    if _VAGUE_SEARCH_RE.search(raw):
        return True
    if _VIDEO_HINT_RE.search(raw) and len(raw.split()) <= 12 and not _looks_like_new_question(user_text):
        return True
    return False


_VIDEO_ACTION_LEAD_RE = re.compile(
    r"^(?:hey\s+\w+\s*,?\s*)?(?:please\s+)?(?:can you |could you |would you )?"
    r"(?:bring\s+up|pull\s+up|show(?:\s+me)?|find(?:\s+me)?|get(?:\s+me)?|"
    r"open|watch|search(?:\s+for)?)\s+"
    r"(?:some\s+|a\s+|an\s+|the\s+)?"
    r"(?:videos?|lectures?|clips?|youtube)\s*"
    r"(?:on|about|explaining|for|of|regarding)?\s*",
    re.I,
)
_VIDEO_FILLER = frozenset(
    {
        "some",
        "a",
        "an",
        "the",
        "please",
        "pls",
        "video",
        "videos",
        "lecture",
        "lectures",
        "youtube",
        "clip",
        "clips",
        "it",
        "this",
        "that",
        "them",
        "those",
    }
)


def named_search_topic(user_text: str) -> str:
    """If this message names what to search/watch, return that — not the prior thread."""
    raw = " ".join((user_text or "").split())
    if not raw:
        return ""
    rest = _VIDEO_ACTION_LEAD_RE.sub("", raw, count=1).strip()
    if rest == raw:
        mid = re.search(
            r"(?:videos?|lectures?|youtube)\s+(?:on|about|explaining|for|of)\s+(.+)",
            raw,
            re.I,
        )
        if mid:
            rest = mid.group(1).strip()
        elif wants_videos(raw):
            stripped = re.sub(
                r"\b(?:bring\s+up|pull\s+up|show(?:\s+me)?|find(?:\s+me)?|please|pls|"
                r"youtube|videos?|lectures?|clips?)\b",
                " ",
                raw,
                flags=re.I,
            )
            rest = " ".join(stripped.split())
        else:
            return ""
    rest = re.sub(r"\b(?:is\s+)?pre?ff?err?e?d\b", " ", rest, flags=re.I)
    rest = re.sub(r"\b(?:please|pls)\b", " ", rest, flags=re.I)
    rest = " ".join(rest.split()).strip(" .,!;:")
    words = [w for w in rest.split() if w.casefold().strip(".,!") not in _VIDEO_FILLER]
    weak = _VIDEO_FILLER | {"of", "on", "about", "for", "me", "some", "any"}
    if not words or all(w.casefold().strip(".,!") in weak for w in rest.split()):
        return ""
    return rest[:160]


def wants_videos(user_text: str) -> bool:
    raw = " ".join((user_text or "").split())
    if not raw or not _VIDEO_HINT_RE.search(raw):
        return False
    if re.search(r"\b(?:pull\s+up|show|find|get|bring\s+up|open|watch|search)\b", raw, re.I):
        return True
    return len(raw.split()) <= 12 and not _looks_like_new_question(user_text)


def is_action_confirm(user_text: str) -> bool:
    return bool(_ACTION_CONFIRM_RE.match(" ".join((user_text or "").split())))


def is_short_ack(user_text: str) -> bool:
    raw = " ".join((user_text or "").split())
    if is_action_confirm(raw):
        return False
    return bool(_SHORT_ACK_RE.match(raw))


def offered_to_search(assistant_text: str) -> bool:
    low = (assistant_text or "").casefold()
    if offered_to_calendar(low):
        return False
    return any(m in low for m in _OFFERED_SEARCH_MARKERS)


def offered_to_calendar(assistant_text: str) -> bool:
    low = (assistant_text or "").casefold()
    if any(m in low for m in _OFFERED_CALENDAR_MARKERS):
        return True
    if "calendar" in low and any(
        p in low for p in ("would you like me to", "want me to", "shall i", "should i")
    ):
        return True
    if "event" in low and any(
        p in low for p in ("would you like me to create", "would you like me to add", "want me to add")
    ):
        return True
    return False


def last_user_wanted_videos(recent_turns: list[tuple[str, str]] | None) -> bool:
    for user, _ans in reversed(recent_turns or []):
        if wants_videos(user) or is_vague_search_followup(user):
            return True
        if is_action_confirm(user) or is_short_ack(user):
            continue
        return False
    return False


def topic_for_search(
    memory: str,
    recent_turns: list[tuple[str, str]] | None = None,
) -> str:
    """Noun-ish topic for a follow-up search, not the action phrase itself."""
    topic = followup_topic(memory, recent_turns)
    if topic and (
        wants_videos(topic) or is_action_confirm(topic) or is_short_ack(topic)
    ):
        topic = ""
    if not topic:
        for user, _ans in reversed(recent_turns or []):
            bit = " ".join((user or "").split())
            if not bit or wants_videos(bit) or is_action_confirm(bit) or is_short_ack(bit):
                continue
            topic = bit
            break
    topic = re.sub(
        r"^(?:what|who|how|when|where|why|which|please|explain|tell\s+me)\s+"
        r"(?:is|are|was|were|the|a|an|to|about)?\s*",
        "",
        topic or "",
        flags=re.I,
    )
    topic = re.sub(r"^(?:the|a|an)\s+", "", topic.strip(), flags=re.I)
    return topic.strip(" ?.!")[:160]


def enrich_web_search_query(query: str, user_text: str, standing: str) -> tuple[str, bool]:
    """When Hayden's follow-up is vague, search the standing request instead of a bad guess."""
    named = named_search_topic(user_text)
    if named:
        user = " ".join((user_text or "").split())
        if _VIDEO_HINT_RE.search(user) and not re.search(r"\byoutube\b", named, re.I):
            new_q = f"{named} youtube video"
        else:
            new_q = named
        new_q = new_q[:200].strip()
        old_q = " ".join((query or "").split()).strip()
        return new_q, new_q.casefold() != old_q.casefold()

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


def is_vague_pronoun_followup(user_text: str) -> bool:
    """Follow-ups that depend on prior context (it/that/this/them)."""
    raw = " ".join((user_text or "").split())
    if not raw or len(raw.split()) > 18:
        return False
    low = f" {raw.casefold()} "
    if not re.search(r"\b(?:it|that|this|them|those|same)\b", low):
        return False
    return True


CLARIFIER_MARKERS = (
    "a bit vague",
    "a bit ambiguous",
    "could you please clarify",
    "could you please specify",
    "please specify what",
    "please specify",
    "too vague",
    "not sure what you mean",
    "what you mean by",
    "what you're referring",
    "what you are referring",
    "what it refers to",
    'what "it" refers',
    "what 'it' refers",
    "clarify what",
    "would you like me to search",
    "let me try a different approach",
)


def reply_looks_like_clarifier(reply: str) -> bool:
    low = (reply or "").casefold()
    if not low.strip():
        return False
    return any(m in low for m in CLARIFIER_MARKERS)


def strip_leading_clarifier(reply: str) -> str:
    """Drop a 'what does it mean?' preamble glued onto a real answer."""
    raw = (reply or "").strip()
    if not raw or not reply_looks_like_clarifier(raw):
        return raw
    glued = re.search(r"\?([A-Z])", raw)
    if glued and glued.start() > 40:
        rest = raw[glued.start() + 1 :].strip()
        if len(rest) > 40:
            return rest
    parts = re.split(r"\n\s*\n", raw, maxsplit=1)
    if len(parts) == 2 and reply_looks_like_clarifier(parts[0]) and len(parts[1]) > 40:
        return parts[1].strip()
    return raw


def followup_topic(memory: str, recent_turns: list[tuple[str, str]] | None = None) -> str:
    """Best guess at what 'it' refers to this turn."""
    topic = standing_request(memory).strip()
    if topic:
        return topic[:200]
    for user, _ans in reversed(recent_turns or []):
        bit = " ".join((user or "").split())
        if bit:
            return bit[:200]
    return ""


def recent_turns_block(
    turns: list[tuple[str, str]],
    links: list[tuple[str, str]] | None = None,
    *,
    max_turns: int = 5,
) -> str:
    """Pin the last 2–3 user+answer pairs so pronouns and short follow-ups resolve."""
    cleaned: list[tuple[str, str]] = []
    for user_text, assistant_text in turns or []:
        user = " ".join((user_text or "").split())
        answer = (assistant_text or "").strip()
        if not user and not answer:
            continue
        cleaned.append((user, answer))
    if not cleaned:
        return ""
    slice_turns = cleaned[-max(1, max_turns) :]
    lines = [
        "\n\nRecent turns (this is the live conversation — stay in it. "
        "Resolve pronouns like it/that/this from these. Use names, places, "
        "and constraints Hayden already gave. Do not ask what 'it' means "
        "when the topic is already clear here or in Standing request):",
    ]
    # Older turns get a tighter cap so the latest stays richest.
    for i, (user, answer) in enumerate(slice_turns):
        is_latest = i == len(slice_turns) - 1
        u_cap, a_cap = (700, 900) if is_latest else (400, 480)
        if len(user) > u_cap:
            user = user[:u_cap] + "…"
        if len(answer) > a_cap:
            answer = answer[:a_cap] + "…"
        label = "Latest" if is_latest else f"Earlier ({i + 1})"
        lines.append(f"— {label} —")
        lines.append(f"Hayden: {user or '(none)'}")
        lines.append(f"You: {answer or '(none)'}")

    latest_answer = slice_turns[-1][1]
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
        for href in _URL_RE.findall(latest_answer):
            href = href.rstrip(").,;]")
            if href in seen:
                continue
            seen.add(href)
            clean_links.append(f"- {href}")
            if len(clean_links) >= 8:
                break
    if clean_links:
        lines.append("Links from the latest turn (use these if Hayden says yes / open them):")
        lines.extend(clean_links)
    return "\n".join(lines)


def last_turn_block(
    user_text: str,
    assistant_text: str,
    links: list[tuple[str, str]] | None = None,
) -> str:
    """Pin the previous user+answer (and URLs) so follow-ups like 'open those' still work."""
    return recent_turns_block([(user_text, assistant_text)], links, max_turns=1)


def known_context_block(digest: str) -> str:
    body = (digest or "").strip()
    if not body:
        return ""
    cal_note = ""
    if "CALENDAR" in body.splitlines()[0] or "\nCALENDAR\n" in f"\n{body}\n":
        cal_note = (
            "CALENDAR lines are Hayden's real schedule — answer class/lab/meeting "
            "timing from them. Never claim you lack access to his schedule when "
            "CALENDAR context is present. "
        )
    return (
        "\n\nKNOWN CONTEXT (host-retrieved from the database/calendar for this turn — "
        "use silently; do not recap as a biography; do not list unrelated traits):\n"
        f"{body}\n"
        f"{cal_note}"
        "You may still call query_db if you need a fact that is not here. "
        "You may still call query_calendar if you need more of the schedule. "
        "You may still web_search for public/external facts."
    )


def memory_system_suffix(memory: str) -> str:
    body = _labeled_memory(memory)
    return (
        "\n\nROLLING MEMORY STATE\n"
        "This is hidden host context, not spoken text:\n"
        f"{MEM_OPEN}\n{body}\n{MEM_CLOSE}\n"
        "Answer Hayden in plain text first, then replace this block at the very end "
        "with exactly three concise labeled lines: "
        "Standing request, Context (the running thread: people, places, constraints, "
        "what he already rejected), and Last answer. "
        "Keep the standing request until Hayden changes it. "
        "If Hayden asks a new clear question, replace Standing request with that question "
        "but keep Context as the thread unless the topic actually changed. "
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

    def reset(self) -> None:
        self._hold = ""
        self._hidden = False

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
