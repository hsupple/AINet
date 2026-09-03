"""query_db routing, prefetch, and reply-quality hints."""

from __future__ import annotations

import re
from typing import Any

_SELF_MARKERS = (
    " who am i",
    " who i am",
    " what am i like",
    " what i'm like",
    " what im like",
    " my characteristics",
    " figure out my characteristics",
    " about myself",
    " about me",
    " tell me about me",
    " what do you know about me",
    " my personality",
    " what i'm like",
    " where do i go to school",
    " where do i study",
    " my school",
    " what kind of person am i",
    " when it comes to quality",
)


def hayden_asking_about_self(text: str) -> bool:
    """True when Hayden is asking about himself, not other people."""
    low = f" {(text or '').casefold()} "
    if any(w in low for w in (" pin", " pins", "password", "secret")):
        return False
    if any(marker in low for marker in _SELF_MARKERS):
        return True
    stripped = low.strip()
    return stripped.startswith(("who am i", "what am i", "what's my", "whats my"))


def self_identity_context(user_text: str, standing_request: str = "") -> bool:
    """True when this turn continues an inquiry about who Hayden is."""
    if hayden_asking_about_self(user_text):
        return True
    if hayden_asking_about_self(standing_request):
        return True
    low = f" {(user_text or '').casefold()} "
    if any(
        phrase in low
        for phrase in (
            "tell me more",
            "go on",
            "yup",
            "yeah",
            "yes",
            "continue",
            "what else",
        )
    ) and hayden_asking_about_self(standing_request):
        return True
    return False


# First-person / personal-life talk — questions OR statements that need DB context.
_PERSONAL_MEMORY_RE = re.compile(
    r"(?:"
    r"\b(?:who am i|what am i|what(?:'s| is) my|who have i|who(?:'s| is) my|"
    r"what do you know about me|tell me about me|about myself)\b"
    r"|"
    r"\b(?:i(?:'m|m| am|ve| have)|i'd|idk|my|me|i)\b.{0,80}(?:"
    r"friend|friends|hang|school|college|purdue|personality|curious|curiosity|"
    r"habit|habits|prefer|preference|anxious|anxiety|stressed|stress|"
    r"home|apartment|running low|out of|wrist|knee|back|body|"
    r"workout|gym|campus|class|classes|club|labs?|engineering|"
    r"run|running|focus|caffeine|sleep|matcha|espresso|coffee|quality|craft|"
    r"memory|memories|win|wins|rebuild|rebuilt|car|bmw|notion|markdown|"
    r"pin|password|secret|desire|want|wanted|goal|value|craft|climbing|"
    r"pair-program|ros2|mic|hackathon|faucet|soap|oat|bodybuilding"
    r")"
    r"|"
    r"\b(?:hanging out|closest friend|body issues|note-taking|personal ai|"
    r"open loops|spent fuel|climbing gym|making friends|obhr|"
    r"running low|at home|around the apartment|big wins|rebuild a car|oat milk|"
    r"out of|sleep or caffeine|bmw rebuild)\b"
    r")",
    re.I | re.S,
)

_EXTERNAL_OVERRIDE_RE = re.compile(
    r"\b(?:weather|stock price|news|score|election|wikipedia|google|"
    r"youtube|how (?:do|does|to) (?:a|an|the|you)|what is a |"
    r"statistics?|prevalence|meta[- ]analysis)\b",
    re.I,
)

_WANTS_WEB_RE = re.compile(
    r"\b(?:"
    r"search|look(?:\s+it)?\s+up|google|wikipedia|youtube|"
    r"statistics?|prevalence|current|latest|news|weather|"
    r"how (?:do|does|to) (?:a|an|the|you|people|others)|"
    r"what (?:is|are) (?:a|an|the)|"
    r"bring me up|pull up|find (?:me )?(?:info|information|articles?|sources?)"
    r")\b",
    re.I,
)


def wants_open_web(text: str) -> bool:
    """Public/external lookup — keep web tools available."""
    raw = (text or "").strip()
    if not raw:
        return False
    if hayden_asking_about_self(raw):
        return False
    return bool(_WANTS_WEB_RE.search(raw) or _EXTERNAL_OVERRIDE_RE.search(raw))


def personal_memory_question(text: str) -> bool:
    """True when the turn is about Hayden's stored personal life (DB), not the open web.

    Covers questions and first-person statements that need personal context.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    if hayden_asking_about_self(raw):
        return True
    # "bring me up stats" / "how have other people" is public, even though "me" appears.
    if wants_open_web(raw) and not re.search(
        r"\b(?:i(?:'m|m| am|ve| have)|my (?:friend|friends|school|habit|anxiety|wrist|personality))\b",
        raw,
        re.I,
    ):
        return False
    if _EXTERNAL_OVERRIDE_RE.search(raw) and not re.search(
        r"\b(?:i(?:'m|m| am|ve| have)|my)\b", raw, re.I
    ):
        return False
    if _PERSONAL_MEMORY_RE.search(raw):
        return True
    if re.search(
        r"\b[A-Z][a-z]{1,24}\b.{0,48}\b(?:came|texted|visited|hang|hanging|friend|ghosted|over|lunch)\b",
        raw,
    ):
        return True
    return False


def is_hayden_db_dest(dest: str) -> bool:
    key = (dest or "").replace("\\", "/").strip("/").casefold()
    if not key:
        return False
    if key in {"hayden", "hayden.json"}:
        return True
    return key in {
        "characteristics",
        "preferences",
        "habits",
        "values",
        "desires",
        "body",
        "psychology",
    }


def query_result_includes_hayden(result: dict) -> bool:
    matches = result.get("matches")
    if not isinstance(matches, list):
        return False
    for row in matches:
        if not isinstance(row, dict):
            continue
        if str(row.get("file") or "").casefold() == "hayden.json":
            return True
    return False


def wrong_dest_for_self_query(dest: str) -> bool:
    key = (dest or "").replace("\\", "/").strip("/").casefold()
    return key in {
        "people",
        "people.json",
        "person",
        "relationship",
        "relationships",
    }


_BAD_SELF_REPLY_MARKERS = (
    "i'm hayden",
    "im hayden",
    "i am hayden",
    "i've retrieved",
    "i have retrieved",
    "would you like me to summarize",
    "let me use query_db",
    "let me know if you'd like",
    "retrieved information about hayden",
    "what about you",
    "what are you like",
    "could you please clarify",
    "could you clarify",
    "what specific aspects",
    "haven't stored any",
    "have not stored any",
    "no matches to report",
    "i am your ai assistant",
    "i'm your ai assistant",
    "how can i assist you today",
    "you are hayden, and i am",
)


def looks_like_bad_self_reply(text: str) -> bool:
    low = (text or "").casefold()
    return any(marker in low for marker in _BAD_SELF_REPLY_MARKERS)


_DUMP_GLUE = re.compile(
    r"(?:You're a (?:mechanical engineering student|student at)|"
    r"You are a (?:mechanical engineering student|student at)|"
    r"You're technically oriented|"
    r"You are technically oriented|"
    r"Your interests span|"
    r"You have hands-on (?:engineering )?experience)",
    re.I,
)

_EMPTY_THERAPY_CLOSERS = (
    "if you ever need advice",
    "if you ever need someone",
    "i'm here",
    "i am here",
    "i'm always here",
    "here if you need",
)
_EMPTY_THERAPY_OPENERS = (
    "i understand how you feel",
    "it sounds like you're dealing",
    "it sounds like you are dealing",
    "you're not alone",
    "you are not alone",
    "totally normal",
    "completely normal",
    "keep being yourself",
)


def looks_like_profile_dump(text: str) -> bool:
    raw = text or ""
    if _DUMP_GLUE.search(raw):
        return True
    low = raw.casefold()
    markers = (
        "your interests span",
        "technically oriented, curious",
        "bmw 540i",
        "autonomous racing",
        "cad, fea",
        "mechanical engineering student at purdue",
    )
    hits = sum(1 for m in markers if m in low)
    return hits >= 2


def strip_profile_dump(reply: str, digest: str = "") -> str:
    """Drop a glued-on biography recap. Keep the actual answer."""
    raw = (reply or "").strip()
    if not raw:
        return raw
    match = _DUMP_GLUE.search(raw)
    if match and match.start() > 60:
        return raw[: match.start()].rstrip()

    if digest:
        lines = [
            ln.strip()
            for ln in digest.splitlines()
            if ln.strip() and ":" in ln and len(ln.strip()) > 24
        ]
        if len(lines) >= 3:
            tail = raw[int(len(raw) * 0.45) :]
            hits = sum(1 for ln in lines if ln.split(":", 1)[-1].strip()[:40].casefold() in tail.casefold())
            if hits >= 3 and match:
                return raw[: match.start()].rstrip()
            if hits >= 4:
                # Last sentence-or-so is a recap — cut at the last period before the dump-y tail.
                cut = raw.rfind(". ", 0, int(len(raw) * 0.7))
                if cut > 80:
                    return raw[: cut + 1].strip()
    return raw


def _fold(text: str) -> str:
    return (text or "").replace("\u2019", "'").replace("\u2018", "'").casefold()


def looks_like_empty_therapy(reply: str) -> bool:
    """Generic empathy closer that does not actually engage the thread."""
    low = _fold(reply)
    if not low.strip():
        return False
    if "you're not alone" in low or "you are not alone" in low:
        return True
    has_closer = any(c in low for c in _EMPTY_THERAPY_CLOSERS)
    has_opener = any(o in low for o in _EMPTY_THERAPY_OPENERS)
    return has_closer and has_opener


_FRESH_START_MARKERS = (
    "how can i assist you",
    "how can i help you",
    "what can i do for you",
    "i'm here!",
    "i am here!",
    "how may i help",
)


def looks_like_fresh_start(reply: str) -> bool:
    """Model restarted as a new assistant instead of staying in the thread."""
    low = _fold(reply)
    if not low.strip():
        return False
    return any(m in low for m in _FRESH_START_MARKERS)


_STOPWORDS = frozenset(
    """
    the a an and or but if to of in on at for from with about into over after
    this that these those then than so just really like kinda kinda lowkey
    you your yours we they them their it its it's is are was were be been being
    do does did doing have has had having will would could should can cannot
    not no nor never what which who whom how when where why whom
    i i'm im ive i've id i'd me my myself mine
    he she his her they them
    get got getting go going went gone come came
    also still even much many some any all more most other another
    yeah yes yup ok okay cool thanks thank please
    shit fuck fucking kinda kinda idk dont don't wont won't gonna
    """.split()
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#./-]{1,}")


_TOKEN_EXPAND: dict[str, tuple[str, ...]] = {
    "school": ("education", "college", "university", "student", "purdue"),
    "hanging": ("friend", "people"),
    "friends": ("people",),
    "friend": ("people",),
    "home": ("household", "apartment"),
    "apartment": ("household", "faucet"),
    "broken": ("household", "faucet"),
    "annoying": ("household", "faucet"),
    "car": ("bmw", "rebuild"),
    "rebuild": ("bmw", "car"),
    "caffeine": ("coffee", "espresso", "matcha", "sleep"),
    "sleep": ("coffee", "espresso", "matcha"),
    "coffee": ("caffeine", "espresso"),
    "focus": ("habit", "stretch"),
    "breaks": ("focus", "habit", "stretch"),
    "quality": ("craft", "value"),
    "wins": ("memory", "hackathon", "esp32"),
    "run": ("running", "habit"),
    "regularly": ("running", "habit"),
    "lunch": ("people",),
    "low": ("household",),
    "pins": ("secrets", "pin"),
    "pin": ("secrets",),
}


def extract_query_tokens(*texts: str, limit: int = 14) -> list[str]:
    """Content tokens for query_db prefetch, first-seen order, latest text first."""
    seen: list[str] = []
    have: set[str] = set()
    for text in texts:
        for match in _TOKEN_RE.findall(text or ""):
            word = match.strip(".-/")
            if len(word) < 3:
                continue
            key = word.casefold()
            if key in _STOPWORDS or key in have:
                continue
            have.add(key)
            seen.append(word)
            for extra in _TOKEN_EXPAND.get(key, ()):
                if extra not in have:
                    have.add(extra)
                    seen.append(extra)
            if len(seen) >= limit:
                return seen
    return seen


def should_prefetch_calendar(
    user_text: str,
    standing: str = "",
    recent_turns: list[tuple[str, str]] | None = None,
) -> bool:
    """True when this turn is about Hayden's schedule / calendar."""
    from ainet.calendar_store import looks_like_calendar_write, looks_like_schedule_ask

    # Writes are handled by add_calendar_event / host add — do not dump the schedule.
    if looks_like_calendar_write(user_text):
        return False
    if looks_like_schedule_ask(user_text):
        return True
    if standing and looks_like_schedule_ask(standing):
        low = f" {(user_text or '').casefold()} "
        if any(
            phrase in low
            for phrase in (
                "tell me more",
                "go on",
                "what else",
                "and tomorrow",
                "what about",
            )
        ):
            return True
    for user, _assistant in (recent_turns or [])[-2:]:
        if looks_like_schedule_ask(user) and looks_like_schedule_ask(user_text):
            return True
    return False


def should_prefetch(
    user_text: str,
    standing: str = "",
    recent_turns: list[tuple[str, str]] | None = None,
) -> bool:
    # Schedule / class / lab asks belong to calendar prefetch — do not steal with query_db.
    if should_prefetch_calendar(user_text, standing, recent_turns):
        if not hayden_asking_about_self(user_text):
            return False
    if hayden_asking_about_self(user_text) or personal_memory_question(user_text):
        return True
    if standing and personal_memory_question(standing):
        return True
    for user, _assistant in (recent_turns or [])[-4:]:
        if personal_memory_question(user):
            return True
    return False


def compact_digest(digest: str, tokens: list[str], *, max_chars: int = 1400) -> str:
    """Keep observation lines that overlap the current thread; drop unrelated bio."""
    lines = [ln.strip() for ln in (digest or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    want = [t.casefold() for t in tokens if len(t) >= 3]
    scored: list[tuple[int, str]] = []
    for line in lines:
        low = line.casefold()
        hits = sum(1 for t in want if t in low) if want else 1
        scored.append((hits, line))
    scored.sort(key=lambda item: item[0], reverse=True)
    kept: list[str] = []
    total = 0
    for hits, line in scored:
        if want and hits <= 0 and kept:
            continue
        if total + len(line) + 1 > max_chars:
            break
        kept.append(line)
        total += len(line) + 1
        if len(kept) >= 12:
            break
    if not kept:
        for hits, line in scored[:6]:
            if total + len(line) + 1 > max_chars:
                break
            kept.append(line)
            total += len(line) + 1
    return "\n".join(kept)


def prefetch_personal_context(
    db: Any,
    user_text: str,
    standing: str = "",
    recent_turns: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Host-side query_db for the live thread. Inject, don't recite."""
    from ainet.logstore import query_db as query_db_fn

    prior_user = " ".join(u for u, _a in (recent_turns or [])[-4:])
    tokens = extract_query_tokens(user_text, standing, prior_user)
    if hayden_asking_about_self(user_text):
        result = query_db_fn(db, dest="hayden", q=" ".join(tokens[:8]), limit=10)
        if not (isinstance(result, dict) and (result.get("digest") or result.get("matches"))):
            result = query_db_fn(db, dest="hayden", limit=10)
    else:
        q = " ".join(tokens[:10])
        result = query_db_fn(db, q=q, limit=10) if q else {"ok": True, "digest": "", "matches": []}
        if isinstance(result, dict) and not (result.get("digest") or result.get("matches")) and tokens:
            result = query_db_fn(db, dest="hayden", q=q, limit=8)

    digest = ""
    count = 0
    if isinstance(result, dict):
        digest = str(result.get("digest") or "").strip()
        count = int(result.get("count") or 0)
        if not digest:
            parts: list[str] = []
            for row in result.get("matches") or []:
                if not isinstance(row, dict):
                    continue
                key = str(row.get("name") or "").strip()
                for entry in row.get("entries") or []:
                    if isinstance(entry, dict) and entry.get("text"):
                        text = str(entry["text"]).strip()
                        parts.append(f"{key}: {text}" if key else text)
            digest = "\n".join(parts)
    digest = compact_digest(digest, tokens)
    return {
        "ok": bool(digest),
        "digest": digest,
        "tokens": tokens,
        "count": count,
        "query": " ".join(tokens[:10]),
    }


def distinctive_user_tokens(user_text: str, *, limit: int = 8) -> list[str]:
    """Proper nouns and uncommon words Hayden just said — replies should use some of them."""
    out: list[str] = []
    have: set[str] = set()
    for match in _TOKEN_RE.findall(user_text or ""):
        word = match.strip(".-/")
        if len(word) < 4:
            continue
        key = word.casefold()
        if key in _STOPWORDS or key in have:
            continue
        if word[0].isupper() or len(word) >= 6:
            have.add(key)
            out.append(word)
            if len(out) >= limit:
                break
    return out


def looks_like_generic_ignore(reply: str, user_text: str) -> bool:
    """Reply ignored the specifics Hayden just named and fell back to a template."""
    specs = distinctive_user_tokens(user_text)
    if len(specs) < 3:
        return False
    low = (reply or "").casefold()
    hits = sum(1 for s in specs if s.casefold() in low)
    if hits >= 2:
        return False
    return looks_like_empty_therapy(reply) or looks_like_profile_dump(reply)
