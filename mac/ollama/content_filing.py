"""Content-based filing hints. Ignore OAC mode — file from what was said."""

from __future__ import annotations

import re
from typing import Any


_EPHEMERAL = re.compile(
    r"^\s*(hi|hey|hello|yo|sup|hiya|hi bud|hey pal|good (morning|afternoon|evening)"
    r"|thanks|thank you|thx|ty|cool|ok|okay|got it|bye|good ?night|gg|np|yw|lol ok)\b[.!]?\s*$",
    re.I,
)
_PSYCH = re.compile(
    r"("
    r"\bfeelings?\b|\blonely\b|\bloneliness\b|\bfloating\b|"
    r"\bi feel\b|i just feel|i feel like|"
    r"i'?m (pretty |kinda |so |really )?(anxious|stressed|scared|sad|depressed|lonely|overwhelmed|worthless)|"
    r"discuss my feelings|talk about my feelings|my feelings|"
    r"i (don'?t|do not) like myself|i hate myself|i dislike myself|"
    r"self[- ]esteem|self[- ]hate|trigger|open loops?|coping|defense|attachment|"
    r"i'?m struggling|makes me (feel|anxious)|"
    r"need a (friend|partner)|my own best friend|"
    r"friends don'?t|family doesn'?t"
    r")",
    re.I,
)
_HABIT = re.compile(
    r"\b(every (morning|afternoon|night|day)|routine|habit|pomodoro|"
    r"i always|i keep|discipline|vice|i'?ve been switching|"
    r"gym|workout|went to the gym)\b",
    re.I,
)
_IDENTITY = re.compile(
    r"\b(i (care|value|hate|love) |who i am|that'?s so me|craftsmanship|"
    r"how i (talk|speak|sound)|my (tone|personality|style))\b",
    re.I,
)
_VOICE_CUE = re.compile(
    r"\b(fuck|shit|ass|bitch|damn|crap|retard|lol|lmao|ngl|fr)\b",
    re.I,
)
def is_ephemeral_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t) < 80 and _EPHEMERAL.match(t):
        return True
    return False


def content_kind(user_text: str, assistant_text: str = "") -> str:
    """Where lasting content belongs. Default is store, not discard."""
    user = (user_text or "").strip()
    asst = (assistant_text or "").strip()
    if is_ephemeral_text(user):
        return "discard"
    if _PSYCH.search(user):
        return "psychology"
    if _HABIT.search(user):
        return "habits"
    if _IDENTITY.search(user):
        return "identity"
    if _VOICE_CUE.search(user):
        return "voice"
    return "general"


def cop_name_in_text(path: str, user_text: str) -> bool:
    """True when the COP folder name actually appears in the turn."""
    name = (path or "").replace("\\", "/").rstrip("/").split("/")[-1]
    if name.lower().endswith(".json"):
        parent = (path or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[0]
        name = parent.split("/")[-1] if parent else name[:-5]
    token = re.sub(r"[^a-z0-9]+", "", name.lower())
    blob = re.sub(r"[^a-z0-9]+", "", (user_text or "").lower())
    if len(token) < 3 or not blob:
        return False
    return token in blob


def entry_kind(entry: dict[str, Any]) -> str:
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    user = str(details.get("user_text") or entry.get("summary") or "").strip()
    asst = str(details.get("assistant_text") or "").strip()
    return content_kind(user, asst)
