"""Content-based filing hints. Ignore OAC mode — file from what was said."""

from __future__ import annotations

import re
from typing import Any


_EPHEMERAL = re.compile(
    r"^\s*(hi|hey|hello|yo|sup|hiya|hi bud|hey pal|good (morning|afternoon|evening)"
    r"|thanks|thank you|thx|ty|cool|ok|okay|got it|bye|good ?night|gg|np|yw|lol ok)\b[.!]?\s*$",
    re.I,
)
_RESEARCH = re.compile(
    r"\b("
    r"how does|how do|what does|what is|what'?s|what are|why does|why do|"
    r"is this|does the|explain|mechanism|walk me through|under the hood|"
    r"learn(ing)? about|teach me|deep dive|rabbit hole|"
    r"vein|vena cava|jugular|blood|brain|oxygen|anatomy|physiology|"
    r"nuclear|fission|uranium|fuel rod|mitochondr|atp|enzyme|neuron|"
    r"qwen|alibaba|company|physics|biology|chemistry|quantum"
    r")\b",
    re.I,
)
_PSYCH = re.compile(
    r"\b(i feel|i'?m (anxious|stressed|scared|overwhelmed)|trigger|open loops?|"
    r"coping|defense|attachment|i'?m struggling|makes me anxious)\b",
    re.I,
)
_HABIT = re.compile(
    r"\b(every (morning|afternoon|night|day)|routine|habit|pomodoro|"
    r"i always|i keep|discipline|vice|i'?ve been switching)\b",
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
_FOLLOWUP = re.compile(
    r"^\s*(yeah|and so|awesome|ok so|wait|so then|and\??|what about|oh no way|no way)\b",
    re.I,
)
_QUESTION = re.compile(r"\?|(?:^(?:what|why|how|does|is|are)\b)", re.I)

_TOPIC_FAMILIES: list[tuple[str, str]] = [
    (r"vena cava|\bivc\b|jugular|vein that goes to the brain|main vein", "Inferior Vena Cava"),
    (r"brain.*(blood|oxygen)|blood.*brain|need so much blood|affect survival", "Brain Blood Requirements"),
    (r"uranium|fuel\s*rods?|fission|nuclear\s*reactor|spent\s+fuel|coolant", "Nuclear Fuel Rods"),
    (r"mitochondr|atp synthase|proton\s+gradient|chemiosmosis", "Mitochondria"),
    (r"\bqwen\b|alibaba", "Qwen"),
    (r"wrist|trackpad|keyboard ergonomic|reverse\s*curl|brachioradialis", "Wrist anatomy and keyboard ergonomics"),
]


def is_research_followup(user_text: str) -> bool:
    """Short continue phrases only — not 'Yeah, so I have <new topic>…'."""
    text = (user_text or "").strip()
    if not text or len(text) > 80:
        return False
    return bool(_FOLLOWUP.match(text))


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
    if _VOICE_CUE.search(user) and not _RESEARCH.search(user) and not _QUESTION.search(user):
        return "voice"
    if _RESEARCH.search(user) or _QUESTION.search(user):
        return "research"
    if is_research_followup(user) and len(asst) > 180:
        return "research"
    return "general"


def family_title_from_text(user_text: str) -> str | None:
    text = (user_text or "").strip()
    if not text:
        return None
    for pat, title in _TOPIC_FAMILIES:
        if re.search(pat, text, re.I):
            return title
    return None


def topic_title_from_text(user_text: str) -> str | None:
    text = (user_text or "").strip()
    if not text:
        return None
    fam = family_title_from_text(text)
    if fam:
        return fam
    cleaned = re.sub(
        r"^\s*(yeah|and so|awesome|ok so|wait|cant complain\.?)\s*",
        "",
        text,
        flags=re.I,
    )
    cleaned = re.sub(r"[?].*$", "", cleaned).strip(" .,!;:")
    if 8 <= len(cleaned) <= 72:
        return cleaned[:1].upper() + cleaned[1:]
    return None


def entry_kind(entry: dict[str, Any]) -> str:
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    user = str(details.get("user_text") or entry.get("summary") or "").strip()
    asst = str(details.get("assistant_text") or "").strip()
    return content_kind(user, asst)
