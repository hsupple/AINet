"""Detect when Hayden is asking about stored personal facts (not the open web)."""

from __future__ import annotations

import re
from typing import Any

_IM = r"(?:'m|m|\s+am)"
_PERSONAL_PRONOUN = re.compile(
    rf"\b(?:"
    rf"my\b|who am i\b|what am i like\b|how am i\b|about me\b|about myself\b|"
    rf"what do you know about me\b|look up who i am\b|find out who i am\b|"
    rf"what i{_IM} like\b|who i am\b|tell me about me\b"
    rf")\b",
    re.I,
)
_DB_EXPLICIT = re.compile(
    r"\b(?:database|query_db|hayden\.json|people\.json|stored|logged|filed|in the db)\b",
    re.I,
)
_MY_TRAIT = re.compile(
    rf"\bwhat(?:'s|'s| is|s)?\s+my\s+(\w+)\b|\bmy\s+(\w+)\b(?:\s+like)?\b",
    re.I,
)
_LOOK_UP = re.compile(
    rf"\b(?:look up|find out|check|search)\b.*\b(?:me|myself|who i am|what i{_IM} like)\b",
    re.I,
)
_NOT_PERSONAL = re.compile(
    r"\b(?:web|google|online|internet|youtube|video|news|article|paper|wiki)\b",
    re.I,
)


def looks_like_personal_db_query(text: str) -> bool:
    raw = (text or "").strip()
    if not raw or _NOT_PERSONAL.search(raw):
        return False
    if _DB_EXPLICIT.search(raw):
        return True
    if _PERSONAL_PRONOUN.search(raw):
        return True
    if _LOOK_UP.search(raw):
        return True
    if _MY_TRAIT.search(raw):
        trait = (_MY_TRAIT.search(raw).group(1) or _MY_TRAIT.search(raw).group(2) or "").casefold()
        if trait and trait not in {"database", "data", "file", "db", "phone", "computer", "pc"}:
            return True
    return False


def build_query_db_args(user_text: str) -> dict[str, Any]:
    raw = (text := (user_text or "").strip())
    low = raw.casefold()

    if any(w in low for w in ("people", "friends", "friend", "relationship")):
        return {"dest": "people", "limit": 16}

    if any(
        phrase in low
        for phrase in (
            "who i am",
            "who am i",
            "what i am like",
            "what i'm like",
            "what im like",
            "about me",
            "about myself",
            "look up who i am",
            "find out who i am",
            "look up what im like",
            "look up what i'm like",
            "in the database",
            "in my database",
            "what you know about me",
        )
    ):
        return {"dest": "hayden", "limit": 24}

    match = _MY_TRAIT.search(raw)
    if match:
        trait = (match.group(1) or match.group(2) or "").strip()
        if trait and trait.casefold() not in {"database", "data", "file", "db"}:
            return {"dest": "hayden", "name": trait, "limit": 16}

    return {"dest": "hayden", "q": " ".join(raw.split()[:8]), "limit": 16}
