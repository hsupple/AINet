"""Small helpers for query_db routing — hints only, no forced prefetch."""

from __future__ import annotations

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
)


def hayden_asking_about_self(text: str) -> bool:
    """True when Hayden is asking about himself, not other people."""
    low = f" {(text or '').casefold()} "
    if any(marker in low for marker in _SELF_MARKERS):
        return True
    stripped = low.strip()
    return stripped.startswith(("who am i", "what am i"))


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
)


def looks_like_bad_self_reply(text: str) -> bool:
    low = (text or "").casefold()
    return any(marker in low for marker in _BAD_SELF_REPLY_MARKERS)
