"""Cheap heuristic mode router — NO extra LLM call (latency/tokens matter)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ollama.modes import DEFAULT_MODE_ID


@dataclass(frozen=True)
class RouteDecision:
    mode_id: str
    confidence: float
    reason: str


_RESEARCH = re.compile(
    r"\b("
    r"how does|how do|mechanism|explain (in )?(depth|detail)|deep dive|rabbit hole|"
    r"derive|proof|theorem|quantum|black hole|synthase|mitochondr|neuron|enzyme|"
    r"history of|discovered|evidence for|why does|walk me through the|"
    r"under the hood|fundamentals of"
    r")\b",
    re.I,
)
_PLANNER = re.compile(
    r"\b("
    r"plan my|schedule|next actions?|to-?do|this week|agenda|prioritiz|"
    r"break down (the |this )?project|milestones for"
    r")\b",
    re.I,
)
_CONVERSATION = re.compile(
    r"\b("
    r"i feel|i've been|ive been|what do you think about me|can we talk|"
    r"on my mind|been thinking|relationship with|i'm struggling|im struggling"
    r")\b",
    re.I,
)
_COMPANION_SHORT = re.compile(
    r"^\s*(hi|hey|hello|yo|sup|thanks|thank you|ok|okay|cool|got it|nm|never ?mind)\b",
    re.I,
)
_CONTINUE = re.compile(
    r"^\s*(ok|okay|cool|got it|go on|continue|more|and\??|why\??|how\??|"
    r"yes|yeah|yep|what about|tell me more)\b",
    re.I,
)


def suggest_mode(user_text: str, current_mode_id: str) -> RouteDecision:
    """Return a mode suggestion. Low confidence ⇒ stay on current mode."""
    text = (user_text or "").strip()
    if not text:
        return RouteDecision(current_mode_id, 0.0, "empty")

    # Explicit lock phrases
    if re.search(r"\b(stay in|keep|switch to) (companion|conversation|research|planner)\b", text, re.I):
        m = re.search(r"\b(companion|conversation|research|planner)\b", text, re.I)
        if m:
            return RouteDecision(m.group(1).lower(), 0.99, "explicit request")

    # Sticky deep modes: short follow-ups should not bounce to companion
    if current_mode_id in {"research", "conversation", "planner"} and (
        _CONTINUE.match(text) or (len(text) < 60 and not _PLANNER.search(text) and not re.match(r"^\s*(hi|hey|hello)\b", text, re.I))
    ):
        return RouteDecision(current_mode_id, 0.7, "sticky follow-up")

    if _COMPANION_SHORT.search(text) and len(text) < 40:
        return RouteDecision("companion", 0.85, "short spoken utterance")

    if _PLANNER.search(text):
        return RouteDecision("planner", 0.8, "planning language")

    if _RESEARCH.search(text) or (len(text) > 180 and "?" in text):
        conf = 0.82 if _RESEARCH.search(text) else 0.55
        return RouteDecision("research", conf, "deep-explain / long question")

    if _CONVERSATION.search(text):
        return RouteDecision("conversation", 0.75, "personal dialogue cues")

    if current_mode_id in {"research", "conversation", "planner"} and len(text) > 20:
        return RouteDecision(current_mode_id, 0.4, "sticky follow-up")

    return RouteDecision(current_mode_id or DEFAULT_MODE_ID, 0.2, "default stay")
