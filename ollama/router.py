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


_PLANNER = re.compile(
    r"\b("
    r"plan my|schedule|next actions?|to-?do|this week|agenda|prioritiz|"
    r"break down (the |this )?project|milestones for"
    r")\b",
    re.I,
)
_OPEN_PROJECT = re.compile(
    r"\b("
    r"open (the |my )?project|focus (on )?(the |my )?project|"
    r"switch to (the |my )?project|go (in)?to (the |my )?project"
    r")\b",
    re.I,
)
_CREATE_PROJECT = re.compile(
    r"\b(create|make|start|new) (a |an |the |my )?project\b",
    re.I,
)
_DEEP_RESEARCH = re.compile(
    r"\b("
    r"deep[- ]research|use deep research|do (a )?deep research|"
    r"research brief|literature review"
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

    # Stay in project focus unless Hayden explicitly exits (tools handle close).
    if current_mode_id == "project":
        return RouteDecision("project", 0.9, "project focus sticky")

    if re.search(
        r"\b(stay in|keep|switch to) (companion|conversation|planner|project|deep[- ]?research)\b",
        text,
        re.I,
    ):
        m = re.search(
            r"\b(companion|conversation|planner|project|deep[- ]?research)\b", text, re.I
        )
        if m:
            picked = m.group(1).lower().replace(" ", "_").replace("-", "_")
            if picked == "deep_research" or picked.startswith("deep"):
                picked = "deep_research"
            return RouteDecision(picked, 0.99, "explicit request")

    if _DEEP_RESEARCH.search(text):
        return RouteDecision("deep_research", 0.95, "deep research request")

    # open_project / create_project are tools — don't auto-switch mode without a named project.
    if _OPEN_PROJECT.search(text) or _CREATE_PROJECT.search(text):
        return RouteDecision(current_mode_id, 0.55, "project tool request")

    if current_mode_id in {"conversation", "planner", "deep_research", "project"} and (
        _CONTINUE.match(text)
        or (
            len(text) < 60
            and not _PLANNER.search(text)
            and not _DEEP_RESEARCH.search(text)
            and not re.match(r"^\s*(hi|hey|hello)\b", text, re.I)
        )
    ):
        return RouteDecision(current_mode_id, 0.7, "sticky follow-up")

    if _COMPANION_SHORT.search(text) and len(text) < 40:
        return RouteDecision("companion", 0.85, "short spoken utterance")

    if _PLANNER.search(text):
        return RouteDecision("planner", 0.8, "planning language")

    if _CONVERSATION.search(text):
        return RouteDecision("conversation", 0.75, "personal dialogue cues")

    if current_mode_id in {"conversation", "planner", "deep_research", "project"} and len(text) > 20:
        return RouteDecision(current_mode_id, 0.4, "sticky follow-up")

    return RouteDecision(current_mode_id or DEFAULT_MODE_ID, 0.2, "default stay")
