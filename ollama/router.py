"""Cheap heuristic router — capability modes only. No personality flavors."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ollama.modes import DEFAULT_MODE_ID


@dataclass(frozen=True)
class RouteDecision:
    mode_id: str
    confidence: float
    reason: str


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


def suggest_mode(user_text: str, current_mode_id: str) -> RouteDecision:
    """Capability routing only. Companion / conversation / planner are the same OAC."""
    text = (user_text or "").strip()
    current = current_mode_id or DEFAULT_MODE_ID
    if current in {"conversation", "planner", "quiz"}:
        current = DEFAULT_MODE_ID
    if not text:
        return RouteDecision(current, 0.0, "empty")

    if current == "project":
        return RouteDecision("project", 0.9, "project focus sticky")

    if re.search(
        r"\b(stay in|keep|switch to) (companion|conversation|planner|oac|project|deep[- ]?research)\b",
        text,
        re.I,
    ):
        picked_match = re.search(
            r"\b(companion|conversation|planner|oac|project|deep[- ]?research)\b", text, re.I
        )
        if picked_match:
            picked = picked_match.group(1).lower().replace(" ", "_").replace("-", "_")
            if picked in {"companion", "conversation", "planner", "oac"}:
                picked = DEFAULT_MODE_ID
            if picked.startswith("deep"):
                picked = "deep_research"
            return RouteDecision(picked, 0.99, "explicit request")

    if _DEEP_RESEARCH.search(text):
        return RouteDecision("deep_research", 0.95, "deep research request")

    if _OPEN_PROJECT.search(text) or _CREATE_PROJECT.search(text):
        return RouteDecision(current, 0.55, "project tool request")

    if current == "deep_research" and len(text) < 80 and not re.match(
        r"^\s*(hi|hey|hello|thanks|never ?mind)\b", text, re.I
    ):
        return RouteDecision("deep_research", 0.7, "sticky research follow-up")

    return RouteDecision(current, 0.2, "stay")
