"""Helpers for SOI filing: ephemeral discard + COP-name checks. Not a folder router."""

from __future__ import annotations

import re
from typing import Any

_EPHEMERAL = re.compile(
    r"^\s*("
    r"hi|hey|hello|heyo|yo|sup|hiya|hi bud|hey pal|bruh|"
    r"good (morning|afternoon|evening)|"
    r"thanks|thank you|thx|ty|"
    r"cool( go on)?|ok(ay)?|k+|kk|sure|yeah|yep|yup|bet|alright|got it|"
    r"go on|continue|keep going|sounds good|"
    r"bye|good ?night|gg|np|yw|lol( ok)?|lmao|nvm|nm"
    r")[.!?]*\s*$",
    re.I,
)


def is_ephemeral_text(text: str) -> bool:
    """True for greetings / acknowledgment-only turns — discard, do not file."""
    t = (text or "").strip()
    if not t:
        return True
    if len(t) < 80 and _EPHEMERAL.match(t):
        return True
    return False


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


def entry_user_text(entry: dict[str, Any]) -> str:
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    return str(details.get("user_text") or entry.get("summary") or "").strip()
