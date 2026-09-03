"""Hayden's calendar — JSON file in db/, shared by the UI and OAC tools."""

from __future__ import annotations

import calendar as pycal
import json
import re
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ainet.tools.fsutil import atomic_write_text

CALENDAR_FILE = "Calendar.json"
_LOCK = threading.RLock()
_MAX_OCCURRENCES = 400

REPEAT_NONE = "none"
REPEAT_DAILY = "daily"
REPEAT_WEEKLY = "weekly"
REPEAT_MONTHLY = "monthly"
REPEAT_YEARLY = "yearly"
REPEAT_CHOICES = (REPEAT_NONE, REPEAT_DAILY, REPEAT_WEEKLY, REPEAT_MONTHLY, REPEAT_YEARLY)

CATEGORIES: dict[str, str] = {
    "school": "#0c8f55",
    "work": "#3d6b8f",
    "personal": "#5c4d7a",
    "health": "#b7791f",
    "social": "#c45c7a",
    "other": "#6a726c",
}

_WEEKDAY_ALIAS: dict[str, int] = {
    "mo": 0,
    "mon": 0,
    "monday": 0,
    "tu": 1,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "we": 2,
    "wed": 2,
    "wednesday": 2,
    "th": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fr": 4,
    "fri": 4,
    "friday": 4,
    "sa": 5,
    "sat": 5,
    "saturday": 5,
    "su": 6,
    "sun": 6,
    "sunday": 6,
}
_WEEKDAY_ICS = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")

_SCHEDULE_RE = re.compile(
    r"\b(?:"
    r"calendar|schedule|agenda|appointment|appointments|"
    r"meeting|meetings|meetup|"
    r"what(?:'s|s| is) on|"
    r"what do i have|what have i got|"
    r"am i free|am i busy|i free|i busy|"
    r"plans? for|busy on|free on|"
    r"this week|next week|this weekend|this month|"
    r"tomorrow|tonight|"
    r"add (?:an? )?(?:event|meeting|appointment)|"
    r"put (?:it |that )?(?:on|in) (?:my )?calendar|"
    r"remind me"
    r")\b",
    re.I,
)
_CALENDAR_WRITE_RE = re.compile(
    r"(?:"
    r"\b(?:set|add|put|create|save|schedule|block)\b.{0,40}\b(?:calendar|event|appointment)\b"
    r"|"
    r"\b(?:on|in|to)\s+(?:my\s+)?calendar\b"
    r"|"
    r"\bcalendar\b.{0,48}\b(?:test|exam|quiz|midterm|final|meeting|appointment|event)\b"
    r"|"
    r"\b(?:i have|ive got|i've got|got)\s+a\s+(?:test|exam|quiz|midterm|final|meeting)\b"
    r".{0,80}\b(?:\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday|tomorrow|tonight|today)\b"
    r")",
    re.I | re.S,
)
_TIME_TOKEN_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b",
    re.I,
)
_DAY_WORD_RE = re.compile(
    r"\b(?:(next)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"today|tomorrow|tonight)\b",
    re.I,
)
# Class / lab / "when does X start" — not covered by generic schedule words alone.
_CLASS_SCHEDULE_RE = re.compile(
    r"(?:"
    r"\b(?:when|what time|what day)\b.{0,80}\b(?:"
    r"class|classes|lab|labs|lecture|lec|recitation|seminar|course|"
    r"meeting|starts?|begins?|ends?|test|exam|quiz|midterm|final"
    r")\b"
    r"|"
    r"\b(?:class|classes|lab|labs|lecture|lec|recitation|seminar|course|"
    r"test|exam|quiz|midterm|final)\b"
    r".{0,48}\b(?:start|starts|begin|begins|when|time|end|ends)\b"
    r"|"
    r"\b(?:my|the)\s+(?:[A-Za-z]{2,5}\s*[- ]?\d{2,5}\s+)?"
    r"(?:lab|labs|lecture|lec|class|classes|recitation|test|exam|quiz|midterm|final)\b"
    r"|"
    r"\b[A-Za-z]{2,5}\s*[- ]?\d{2,5}\b.{0,48}\b(?:"
    r"lab|labs|lecture|lec|class|start|starts|when|time|test|exam|quiz|midterm|final"
    r")\b"
    r"|"
    r"\b(?:test|exam|quiz|midterm|final)\b.{0,48}\b[A-Za-z]{2,5}\s*[- ]?\d{2,5}\b"
    r")",
    re.I | re.S,
)
_COURSE_CODE_RE = re.compile(r"\b([A-Za-z]{2,5})\s*[- ]?(\d{2,5})\b")
_EVENT_KIND_RE = re.compile(
    r"\b(test|exam|quiz|midterm|final|lab|lecture|lec|recitation|"
    r"dentist|doctor|interview|flight|shift|office)\b",
    re.I,
)
_WEEKDAY_ASK_RE = re.compile(
    r"\b(?:on |this |next )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.I,
)
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_SCHEDULE_Q_STOP = frozenset(
    {
        "hey",
        "bud",
        "buddy",
        "pal",
        "dude",
        "bro",
        "bruh",
        "man",
        "mate",
        "homie",
        "dawg",
        "fam",
        "chief",
        "boss",
        "friend",
        "hi",
        "hello",
        "yo",
        "please",
        "thanks",
        "thank",
        "when",
        "what",
        "whats",
        "where's",
        "wheres",
        "where",
        "how",
        "does",
        "do",
        "did",
        "is",
        "are",
        "the",
        "a",
        "an",
        "my",
        "me",
        "i",
        "im",
        "i'm",
        "you",
        "your",
        "start",
        "starts",
        "starting",
        "begin",
        "begins",
        "beginning",
        "end",
        "ends",
        "ending",
        "time",
        "times",
        "clock",
        "have",
        "got",
        "get",
        "tell",
        "know",
        "about",
        "for",
        "from",
        "with",
        "into",
        "onto",
        "on",
        "at",
        "of",
        "calendar",
        "schedule",
        "agenda",
        "appointment",
        "appointments",
        "meeting",
        "meetings",
        "class",
        "classes",
        "course",
        "courses",
        "event",
        "events",
        "stuff",
        "things",
        "anything",
        "something",
        "planned",
        "plans",
        "busy",
        "free",
        "this",
        "next",
        "week",
        "weeks",
        "today",
        "tonight",
        "tomorrow",
        "month",
        "weekend",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    }
)


def local_tz_name() -> str:
    try:
        tz = datetime.now().astimezone().tzinfo
        key = getattr(tz, "key", None)
        if key:
            return str(key)
        name = tz.tzname(datetime.now()) if tz is not None else ""
        if name:
            return str(name)
    except Exception:
        pass
    return "local"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _calendar_path(root: Path | str) -> Path:
    base = Path(root)
    exact = base / CALENDAR_FILE
    if exact.exists():
        return exact
    wanted = CALENDAR_FILE.casefold()
    try:
        for child in base.iterdir():
            if child.is_file() and child.name.casefold() == wanted:
                return child
    except OSError:
        pass
    return exact


def empty_doc() -> dict[str, Any]:
    return {
        "version": 1,
        "mutable_by": "code_only",
        "timezone": local_tz_name(),
        "events": [],
    }


def load_doc(root: Path | str) -> dict[str, Any]:
    path = _calendar_path(root)
    if not path.is_file():
        return empty_doc()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_doc()
    if not isinstance(raw, dict):
        return empty_doc()
    events = raw.get("events")
    if not isinstance(events, list):
        raw["events"] = []
    raw.setdefault("version", 1)
    raw.setdefault("mutable_by", "code_only")
    raw.setdefault("timezone", local_tz_name())
    return raw


def save_doc(root: Path | str, doc: dict[str, Any]) -> Path:
    path = _calendar_path(root)
    payload = {
        "version": int(doc.get("version") or 1),
        "mutable_by": str(doc.get("mutable_by") or "code_only"),
        "timezone": str(doc.get("timezone") or local_tz_name()),
        "events": list(doc.get("events") or []),
    }
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    return path


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    if "T" in text:
        text = text.split("T", 1)[0]
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        try:
            return datetime.fromisoformat(text + "T00:00:00")
        except ValueError:
            return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _fmt_date(d: date) -> str:
    return d.isoformat()


def _fmt_dt(dt: datetime, *, all_day: bool) -> str:
    if all_day:
        return dt.date().isoformat()
    return dt.replace(microsecond=0).isoformat(timespec="minutes")


def _weekday_num(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        n = int(value)
        return n if 0 <= n <= 6 else None
    key = str(value).strip().casefold()
    return _WEEKDAY_ALIAS.get(key)


def _parse_weekdays(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parts = re.split(r"[\s,|]+", value.strip())
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [value]
    out: list[int] = []
    seen: set[int] = set()
    for part in parts:
        num = _weekday_num(part)
        if num is None or num in seen:
            continue
        seen.add(num)
        out.append(num)
    return out


def _weekdays_ics(nums: list[int]) -> list[str]:
    return [_WEEKDAY_ICS[n] for n in nums if 0 <= n <= 6]


def _parse_reminders(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parts = re.split(r"[\s,|]+", value.strip())
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [value]
    out: list[int] = []
    seen: set[int] = set()
    for part in parts:
        try:
            n = int(part)
        except (TypeError, ValueError):
            continue
        if n < 0 or n in seen:
            continue
        seen.add(n)
        out.append(n)
    out.sort()
    return out[:8]


def _new_id(title: str) -> str:
    slug = _SLUG_RE.sub("-", (title or "").casefold()).strip("-")[:24].strip("-")
    suffix = uuid.uuid4().hex[:8]
    return f"{slug}-{suffix}" if slug else suffix


def _add_months(d: date, months: int) -> date:
    month_index = d.year * 12 + (d.month - 1) + months
    year, month0 = divmod(month_index, 12)
    month = month0 + 1
    last = pycal.monthrange(year, month)[1]
    return date(year, month, min(d.day, last))


def _normalize_repeat(value: Any) -> str:
    text = str(value or REPEAT_NONE).strip().casefold()
    aliases = {
        "": REPEAT_NONE,
        "no": REPEAT_NONE,
        "off": REPEAT_NONE,
        "never": REPEAT_NONE,
        "once": REPEAT_NONE,
        "rrule": REPEAT_NONE,
        "day": REPEAT_DAILY,
        "week": REPEAT_WEEKLY,
        "month": REPEAT_MONTHLY,
        "year": REPEAT_YEARLY,
        "annually": REPEAT_YEARLY,
        "annual": REPEAT_YEARLY,
    }
    text = aliases.get(text, text)
    return text if text in REPEAT_CHOICES else REPEAT_NONE


def _normalize_category(value: Any) -> str:
    key = str(value or "other").strip().casefold()
    return key if key in CATEGORIES else "other"


def normalize_event(raw: dict[str, Any] | None, *, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    src = dict(defaults or {})
    if isinstance(raw, dict):
        src.update(raw)
    title = str(src.get("title") or "").strip()
    if not title:
        raise ValueError("title is required")
    all_day = bool(src.get("all_day"))
    start_dt = _parse_dt(src.get("start"))
    if start_dt is None:
        raise ValueError("start must be a date or datetime (YYYY-MM-DD or ISO)")
    if all_day:
        start_dt = datetime.combine(start_dt.date(), datetime.min.time())

    end_dt = _parse_dt(src.get("end"))
    duration = src.get("duration_minutes")
    duration_n: int | None = None
    if duration not in (None, ""):
        try:
            duration_n = max(1, int(duration))
        except (TypeError, ValueError) as exc:
            raise ValueError("duration_minutes must be an integer") from exc
    if end_dt is None and duration_n is None:
        duration_n = 1440 if all_day else 60
        end_dt = start_dt + timedelta(minutes=duration_n)
    elif end_dt is None:
        end_dt = start_dt + timedelta(minutes=int(duration_n or 60))
    elif duration_n is None:
        delta = end_dt - start_dt
        duration_n = max(1, int(delta.total_seconds() // 60) or (1440 if all_day else 60))
    if end_dt <= start_dt:
        end_dt = start_dt + timedelta(minutes=duration_n or (1440 if all_day else 60))
        duration_n = max(1, int((end_dt - start_dt).total_seconds() // 60))

    repeat = _normalize_repeat(src.get("repeat") or src.get("rrule"))
    try:
        interval = max(1, int(src.get("interval") or 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("interval must be a positive integer") from exc
    until = _parse_date(src.get("until") or src.get("repeat_until"))
    byweekday = _parse_weekdays(src.get("byweekday") or src.get("weekdays"))
    if repeat == REPEAT_WEEKLY and not byweekday:
        byweekday = [start_dt.weekday()]
    if repeat == REPEAT_NONE:
        interval = 1
        until = None
        byweekday = []

    category = _normalize_category(src.get("category") or src.get("tag"))
    color = str(src.get("color") or "").strip() or CATEGORIES[category]
    event_id = str(src.get("id") or "").strip() or _new_id(title)
    created = str(src.get("created") or "").strip() or _now_iso()
    return {
        "id": event_id,
        "title": title,
        "start": _fmt_dt(start_dt, all_day=all_day),
        "end": _fmt_dt(end_dt, all_day=all_day),
        "duration_minutes": int(duration_n or 60),
        "all_day": all_day,
        "timezone": str(src.get("timezone") or local_tz_name()).strip() or local_tz_name(),
        "repeat": repeat,
        "interval": interval,
        "until": until.isoformat() if until else "",
        "byweekday": _weekdays_ics(byweekday),
        "location": str(src.get("location") or "").strip(),
        "notes": str(src.get("notes") or "").strip(),
        "category": category,
        "color": color,
        "reminder_minutes": _parse_reminders(src.get("reminder_minutes") or src.get("reminders")),
        "cancelled": bool(src.get("cancelled")),
        "created": created,
        "updated": _now_iso(),
    }


def _event_start(event: dict[str, Any]) -> datetime:
    parsed = _parse_dt(event.get("start"))
    if parsed is None:
        raise ValueError("event is missing a valid start")
    if event.get("all_day"):
        return datetime.combine(parsed.date(), datetime.min.time())
    return parsed


def _event_end(event: dict[str, Any], start: datetime) -> datetime:
    parsed = _parse_dt(event.get("end"))
    if parsed is not None:
        if event.get("all_day"):
            parsed = datetime.combine(parsed.date(), datetime.min.time())
        if parsed > start:
            return parsed
    minutes = int(event.get("duration_minutes") or (1440 if event.get("all_day") else 60))
    return start + timedelta(minutes=max(1, minutes))


def _until_date(event: dict[str, Any]) -> date | None:
    return _parse_date(event.get("until"))


def expand_event(
    event: dict[str, Any],
    range_start: date,
    range_end: date,
) -> list[dict[str, Any]]:
    """Expand one stored event into occurrences overlapping [range_start, range_end]."""
    if event.get("cancelled"):
        return []
    try:
        series_start = _event_start(event)
    except ValueError:
        return []
    duration = _event_end(event, series_start) - series_start
    if duration.total_seconds() <= 0:
        duration = timedelta(minutes=60)
    until = _until_date(event)
    repeat = _normalize_repeat(event.get("repeat"))
    try:
        interval = max(1, int(event.get("interval") or 1))
    except (TypeError, ValueError):
        interval = 1
    byweekday = _parse_weekdays(event.get("byweekday"))
    if repeat == REPEAT_WEEKLY and not byweekday:
        byweekday = [series_start.weekday()]

    hard_end = range_end
    if until is not None:
        hard_end = min(hard_end, until)
    if hard_end < range_start:
        return []

    out: list[dict[str, Any]] = []

    def _emit(occ_start: datetime) -> None:
        if len(out) >= _MAX_OCCURRENCES:
            return
        occ_end = occ_start + duration
        occ_day = occ_start.date()
        last_day = (occ_end - timedelta(minutes=1)).date() if occ_end > occ_start else occ_day
        if last_day < range_start or occ_day > hard_end:
            return
        if until is not None and occ_day > until:
            return
        all_day = bool(event.get("all_day"))
        row = {
            "id": event.get("id"),
            "title": event.get("title"),
            "start": _fmt_dt(occ_start, all_day=all_day),
            "end": _fmt_dt(occ_end, all_day=all_day),
            "occurrence_date": occ_day.isoformat(),
            "all_day": all_day,
            "timezone": event.get("timezone") or local_tz_name(),
            "location": event.get("location") or "",
            "notes": event.get("notes") or "",
            "category": event.get("category") or "other",
            "color": event.get("color") or CATEGORIES["other"],
            "repeat": repeat,
            "reminder_minutes": list(event.get("reminder_minutes") or []),
            "cancelled": False,
        }
        out.append(row)

    if repeat == REPEAT_NONE:
        _emit(series_start)
        return out

    cursor = series_start
    guard = 0
    if repeat == REPEAT_DAILY:
        if cursor.date() < range_start:
            skip = (range_start - cursor.date()).days
            steps = skip // interval
            cursor = cursor + timedelta(days=steps * interval)
            while cursor.date() < range_start:
                cursor = cursor + timedelta(days=interval)
        while cursor.date() <= hard_end and guard < _MAX_OCCURRENCES:
            _emit(cursor)
            cursor = cursor + timedelta(days=interval)
            guard += 1
        return out

    if repeat == REPEAT_WEEKLY:
        week0 = series_start.date() - timedelta(days=series_start.weekday())
        days = byweekday or [series_start.weekday()]
        week = week0
        if week + timedelta(days=6) < range_start:
            delta_weeks = ((range_start - week).days // 7 // interval) * interval
            week = week + timedelta(weeks=delta_weeks)
        while week <= hard_end and guard < _MAX_OCCURRENCES:
            if ((week - week0).days // 7) % interval == 0:
                for wd in days:
                    occ = datetime.combine(week + timedelta(days=wd), cursor.time())
                    if occ.date() < series_start.date():
                        continue
                    _emit(occ)
                    guard += 1
            week = week + timedelta(weeks=1)
        return out

    if repeat == REPEAT_MONTHLY:
        cursor_date = series_start.date()
        while cursor_date < range_start and guard < 240:
            cursor_date = _add_months(cursor_date, interval)
            guard += 1
        guard = 0
        while cursor_date <= hard_end and guard < _MAX_OCCURRENCES:
            _emit(datetime.combine(cursor_date, series_start.time()))
            cursor_date = _add_months(cursor_date, interval)
            guard += 1
        return out

    if repeat == REPEAT_YEARLY:
        cursor_date = series_start.date()
        while cursor_date < range_start and guard < 80:
            cursor_date = _add_months(cursor_date, 12 * interval)
            guard += 1
        guard = 0
        while cursor_date <= hard_end and guard < _MAX_OCCURRENCES:
            _emit(datetime.combine(cursor_date, series_start.time()))
            cursor_date = _add_months(cursor_date, 12 * interval)
            guard += 1
        return out

    _emit(series_start)
    return out


def _text_blob(event: dict[str, Any]) -> str:
    return " ".join(
        str(event.get(k) or "")
        for k in ("title", "location", "notes", "category")
    ).casefold()


def _query_tokens(q: str) -> list[str]:
    """Meaningful search tokens; drop schedule filler like 'classes' / 'today's stuff'."""
    kept: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9+#./'-]+", (q or "").casefold()):
        w = raw.replace("'", "")
        if len(w) < 2:
            continue
        if w in _SCHEDULE_Q_STOP or w.rstrip("s") in _SCHEDULE_Q_STOP:
            continue
        kept.append(w)
    return kept


def _matches_q(event: dict[str, Any], q: str) -> bool:
    """All query tokens must appear (AND). Keeps 'ME 365 lab' from matching every lab."""
    words = _query_tokens(q)
    if not words:
        return True
    blob = _text_blob(event)
    return all(word in blob for word in words)


def _course_in_title(title: str, dept: str, num: str) -> bool:
    low = (title or "").casefold()
    d = (dept or "").casefold()
    n = (num or "").casefold()
    if not n:
        return False
    if d and re.search(rf"\b{re.escape(d)}\s*[- ]?{re.escape(n)}\b", low):
        return True
    # ME 36500 should match ask for ME 365; MA 265 matches MA265.
    return bool(re.search(rf"\b{re.escape(d)}\s*[- ]?{re.escape(n)}\d*\b", low)) if d else (
        bool(re.search(rf"\b[a-z]{{2,5}}\s*[- ]?{re.escape(n)}\d*\b", low))
    )


def query_events(
    root: Path | str,
    *,
    start: str = "",
    end: str = "",
    q: str = "",
    upcoming: int | None = None,
    limit: int = 40,
    include_cancelled: bool = False,
) -> dict[str, Any]:
    today = date.today()
    start_d = _parse_date(start) or today
    end_d = _parse_date(end)
    upcoming_n = 0
    if upcoming not in (None, ""):
        try:
            upcoming_n = max(0, int(upcoming))
        except (TypeError, ValueError):
            upcoming_n = 0
    if end_d is None:
        # One day unless they asked for upcoming / a wider range.
        end_d = start_d + timedelta(days=60) if upcoming_n else start_d
    if end_d < start_d:
        start_d, end_d = end_d, start_d
    limit_n = max(1, min(int(limit or 40), 80))

    with _LOCK:
        doc = load_doc(root)
        masters = [e for e in doc.get("events") or [] if isinstance(e, dict)]
        if not include_cancelled:
            masters = [e for e in masters if not e.get("cancelled")]

    expand_end = end_d
    if upcoming_n:
        expand_end = max(expand_end, today + timedelta(days=90))

    def _collect(match_fn) -> list[dict[str, Any]]:
        occurrences: list[dict[str, Any]] = []
        for event in masters:
            if not match_fn(event):
                continue
            occurrences.extend(
                expand_event(event, start_d if not upcoming_n else today, expand_end)
            )
        occurrences.sort(
            key=lambda row: (str(row.get("start") or ""), str(row.get("title") or ""))
        )
        return occurrences

    occurrences = _collect(lambda e: _matches_q(e, q))
    # Soft fallback: MA365 → ME 365 / MA 265 style typos, or number-only hits in titles.
    if q and not occurrences:
        course = _COURSE_CODE_RE.search(q)
        if course:
            dept, num = course.group(1), course.group(2)
            want_lab = "lab" in q.casefold()

            def _soft(event: dict[str, Any]) -> bool:
                title = str(event.get("title") or "")
                if not _course_in_title(title, dept, num) and not _course_in_title(
                    title, "", num
                ):
                    return False
                if want_lab and "lab" not in title.casefold():
                    return False
                return True

            occurrences = _collect(_soft)
            # Prefer same-dept-ish titles first when multiple depts share a number.
            if occurrences and dept:
                same = [
                    r
                    for r in occurrences
                    if _course_in_title(str(r.get("title") or ""), dept, num)
                ]
                if same:
                    occurrences = same

    if upcoming_n:
        now = datetime.now().replace(microsecond=0)
        kept: list[dict[str, Any]] = []
        for row in occurrences:
            occ = _parse_dt(row.get("end") or row.get("start"))
            if occ is None:
                continue
            if occ >= now:
                kept.append(row)
            if len(kept) >= upcoming_n:
                break
        occurrences = kept
    else:
        in_range: list[dict[str, Any]] = []
        for row in occurrences:
            day = _parse_date(row.get("occurrence_date") or row.get("start"))
            if day is None or day < start_d or day > end_d:
                continue
            in_range.append(row)
        occurrences = in_range[:limit_n]

    return {
        "ok": True,
        "path": CALENDAR_FILE,
        "timezone": str(doc.get("timezone") or local_tz_name()),
        "start": start_d.isoformat(),
        "end": end_d.isoformat(),
        "count": len(occurrences),
        "events": occurrences,
        "digest": digest_events(occurrences, limit=40),
        "masters": [
            e
            for e in masters
            if include_cancelled or not e.get("cancelled")
        ],
    }


def _clock(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")


def format_occurrence(row: dict[str, Any], *, date_prefix: bool = True) -> str:
    start = _parse_dt(row.get("start"))
    end = _parse_dt(row.get("end"))
    title = str(row.get("title") or "Untitled").strip()
    loc = str(row.get("location") or "").strip()
    if start is None:
        when = str(row.get("start") or "")
    elif row.get("all_day"):
        when = "all day" if not date_prefix else f"{start.strftime('%a %b')} {start.day} all day"
    else:
        when = _clock(start)
        if end and end.date() == start.date():
            when += f"–{_clock(end)}"
        if date_prefix:
            when = f"{start.strftime('%a %b')} {start.day} {when}"
    bits = [when, title]
    if loc:
        bits.append(loc)
    return " · ".join(bits)


def digest_events(rows: list[dict[str, Any]], *, limit: int = 16) -> str:
    slice_rows = rows[:limit]
    if not slice_rows:
        return ""
    days = {
        (_parse_date(r.get("occurrence_date") or r.get("start")) or date.min)
        for r in slice_rows
    }
    one_day = len(days) <= 1
    lines = [format_occurrence(row, date_prefix=not one_day) for row in slice_rows]
    return "CALENDAR\n" + "\n".join(lines)


def spoken_schedule(
    rows: list[dict[str, Any]],
    *,
    about: str = "",
    limit: int = 40,
) -> str:
    """Short spoken list for OAC — one line per occurrence, no tables."""
    slice_rows = list(rows or [])[:limit]
    label = (about or "that day").strip()
    low = label.casefold()
    if low in {"today's events", "today"}:
        label = "Today"
    elif low in {"tomorrow"}:
        label = "Tomorrow"
    elif low.endswith("'s events"):
        label = label[: -len("'s events")]
    if label:
        label = label[0].upper() + label[1:]
    if not slice_rows:
        return f"Nothing on the calendar for {label or 'that'}."
    days = {
        (_parse_date(r.get("occurrence_date") or r.get("start")) or date.min)
        for r in slice_rows
    }
    one_day = len(days) <= 1
    weekday_names = {
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "today",
        "tomorrow",
    }
    day_header = low in weekday_names or low.startswith("today")
    lines = [
        format_occurrence(row, date_prefix=not (one_day and day_header))
        for row in slice_rows
    ]
    return f"{label}:\n" + "\n".join(lines)


def looks_like_schedule_ask(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if looks_like_calendar_write(raw):
        return False
    if _SCHEDULE_RE.search(raw):
        return True
    if _CLASS_SCHEDULE_RE.search(raw):
        return True
    return bool(_WEEKDAY_ASK_RE.search(raw))


def looks_like_calendar_write(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    return bool(_CALENDAR_WRITE_RE.search(raw))


def _parse_clock(hour: str, minute: str | None, ampm: str) -> tuple[int, int]:
    h = int(hour)
    m = int(minute or "0")
    ap = (ampm or "").replace(".", "").casefold()
    if ap.startswith("p") and h < 12:
        h += 12
    elif ap.startswith("a") and h == 12:
        h = 0
    return h, m


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


_WEEKDAY_CANON = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def _closest_weekday(token: str) -> str | None:
    """Map near-misses like 'mnonday' → 'monday'."""
    t = (token or "").strip().casefold()
    if not t:
        return None
    if t in _WEEKDAY_ALIAS:
        # Prefer full day names over mo/tu aliases for repair.
        for name in _WEEKDAY_CANON:
            if _WEEKDAY_ALIAS[name] == _WEEKDAY_ALIAS[t]:
                return name
        return t
    if len(t) < 4 or len(t) > 12:
        return None
    best: str | None = None
    best_d = 99
    for name in _WEEKDAY_CANON:
        if t[0] != name[0]:
            continue
        if abs(len(t) - len(name)) > 2:
            continue
        d = _edit_distance(t, name)
        if d < best_d:
            best_d = d
            best = name
    # Allow 1–2 typos on weekday names Hayden types often.
    if best is not None and best_d <= 2:
        return best
    return None


def _repair_weekday_typos(text: str) -> str:
    """Rewrite mistyped weekday words after next/this/on, or near-exact day names."""

    def _sub(match: re.Match[str]) -> str:
        prefix = match.group(1) or ""
        word = match.group(2)
        hit = _closest_weekday(word)
        if not hit:
            return match.group(0)
        # Without a day cue, only accept very close matches (1 edit).
        if not prefix and _edit_distance(word.casefold(), hit) > 1:
            return match.group(0)
        return f"{prefix}{hit}"

    return re.sub(
        r"\b((?:next|this|on)\s+)?([A-Za-z]{4,12})\b",
        _sub,
        text or "",
        flags=re.I,
    )


def _resolve_day_word(match: re.Match[str], *, today: date | None = None) -> date:
    today = today or date.today()
    word = match.group(2).casefold()
    nxt = bool(match.group(1))
    if word == "today" or word == "tonight":
        return today
    if word == "tomorrow":
        return today + timedelta(days=1)
    want = _WEEKDAY_ALIAS[word]
    delta = (want - today.weekday()) % 7
    if nxt:
        if delta == 0:
            delta = 7
    elif delta == 0 and "tonight" not in word:
        # Bare "monday" on Monday → today; "next monday" already handled.
        pass
    return today + timedelta(days=delta)


def parse_event_from_text(text: str, *, today: date | None = None) -> dict[str, Any] | None:
    """Best-effort natural-language event from Hayden's words. Returns None if too vague."""
    raw = " ".join((text or "").split())
    if not raw:
        return None
    raw = _repair_weekday_typos(raw)
    today = today or date.today()
    times = list(_TIME_TOKEN_RE.finditer(raw))
    day_m = _DAY_WORD_RE.search(raw)
    if not times and not day_m and not looks_like_calendar_write(raw):
        return None

    day = today
    if day_m:
        day = _resolve_day_word(day_m, today=today)

    start_dt: datetime | None = None
    end_dt: datetime | None = None
    all_day = False
    if times:
        h, m = _parse_clock(times[0].group(1), times[0].group(2), times[0].group(3))
        start_dt = datetime.combine(day, datetime.min.time()).replace(hour=h, minute=m)
        if len(times) >= 2:
            h2, m2 = _parse_clock(times[1].group(1), times[1].group(2), times[1].group(3))
            end_dt = datetime.combine(day, datetime.min.time()).replace(hour=h2, minute=m2)
            if end_dt <= start_dt:
                end_dt = start_dt + timedelta(hours=1)
        else:
            end_dt = start_dt + timedelta(hours=1)
    elif day_m:
        all_day = True
        start_dt = datetime.combine(day, datetime.min.time())
        end_dt = start_dt + timedelta(days=1)
    else:
        return None

    course = _COURSE_CODE_RE.search(raw)
    course_label = ""
    if course:
        course_label = f"{course.group(1).upper()} {course.group(2)}"

    title = ""
    kind = re.search(
        r"\b(test|exam|quiz|midterm|final|meeting|appointment|office hours|"
        r"lab|lecture|dentist|doctor|interview|flight|shift)\b",
        raw,
        re.I,
    )
    if kind and course_label:
        title = f"{course_label} {kind.group(1).title()}"
    elif kind:
        word = kind.group(1)
        # Keep a short lead-in noun phrase when present ("dentist appointment").
        lead = re.search(
            rf"\b([A-Za-z][A-Za-z0-9]*(?:\s+[A-Za-z][A-Za-z0-9]*){{0,2}})\s+{re.escape(word)}\b",
            raw,
            re.I,
        )
        if lead and lead.group(1).casefold() not in {"a", "an", "the", "my"}:
            title = f"{lead.group(1).title()} {word.title()}"
        else:
            title = word.title()
    elif course_label:
        title = course_label
    else:
        # Strip command framing and keep a short remainder as the title.
        scrub = re.sub(
            r"^\s*(?:hey\s+\w+\s*,?\s*)?(?:set|add|put|create|save|schedule)\s+"
            r"(?:(?:on|in|to)\s+)?(?:my\s+)?calendar\s*(?:that\s+)?(?:i\s+have\s+)?",
            "",
            raw,
            flags=re.I,
        )
        scrub = _TIME_TOKEN_RE.sub(" ", scrub)
        scrub = _DAY_WORD_RE.sub(" ", scrub)
        scrub = re.sub(r"\b(?:to|from|at|in|on|for|the|a|an|pm|am)\b", " ", scrub, flags=re.I)
        scrub = re.sub(r"\s+", " ", scrub).strip(" .,!")
        title = scrub[:80] if scrub else "Event"
    if not title:
        title = "Event"

    location = ""
    loc_m = re.search(r"\b(?:at|in|@)\s+([A-Za-z0-9][A-Za-z0-9 \-]{1,40})\b", raw, re.I)
    if loc_m:
        cand = loc_m.group(1).strip()
        # "in ma265" is usually the course, not a building — keep as location only if
        # it doesn't look like a bare course code we already used in the title.
        if course and cand.casefold().replace(" ", "") == (
            course.group(1) + course.group(2)
        ).casefold():
            location = course_label
        else:
            location = cand

    payload: dict[str, Any] = {
        "title": title,
        "start": _fmt_dt(start_dt, all_day=all_day),
        "end": _fmt_dt(end_dt, all_day=all_day),
        "all_day": all_day,
        "category": "school" if course_label or kind and kind.group(1).casefold() in {
            "test", "exam", "quiz", "midterm", "final", "lab", "lecture"
        } else "other",
        "location": location,
        "notes": "",
        "repeat": REPEAT_NONE,
    }
    return payload


def _schedule_search_terms(text: str) -> tuple[str, str]:
    """Build (q, about) from course codes, tests, labs — not greetings or 'stuff'."""
    raw = _repair_weekday_typos(text or "")
    low = raw.casefold()
    bits: list[str] = []
    about_bits: list[str] = []
    course = _COURSE_CODE_RE.search(raw)
    if course:
        dept, num = course.group(1), course.group(2)
        bits.extend([dept, num])
        about_bits.append(f"{dept.upper()} {num}")
    kind = _EVENT_KIND_RE.search(raw)
    if kind:
        word = kind.group(1).casefold()
        if word in {"lec", "lecture"}:
            bits.append("lec")
            about_bits.append("lec")
        elif word in {"labs", "lab"}:
            bits.append("lab")
            about_bits.append("lab")
        else:
            bits.append(word)
            about_bits.append(word)
    elif re.search(r"\blabs?\b", low):
        bits.append("lab")
        about_bits.append("lab")
    elif re.search(r"\blec(?:ture)?s?\b", low):
        bits.append("lec")
        about_bits.append("lec")
    elif re.search(r"\brecitations?\b", low):
        bits.append("recitation")
        about_bits.append("recitation")
    if bits:
        q = " ".join(bits)
        about = " ".join(about_bits) if about_bits else q
        return q, about

    words = _query_tokens(raw)
    nums = re.findall(r"\b\d{2,5}\b", raw)
    for n in nums:
        if n not in words:
            words.insert(0, n)
    # Ignore leftover chatter (pal, stuff) — only keep a real-looking event name.
    if not words:
        return "", ""
    if len(words) == 1 and len(words[0]) < 4:
        return "", ""
    q = " ".join(words[:6])
    return q, q


def infer_schedule_query(text: str) -> dict[str, Any]:
    """Guess a range / search from Hayden's wording, including 'today's stuff'."""
    raw = _repair_weekday_typos(text or "")
    today = date.today()
    low = raw.casefold()
    start = today
    end = today + timedelta(days=13)
    about = "upcoming events"
    q = ""
    upcoming = 0
    targeted = False
    date_explicit = False

    if re.search(r"\b(today|tonight)(?:'s)?\b", low):
        start, end, about = today, today, "today's events"
        date_explicit = True
    elif re.search(r"\btomorrow(?:'s)?\b", low):
        day = today + timedelta(days=1)
        start, end, about = day, day, "tomorrow"
        date_explicit = True
    elif re.search(r"\bthis weekend\b", low):
        saturday = today + timedelta(days=(5 - today.weekday()) % 7)
        start, end, about = saturday, saturday + timedelta(days=1), "this weekend"
        date_explicit = True
    elif re.search(r"\bnext week\b", low):
        monday = today - timedelta(days=today.weekday()) + timedelta(days=7)
        start, end, about = monday, monday + timedelta(days=6), "next week's events"
        date_explicit = True
    elif re.search(r"\bthis week\b", low):
        monday = today - timedelta(days=today.weekday())
        start, end, about = monday, monday + timedelta(days=6), "this week's events"
        date_explicit = True
    elif re.search(r"\bthis month\b", low):
        start = today.replace(day=1)
        end = _add_months(start, 1) - timedelta(days=1)
        about = "this month"
        date_explicit = True

    weekday = _WEEKDAY_ASK_RE.search(raw)
    if weekday and not re.search(r"\b(?:this|next) week\b", low):
        want = _WEEKDAY_ALIAS[weekday.group(1).casefold()]
        delta = (want - today.weekday()) % 7
        if re.search(r"\bnext\s+" + re.escape(weekday.group(1)), low, re.I):
            if delta == 0:
                delta = 7
        day = today + timedelta(days=delta)
        start, end, about = day, day, day.strftime("%A")
        date_explicit = True

    q, q_about = _schedule_search_terms(raw)
    if q:
        targeted = True
        if date_explicit:
            about = f"{q_about} · {about}" if q_about else about
        else:
            about = q_about or about
            # Named class/test with no day — look ahead, don't dump filler into q only.
            upcoming = 12
            end = max(end, today + timedelta(days=120))
    elif not date_explicit:
        # Bare "what's on the calendar" — upcoming stretch, not a random leftover word.
        pass
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "q": q,
        "about": about,
        "upcoming": upcoming,
        "targeted": targeted,
        "date_explicit": date_explicit,
    }


def prefetch_calendar_context(root: Path | str, user_text: str) -> dict[str, Any]:
    # Adding/updating an event should not dump unrelated class times into context.
    if looks_like_calendar_write(user_text):
        return {
            "ok": True,
            "digest": "",
            "count": 0,
            "about": "new calendar event",
            "query": {"start": "", "end": "", "q": ""},
            "write": True,
        }
    inferred = infer_schedule_query(user_text)
    q = str(inferred.get("q") or "").strip()
    upcoming_n = int(inferred.get("upcoming") or 0)
    if q:
        # Always filter by terms for named class/lab asks (don't dump the whole week).
        result = query_events(
            root,
            start=str(inferred.get("start") or ""),
            end=str(inferred.get("end") or ""),
            q=q,
            upcoming=upcoming_n or 12,
            limit=20,
        )
        rows = list(result.get("events") or [])
        if not rows:
            result = query_events(
                root,
                start=str(inferred.get("start") or ""),
                end=str(inferred.get("end") or ""),
                q=q,
                limit=20,
            )
            rows = list(result.get("events") or [])
    else:
        result = query_events(
            root,
            start=str(inferred.get("start") or ""),
            end=str(inferred.get("end") or ""),
            q="",
            upcoming=8 if inferred.get("about") == "upcoming events" else 0,
            limit=20,
        )
        rows = list(result.get("events") or [])
    digest = digest_events(rows)
    return {
        "ok": bool(digest),
        "digest": digest,
        "count": len(rows),
        "about": str(inferred.get("about") or "upcoming events"),
        "query": {
            "start": result.get("start"),
            "end": result.get("end"),
            "q": inferred.get("q") or "",
        },
    }


def add_event(root: Path | str, payload: dict[str, Any]) -> dict[str, Any]:
    event = normalize_event(payload)
    with _LOCK:
        doc = load_doc(root)
        events = [e for e in doc.get("events") or [] if isinstance(e, dict)]
        if any(str(e.get("id") or "") == event["id"] for e in events):
            event["id"] = _new_id(event["title"])
        events.append(event)
        doc["events"] = events
        save_doc(root, doc)
    return {"ok": True, "event": event, "path": CALENDAR_FILE}


def update_event(root: Path | str, payload: dict[str, Any]) -> dict[str, Any]:
    event_id = str((payload or {}).get("id") or "").strip()
    if not event_id:
        raise ValueError("id is required")
    with _LOCK:
        doc = load_doc(root)
        events = [e for e in doc.get("events") or [] if isinstance(e, dict)]
        found = None
        for i, existing in enumerate(events):
            if str(existing.get("id") or "") == event_id:
                found = i
                break
        if found is None:
            return {"ok": False, "error": f"No calendar event {event_id!r}"}
        merged = dict(events[found])
        for key, value in (payload or {}).items():
            if key in {"id", "about", "created"}:
                continue
            if value is None:
                continue
            merged[key] = value
        event = normalize_event(merged)
        event["id"] = event_id
        event["created"] = str(events[found].get("created") or event["created"])
        events[found] = event
        doc["events"] = events
        save_doc(root, doc)
    return {"ok": True, "event": event, "path": CALENDAR_FILE}


def cancel_event(root: Path | str, event_id: str, *, delete: bool = False) -> dict[str, Any]:
    event_id = str(event_id or "").strip()
    if not event_id:
        raise ValueError("id is required")
    with _LOCK:
        doc = load_doc(root)
        events = [e for e in doc.get("events") or [] if isinstance(e, dict)]
        found = None
        for i, existing in enumerate(events):
            if str(existing.get("id") or "") == event_id:
                found = i
                break
        if found is None:
            return {"ok": False, "error": f"No calendar event {event_id!r}"}
        removed = events[found]
        if delete:
            events.pop(found)
        else:
            removed = dict(removed)
            removed["cancelled"] = True
            removed["updated"] = _now_iso()
            events[found] = removed
        doc["events"] = events
        save_doc(root, doc)
    return {
        "ok": True,
        "id": event_id,
        "deleted": bool(delete),
        "cancelled": not delete,
        "event": removed,
        "path": CALENDAR_FILE,
    }


def month_payload(root: Path | str, year: int, month: int) -> dict[str, Any]:
    start = date(int(year), int(month), 1)
    end = _add_months(start, 1) - timedelta(days=1)
    with _LOCK:
        doc = load_doc(root)
        masters = [e for e in doc.get("events") or [] if isinstance(e, dict) and not e.get("cancelled")]
    occurrences: list[dict[str, Any]] = []
    for event in masters:
        occurrences.extend(expand_event(event, start, end))
    occurrences.sort(key=lambda row: (str(row.get("start") or ""), str(row.get("title") or "")))
    upcoming = query_events(root, upcoming=12, limit=12)
    return {
        "ok": True,
        "path": CALENDAR_FILE,
        "timezone": str(doc.get("timezone") or local_tz_name()),
        "year": start.year,
        "month": start.month,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "events": masters,
        "occurrences": occurrences,
        "upcoming": list(upcoming.get("events") or []),
        "categories": CATEGORIES,
    }
