"""Deep Research vault under db/runtime/research/ — host-only except inspect_research."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ainet.tools.fsutil import atomic_write_text
from ainet.tools.ops import DatabaseTools

RESEARCH_ROOT = "runtime/research"
BRIEFS_DIR = f"{RESEARCH_ROOT}/briefs"
CURRENT_PATH = f"{RESEARCH_ROOT}/current.json"
_BODY_MAX = 12000
_SUMMARY_MAX = 480
_TITLE_MAX = 120
_FINDING_MAX = 12
_SOURCE_MAX = 16
_SLUG_MAX = 48


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _slugify(title: str) -> str:
    raw = re.sub(r"[^A-Za-z0-9]+", "-", (title or "").strip())
    raw = raw.strip("-") or "brief"
    if raw[0].isdigit():
        raw = "R-" + raw
    return raw[:_SLUG_MAX].rstrip("-") or "brief"


def _normalize_sources(raw: Any) -> list[dict[str, str]]:
    """Accept lists, dicts, JSON strings, and link/href aliases from the model."""
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text.startswith(("[", "{")):
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                # Bare URL or newline-separated URLs
                urls = re.findall(r"https?://[^\s\]\"'<>]+", text)
                return [
                    {"title": "", "url": u.rstrip(".,);"), "publisher": "", "year": "", "note": ""}
                    for u in urls[:_SOURCE_MAX]
                ]
        else:
            urls = re.findall(r"https?://[^\s\]\"'<>]+", text)
            return [
                {"title": "", "url": u.rstrip(".,);"), "publisher": "", "year": "", "note": ""}
                for u in urls[:_SOURCE_MAX]
            ]

    items: list[Any]
    if isinstance(raw, dict):
        # {"0": {...}, "1": {...}} or {"Title": "https://..."} or single source object
        if any(k in raw for k in ("url", "link", "href", "title")):
            items = [raw]
        else:
            items = list(raw.values())
    elif isinstance(raw, list):
        items = raw
    else:
        return []

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items[:_SOURCE_MAX]:
        if isinstance(item, str):
            url = item.strip()
            title = ""
            if not url.startswith(("http://", "https://")):
                found = re.search(r"https?://[^\s\]\"'<>]+", url)
                if found:
                    title = url[: found.start()].strip(" -|:;")
                    url = found.group(0).rstrip(".,);")
                else:
                    continue
            if url in seen:
                continue
            seen.add(url)
            out.append({"title": _clip(title, 160), "url": url, "publisher": "", "year": "", "note": ""})
            continue
        if not isinstance(item, dict):
            continue
        url = str(
            item.get("url")
            or item.get("link")
            or item.get("href")
            or item.get("source_url")
            or ""
        ).strip()
        if not url.startswith(("http://", "https://")):
            continue
        url = url.rstrip(".,);")
        if url in seen:
            continue
        seen.add(url)
        out.append(
            {
                "title": _clip(str(item.get("title") or item.get("name") or ""), 160),
                "url": url,
                "publisher": _clip(str(item.get("publisher") or item.get("source") or ""), 80),
                "year": _clip(str(item.get("year") or ""), 8),
                "note": _clip(str(item.get("note") or item.get("supports") or ""), 240),
            }
        )
    return out


def _sources_from_markdown(body: str) -> list[dict[str, str]]:
    """Pull [title](url) citations out of the brief body as a fallback."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for title, url in re.findall(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", body or ""):
        u = url.rstrip(".,);")
        if u in seen:
            continue
        seen.add(u)
        out.append(
            {
                "title": _clip(title, 160),
                "url": u,
                "publisher": "",
                "year": "",
                "note": "",
            }
        )
        if len(out) >= _SOURCE_MAX:
            break
    return out


def _normalize_findings(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw[:_FINDING_MAX]:
        text = _clip(str(item or ""), 280)
        if text:
            out.append(text)
    return out


def research_dir(db: DatabaseTools) -> Path:
    return db.paths.resolve(RESEARCH_ROOT)


def briefs_dir(db: DatabaseTools) -> Path:
    return db.paths.resolve(BRIEFS_DIR)


def _ensure_vault(db: DatabaseTools) -> Path:
    root = research_dir(db)
    briefs = briefs_dir(db)
    root.mkdir(parents=True, exist_ok=True)
    briefs.mkdir(parents=True, exist_ok=True)
    return briefs


def _unique_brief_path(db: DatabaseTools, slug: str) -> tuple[str, Path]:
    briefs = _ensure_vault(db)
    candidate = briefs / f"{slug}.json"
    if not candidate.exists():
        return f"{BRIEFS_DIR}/{slug}.json", candidate
    for i in range(2, 40):
        name = f"{slug}-{i}.json"
        candidate = briefs / name
        if not candidate.exists():
            return f"{BRIEFS_DIR}/{name}", candidate
    name = f"{slug}-{uuid.uuid4().hex[:6]}.json"
    return f"{BRIEFS_DIR}/{name}", briefs / name


def _write_json_host(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _set_current(db: DatabaseTools, meta: dict[str, Any]) -> None:
    _ensure_vault(db)
    _write_json_host(db.paths.resolve(CURRENT_PATH), meta)


def get_current_brief(db: DatabaseTools) -> dict[str, Any] | None:
    cur_path = db.paths.resolve(CURRENT_PATH)
    if not cur_path.is_file():
        return None
    try:
        meta = json.loads(cur_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    rel = str(meta.get("path") or "").strip()
    if not rel:
        return None
    return load_brief(db, path=rel) or None


def list_briefs(db: DatabaseTools, *, limit: int = 40) -> list[dict[str, Any]]:
    briefs = briefs_dir(db)
    if not briefs.is_dir():
        return []
    rows: list[tuple[float, dict[str, Any]]] = []
    for path in briefs.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        rel = f"{BRIEFS_DIR}/{path.name}"
        rows.append(
            (
                path.stat().st_mtime,
                {
                    "id": str(data.get("id") or path.stem),
                    "title": str(data.get("title") or path.stem),
                    "path": rel,
                    "created_at": str(data.get("created_at") or ""),
                    "summary": _clip(str(data.get("summary") or ""), 200),
                },
            )
        )
    rows.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in rows[: max(1, min(100, int(limit)))]]


def load_brief(
    db: DatabaseTools,
    *,
    brief_id: str = "",
    path: str = "",
) -> dict[str, Any] | None:
    target: Path | None = None
    rel = (path or "").replace("\\", "/").strip()
    if rel:
        if not rel.startswith(f"{BRIEFS_DIR}/") and not rel.startswith(f"{RESEARCH_ROOT}/"):
            # Allow bare filename
            if "/" not in rel:
                rel = f"{BRIEFS_DIR}/{rel if rel.endswith('.json') else rel + '.json'}"
            else:
                return None
        try:
            target = db.paths.resolve(rel)
        except Exception:
            return None
    elif brief_id:
        want = brief_id.strip()
        for row in list_briefs(db, limit=100):
            if row["id"] == want:
                try:
                    target = db.paths.resolve(row["path"])
                except Exception:
                    return None
                break
        if target is None:
            # Fallback: scan filenames containing id
            briefs = briefs_dir(db)
            if briefs.is_dir():
                for p in briefs.glob("*.json"):
                    try:
                        data = json.loads(p.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if isinstance(data, dict) and str(data.get("id") or "") == want:
                        target = p
                        break
    if target is None or not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    out = dict(data)
    out["path"] = db.paths.relative_of(target)
    return out


def inspect_research(
    db: DatabaseTools,
    *,
    brief_id: str = "",
    path: str = "",
    list_only: bool = False,
    limit: int = 20,
) -> dict[str, Any]:
    """AI-facing inspect tool — only way to read the research vault."""
    if list_only or (not brief_id and not path):
        items = list_briefs(db, limit=limit)
        return {
            "ok": True,
            "action": "list",
            "count": len(items),
            "briefs": items,
            "hint": "Pass brief_id or path to load one brief's full body.",
        }
    brief = load_brief(db, brief_id=brief_id, path=path)
    if not brief:
        return {
            "ok": False,
            "error": "Brief not found. Call inspect_research(list_only=true) first.",
        }
    return {"ok": True, "action": "read", "brief": brief}


def save_research(
    db: DatabaseTools,
    *,
    title: str,
    body: str,
    question: str = "",
    summary: str = "",
    key_findings: Any = None,
    sources: Any = None,
    url: str = "",
    link: str = "",
    href: str = "",
) -> dict[str, Any]:
    """Write a cited brief into the host-only research vault (not visible to SOI)."""
    title_text = _clip(title, _TITLE_MAX)
    if not title_text:
        raise ValueError("title is required")
    body_text = (body or "").strip()
    if not body_text:
        raise ValueError("body is required — markdown 1–2 pager with citations")
    if len(body_text) > _BODY_MAX:
        body_text = body_text[:_BODY_MAX].rstrip() + "…"

    sources_n = _normalize_sources(sources)
    # Models sometimes pass a lone url/link at the top level on the first try.
    for extra in (url, link, href):
        extra = str(extra or "").strip()
        if extra.startswith(("http://", "https://")):
            sources_n = _normalize_sources(list(sources_n) + [extra])
    if len(sources_n) < 2:
        # Fall back to markdown links already in the body.
        merged = {s["url"]: s for s in sources_n}
        for row in _sources_from_markdown(body_text):
            merged.setdefault(row["url"], row)
        sources_n = list(merged.values())[:_SOURCE_MAX]
    if len(sources_n) < 2:
        raise ValueError(
            "sources must include at least 2 http(s) citations. "
            "Pass sources=[{title,url}, ...] (link/href also ok)."
        )

    rel, abs_path = _unique_brief_path(db, _slugify(title_text))
    now = _utc_now()
    brief_id = uuid.uuid4().hex[:12]
    question_text = _clip(question, 400) or title_text
    payload = {
        "kind": "research_brief",
        "id": brief_id,
        "title": title_text,
        "question": question_text,
        "summary": _clip(summary, _SUMMARY_MAX) or _clip(body_text, _SUMMARY_MAX),
        "body": body_text,
        "key_findings": _normalize_findings(key_findings),
        "sources": sources_n,
        "created_at": now,
        "updated_at": now,
    }
    _write_json_host(abs_path, payload)
    meta = {
        "id": brief_id,
        "path": rel,
        "title": title_text,
        "updated_at": now,
    }
    _set_current(db, meta)
    return {
        "ok": True,
        "saved": True,
        "id": brief_id,
        "path": rel,
        "title": title_text,
        "source_count": len(sources_n),
        "preview": True,
        "hint": "Brief saved to the private research vault. UI Doc panel shows the preview. SOI cannot see this folder.",
    }
