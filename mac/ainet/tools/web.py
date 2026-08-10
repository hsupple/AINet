"""Internet tools: Brave Search + light URL fetch (stdlib / macOS-friendly)."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
_USER_AGENT = "AINet/1.0 (+local; Brave Search)"
_DEFAULT_SEARCH_COUNT = 5
_MAX_SEARCH_COUNT = 8
_TITLE_MAX = 120
_SNIPPET_MAX = 280
_FETCH_DEFAULT_CHARS = 4000
_FETCH_MAX_CHARS = 8000
_FETCH_BYTE_CAP = 500_000
_HTTP_TIMEOUT_S = 20.0


def _api_key() -> str:
    key = (
        os.environ.get("AINET_BRAVE_API_KEY")
        or os.environ.get("BRAVE_API_KEY")
        or ""
    ).strip()
    if not key:
        raise ValueError(
            "Missing Brave Search API key. Set BRAVE_API_KEY or AINET_BRAVE_API_KEY "
            "(see Rules.txt)."
        )
    return key


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _http_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = _HTTP_TIMEOUT_S,
) -> tuple[bytes, str]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "*/*",
            **(headers or {}),
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            data = resp.read(_FETCH_BYTE_CAP + 1)
            if len(data) > _FETCH_BYTE_CAP:
                data = data[:_FETCH_BYTE_CAP]
            return data, ctype
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read(400).decode("utf-8", errors="replace")
        except Exception:
            pass
        raise ValueError(f"HTTP {exc.code} for {url}: {_clip(body, 200) or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"Network error fetching {url}: {exc.reason}") from exc


class _TextExtractor(HTMLParser):
    """Minimal HTML → visible text (drops script/style/noscript)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in {"script", "style", "noscript"}:
            self._skip += 1
        elif t in {"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in {"script", "style", "noscript"} and self._skip > 0:
            self._skip -= 1
        elif t in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip == 0 and data:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        raw = re.sub(r"[ \t\f\v]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        # Extremely broken markup — fall back to tag strip
        return re.sub(r"<[^>]+>", " ", html)
    return parser.text()


def web_search(query: str, count: int = _DEFAULT_SEARCH_COUNT) -> dict[str, Any]:
    """Search the public web via Brave Search API. Returns concise title/url/snippet rows."""
    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")
    try:
        n = int(count)
    except (TypeError, ValueError) as exc:
        raise ValueError("count must be an integer") from exc
    n = max(1, min(_MAX_SEARCH_COUNT, n))

    key = _api_key()
    params = urllib.parse.urlencode({"q": q, "count": str(n)})
    url = f"{BRAVE_SEARCH_URL}?{params}"
    raw, _ctype = _http_get(
        url,
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": key,
        },
    )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Brave Search returned non-JSON: {exc}") from exc

    web = payload.get("web") if isinstance(payload, dict) else None
    results_raw = (web or {}).get("results") if isinstance(web, dict) else None
    if not isinstance(results_raw, list):
        results_raw = []

    results: list[dict[str, str]] = []
    for item in results_raw[:n]:
        if not isinstance(item, dict):
            continue
        title = _clip(str(item.get("title") or ""), _TITLE_MAX)
        link = str(item.get("url") or "").strip()
        snippet = _clip(
            str(item.get("description") or item.get("snippet") or ""),
            _SNIPPET_MAX,
        )
        if not link:
            continue
        results.append({"title": title, "url": link, "snippet": snippet})

    return {
        "ok": True,
        "query": q,
        "count": len(results),
        "results": results,
        "provider": "brave",
    }


def web_fetch(url: str, max_chars: int = _FETCH_DEFAULT_CHARS) -> dict[str, Any]:
    """Fetch a URL and return truncated plain text (for deep dives after web_search)."""
    target = (url or "").strip()
    if not target:
        raise ValueError("url is required")
    parsed = urllib.parse.urlparse(target)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an absolute http(s) URL")

    try:
        limit = int(max_chars)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_chars must be an integer") from exc
    limit = max(200, min(_FETCH_MAX_CHARS, limit))

    data, ctype = _http_get(target)
    # charset guess: utf-8 default
    text = data.decode("utf-8", errors="replace")
    if ctype in {"text/html", "application/xhtml+xml"} or (
        not ctype and "<html" in text[:500].lower()
    ):
        text = _html_to_text(text)
    elif ctype and not (
        ctype.startswith("text/")
        or ctype in {"application/json", "application/xml", "application/javascript"}
    ):
        raise ValueError(f"Unsupported content type for text extract: {ctype or 'unknown'}")

    truncated = len(text) > limit
    body = text[:limit] + ("…" if truncated else "")
    return {
        "ok": True,
        "url": target,
        "content_type": ctype or "unknown",
        "chars": len(body),
        "truncated": truncated,
        "text": body,
    }
