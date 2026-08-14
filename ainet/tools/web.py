"""Internet tools: Brave Search + light URL fetch (stdlib / Windows-friendly)."""

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
BRAVE_IMAGES_URL = "https://api.search.brave.com/res/v1/images/search"
_USER_AGENT = "AINet/1.0 (+local; Brave Search)"
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
_DEFAULT_SEARCH_COUNT = 5
_MAX_SEARCH_COUNT = 8
_TITLE_MAX = 120
_SNIPPET_MAX = 280
_FETCH_DEFAULT_CHARS = 4000
_FETCH_MAX_CHARS = 12000
_FETCH_BYTE_CAP = 500_000
_HTTP_TIMEOUT_S = 20.0
_PREVIEW_TIMEOUT_S = 4.0
_PREVIEW_BYTE_CAP = 120_000

# Higher rank = more preferred for Deep Research. Not medical-only —
# societies, publishers, preprint servers, and gov/edu labs across fields.
_ACADEMIC_HOST_SCORES: tuple[tuple[str, int], ...] = (
    # Indexes / libraries
    ("ieeexplore.ieee.org", 100),
    ("ieee.org", 95),
    ("dl.acm.org", 100),
    ("acm.org", 95),
    ("arxiv.org", 90),
    ("pubmed.ncbi.nlm.nih.gov", 100),
    ("pmc.ncbi.nlm.nih.gov", 100),
    ("ncbi.nlm.nih.gov", 95),
    ("nih.gov", 90),
    ("nist.gov", 90),
    ("nasa.gov", 85),
    ("energy.gov", 80),
    ("osti.gov", 85),
    ("cdc.gov", 85),
    ("who.int", 80),
    ("cochrane.org", 90),
    # Societies
    ("aps.org", 90),
    ("acs.org", 90),
    ("asme.org", 90),
    ("sae.org", 80),
    ("aiaa.org", 85),
    ("siam.org", 90),
    ("ams.org", 85),
    ("iet.org", 85),
    ("theiet.org", 85),
    ("usenix.org", 85),
    ("nips.cc", 85),
    ("neurips.cc", 85),
    ("openreview.net", 80),
    ("ssrn.com", 70),
    # Journals / publishers
    ("nature.com", 90),
    ("science.org", 90),
    ("pnas.org", 90),
    ("cell.com", 85),
    ("nejm.org", 85),
    ("thelancet.com", 85),
    ("jamanetwork.com", 85),
    ("bmj.com", 80),
    ("sciencedirect.com", 80),
    ("elsevier.com", 75),
    ("springer.com", 80),
    ("springeropen.com", 75),
    ("wiley.com", 75),
    ("tandfonline.com", 75),
    ("sagepub.com", 70),
    ("oup.com", 80),
    ("academic.oup.com", 85),
    ("cambridge.org", 80),
    ("iop.org", 85),
    ("iopscience.iop.org", 85),
    ("aip.org", 85),
    ("royalsocietypublishing.org", 80),
    ("frontiersin.org", 65),
    ("plos.org", 70),
    ("physiology.org", 70),
    ("acsm.org", 65),
    ("mdpi.com", 50),
)


def _host_academic_score(url: str) -> int:
    host = urllib.parse.urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    score = 0
    for suffix, pts in _ACADEMIC_HOST_SCORES:
        if host == suffix or host.endswith("." + suffix):
            score = max(score, pts)
    if host.endswith(".gov") or host.endswith(".edu") or host.endswith(".ac.uk"):
        score = max(score, 50)
    if host.startswith("ieeexplore.") or ".ieee.org" in host:
        score = max(score, 95)
    if host.startswith("dl.acm.") or host.endswith(".acm.org"):
        score = max(score, 95)
    return score


def _prefer_academic(results: list[dict[str, str]]) -> list[dict[str, str]]:
    scored: list[tuple[int, int, dict[str, str]]] = []
    for i, row in enumerate(results):
        scored.append((-_host_academic_score(row.get("url") or ""), i, row))
    scored.sort()
    return [row for _, _, row in scored]


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


def _http_url(value: str) -> str:
    s = (value or "").strip()
    if s.startswith(("http://", "https://")):
        return s
    return ""


def _source_host(item: dict[str, Any] | None, url: str) -> str:
    meta = item.get("meta_url") if isinstance(item, dict) else None
    host = ""
    if isinstance(meta, dict):
        host = str(meta.get("hostname") or meta.get("netloc") or "").strip()
    if not host:
        host = urllib.parse.urlparse(url).netloc
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _brave_thumbnail(item: dict[str, Any]) -> str:
    thumb = item.get("thumbnail")
    if isinstance(thumb, dict):
        return _http_url(str(thumb.get("original") or thumb.get("src") or ""))
    if isinstance(thumb, str):
        return _http_url(thumb)
    return ""


def _youtube_thumb(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    vid = ""
    if "youtube.com" in host:
        vid = (urllib.parse.parse_qs(parsed.query).get("v") or [""])[0]
    elif "youtu.be" in host:
        vid = parsed.path.strip("/").split("/")[0]
    if vid and re.fullmatch(r"[\w-]{6,20}", vid):
        return f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
    return ""


class _OgImageParser(HTMLParser):
    """Pull og:image / twitter:image from a page head."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.image = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.image or tag.lower() != "meta":
            return
        props = {str(k).lower(): (v or "") for k, v in attrs}
        key = (props.get("property") or props.get("name") or "").lower()
        content = (props.get("content") or "").strip()
        if key in {"og:image", "og:image:url", "twitter:image", "twitter:image:src"} and content:
            self.image = content

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


def _page_og_image(url: str) -> str:
    yt = _youtube_thumb(url)
    if yt:
        return yt
    try:
        raw, ctype = _http_get(
            url,
            headers={
                "User-Agent": _BROWSER_UA,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            },
            timeout=_PREVIEW_TIMEOUT_S,
            max_bytes=_PREVIEW_BYTE_CAP,
        )
    except ValueError:
        return ""
    if ctype and "html" not in ctype and "xml" not in ctype:
        return ""
    try:
        html = raw.decode("utf-8", errors="replace")
    except Exception:
        return ""
    parser = _OgImageParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return ""
    img = parser.image.strip()
    if not img:
        return ""
    abs_url = urllib.parse.urljoin(url, img)
    return _http_url(abs_url)


def article_preview_images(urls: list[str]) -> dict[str, str]:
    """Best-effort og:image (or YouTube thumb) for opened article URLs."""
    clean: list[str] = []
    seen: set[str] = set()
    for raw in urls:
        u = _http_url(str(raw or ""))
        if not u or u in seen:
            continue
        seen.add(u)
        clean.append(u)
    if not clean:
        return {}
    from concurrent.futures import ThreadPoolExecutor, as_completed

    found: dict[str, str] = {}
    workers = min(3, len(clean))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_page_og_image, u): u for u in clean}
        for fut in as_completed(futs):
            src = futs[fut]
            try:
                img = fut.result() or ""
            except Exception:
                img = ""
            if img:
                found[src] = img
    return found


def _http_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = _HTTP_TIMEOUT_S,
    max_bytes: int | None = None,
) -> tuple[bytes, str]:
    cap = _FETCH_BYTE_CAP if max_bytes is None else max(1024, int(max_bytes))
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
            data = resp.read(cap + 1)
            if len(data) > cap:
                data = data[:cap]
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
        row: dict[str, str] = {
            "title": title,
            "url": link,
            "snippet": snippet,
            "source": _source_host(item, link),
        }
        thumb = _brave_thumbnail(item)
        if thumb:
            row["thumbnail"] = thumb
        results.append(row)

    results = _prefer_academic(results)
    return {
        "ok": True,
        "query": q,
        "count": len(results),
        "results": results,
        "provider": "brave",
    }


def image_search(
    query: str,
    count: int = 6,
    *,
    open_google: bool = True,
) -> dict[str, Any]:
    """Search public images via Brave. Optionally open Google Images in Chrome."""
    q = (query or "").strip()
    if not q:
        raise ValueError("query is required")
    try:
        n = int(count)
    except (TypeError, ValueError) as exc:
        raise ValueError("count must be an integer") from exc
    n = max(1, min(8, n))

    results: list[dict[str, str]] = []
    brave_error = ""
    try:
        key = _api_key()
        params = urllib.parse.urlencode(
            {
                "q": q,
                "count": str(n),
                "safesearch": "strict",
                "spellcheck": "1",
            }
        )
        raw, _ctype = _http_get(
            f"{BRAVE_IMAGES_URL}?{params}",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": key,
            },
        )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Brave Images returned non-JSON: {exc}") from exc

        rows = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            rows = []

        for item in rows[:n]:
            if not isinstance(item, dict):
                continue
            props = item.get("properties") if isinstance(item.get("properties"), dict) else {}
            thumb_obj = item.get("thumbnail") if isinstance(item.get("thumbnail"), dict) else {}
            meta = item.get("meta_url") if isinstance(item.get("meta_url"), dict) else {}
            image_url = str(props.get("url") or item.get("url") or "").strip()
            thumb = str(thumb_obj.get("src") or image_url).strip()
            host = str(meta.get("hostname") or meta.get("netloc") or "").strip()
            path = str(meta.get("path") or "").strip()
            page_url = str(item.get("url") or "").strip()
            if host and path and not page_url.startswith("http"):
                scheme = str(meta.get("scheme") or "https")
                page_url = f"{scheme}://{host}{path if path.startswith('/') else '/' + path}"
            elif host and not page_url.startswith("http"):
                page_url = f"https://{host}"
            title = _clip(str(item.get("title") or host or "image"), _TITLE_MAX)
            show = thumb or image_url
            if not show.startswith(("http://", "https://")):
                continue
            results.append(
                {
                    "title": title,
                    "url": page_url if page_url.startswith("http") else show,
                    "page_url": page_url if page_url.startswith("http") else "",
                    "image_url": image_url if image_url.startswith("http") else show,
                    "thumbnail": show,
                    "source": host or urllib.parse.urlparse(show).netloc,
                }
            )
    except ValueError as exc:
        brave_error = str(exc)

    google_url = "https://www.google.com/search?" + urllib.parse.urlencode(
        {"tbm": "isch", "q": q}
    )
    opened: list[dict[str, str]] = []
    if open_google:
        from ainet.tools.browser import open_chrome

        chrome = open_chrome(google_url, new_tab=True)
        if chrome.get("ok"):
            opened.append({"title": f"Google Images: {q}", "url": google_url})

    out: dict[str, Any] = {
        "ok": bool(results or opened),
        "query": q,
        "count": len(results),
        "results": results,
        "provider": "brave",
        "google_images": google_url,
        "auto_opened": opened,
    }
    if brave_error:
        out["brave_error"] = brave_error
    if not results and opened:
        out["note"] = "Thumbnails unavailable; opened Google Images in Chrome."
    if not out["ok"]:
        out["error"] = brave_error or "no images found"
    return out


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
