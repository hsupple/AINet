"""HTTP client for the Windows AINet tunnel (pathroom.org).

Mac does not run local Ollama when AINET_REMOTE_URL is set. Chat, SOI, and
status all go through the Windows web API on that host.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from ollama.config import OllamaConfig


class RemoteError(RuntimeError):
    """Raised when the tunneled AINet host is unreachable or blocked."""


def _access_blocked_message(url: str) -> str:
    return (
        f"Cloudflare Access is blocking {url}. Log in once in a browser, or set "
        "CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET (service token)."
    )


class RemoteAinetClient:
    def __init__(self, config: OllamaConfig) -> None:
        base = (config.remote_url or "").rstrip("/")
        if not base:
            raise RemoteError("AINET_REMOTE_URL is empty")
        self.config = config
        self.base = base
        self._opener = urllib.request.build_opener(_AccessRedirectHandler)

    def headers(self, *, content_type: str | None = "application/json") -> dict[str, str]:
        # Cloudflare 1010 blocks Python's default urllib User-Agent.
        out: dict[str, str] = {
            "Accept": "application/json, text/event-stream",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
            ),
        }
        if content_type:
            out["Content-Type"] = content_type
        cid = (self.config.cf_access_client_id or "").strip()
        secret = (self.config.cf_access_client_secret or "").strip()
        if cid and secret:
            out["CF-Access-Client-Id"] = cid
            out["CF-Access-Client-Secret"] = secret
        return out

    def status(self) -> dict[str, Any]:
        return self._json("GET", "/api/status")

    def reset(self) -> dict[str, Any]:
        return self._json("POST", "/api/reset", {})

    def set_mode(self, mode_id: str) -> dict[str, Any]:
        return self._json("POST", "/api/mode", {"mode": mode_id})

    def run_soi(self) -> dict[str, Any]:
        return self._json("POST", "/api/soi", {})

    def cancel(self, *, chat: bool = True, soi: bool = True) -> dict[str, Any]:
        return self._json("POST", "/api/cancel", {"chat": chat, "soi": soi})

    def chat(self, message: str) -> dict[str, Any]:
        return self._json("POST", "/api/chat", {"message": message})

    def open_raw(
        self,
        method: str,
        path_qs: str,
        body: bytes | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        """Open a raw HTTP response to the remote API (caller must close)."""
        if not path_qs.startswith("/"):
            path_qs = "/" + path_qs
        headers = self.headers(content_type="application/json" if body is not None else None)
        req = urllib.request.Request(
            f"{self.base}{path_qs}",
            data=body,
            headers=headers,
            method=method,
        )
        wait = timeout_s
        try:
            resp = self._opener.open(req, timeout=wait)
            self._reject_access(resp, f"{self.base}{path_qs}")
            return resp
        except RemoteError:
            raise
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise RemoteError(self._http_error(exc, f"{self.base}{path_qs}")) from exc
            # Pass AINet 4xx/5xx through so the Mac UI can show the real error.
            return exc
        except urllib.error.URLError as exc:
            raise RemoteError(
                f"Cannot reach AINet at {self.base} ({getattr(exc, 'reason', exc)})"
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RemoteError(f"Timed out talking to {self.base}") from exc

    def chat_stream(self, message: str) -> Iterator[dict[str, Any]]:
        payload = json.dumps({"message": message, "voice": False}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/api/chat-stream",
            data=payload,
            headers=self.headers(),
            method="POST",
        )
        try:
            with self._opener.open(req, timeout=self.config.soi_timeout_s) as resp:
                self._reject_access(resp, f"{self.base}/api/chat-stream")
                yield from self._iter_sse(resp)
        except RemoteError:
            raise
        except urllib.error.HTTPError as exc:
            raise RemoteError(self._http_error(exc, f"{self.base}/api/chat-stream")) from exc
        except urllib.error.URLError as exc:
            raise RemoteError(
                f"Cannot reach AINet at {self.base} ({getattr(exc, 'reason', exc)})"
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RemoteError(f"Timed out talking to {self.base}") from exc

    def _json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        wait = float(self.config.timeout_s if timeout_s is None else timeout_s)
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = self.headers(content_type=None if body is None else "application/json")
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with self._opener.open(req, timeout=wait) as resp:
                self._reject_access(resp, f"{self.base}{path}")
                raw = resp.read().decode("utf-8", errors="replace")
        except RemoteError:
            raise
        except urllib.error.HTTPError as exc:
            raise RemoteError(self._http_error(exc, f"{self.base}{path}")) from exc
        except urllib.error.URLError as exc:
            raise RemoteError(
                f"Cannot reach AINet at {self.base} ({getattr(exc, 'reason', exc)})"
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise RemoteError(f"Timed out talking to {self.base}") from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            preview = raw[:180].replace("\n", " ")
            raise RemoteError(f"{self.base}{path} returned non-JSON: {preview}") from exc
        if not isinstance(parsed, dict):
            raise RemoteError(f"{self.base}{path} returned a non-object JSON payload")
        return parsed

    def _http_error(self, exc: urllib.error.HTTPError, url: str) -> str:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code in {401, 403} or "cloudflareaccess" in detail.lower() or "cf-access" in detail.lower():
            if "1010" in detail:
                return (
                    f"Cloudflare blocked {url} (error 1010). This client already sends a "
                    "browser User-Agent; check Bot Fight Mode / WAF on pathroom.org."
                )
            return _access_blocked_message(url)
        preview = detail[:240].replace("\n", " ")
        return f"AINet HTTP {exc.code} at {url}: {preview}"

    def _reject_access(self, resp: Any, url: str) -> None:
        ctype = str(resp.headers.get("Content-Type") or "").lower()
        location = str(resp.headers.get("Location") or "")
        if "cloudflareaccess.com" in location or "/cdn-cgi/access" in location:
            raise RemoteError(_access_blocked_message(url))
        if "text/html" in ctype:
            raise RemoteError(_access_blocked_message(url))

    @staticmethod
    def _iter_sse(resp: Any) -> Iterator[dict[str, Any]]:
        buf = ""
        while True:
            chunk = resp.read(2048)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                data_lines = [
                    line[6:] if line.startswith("data: ") else line[5:]
                    for line in block.splitlines()
                    if line.startswith("data:")
                ]
                if not data_lines:
                    continue
                raw = "\n".join(data_lines).strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event


class _AccessRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        location = str(headers.get("Location") or newurl or "")
        if "cloudflareaccess.com" in location or "/cdn-cgi/access" in location:
            raise RemoteError(_access_blocked_message(str(req.full_url)))
        return super().redirect_request(req, fp, code, msg, headers, newurl)
