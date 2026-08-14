"""Minimal Ollama HTTP client (stdlib only — Windows-friendly)."""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from ollama.config import OllamaConfig

TokenCallback = Callable[[str], None]
ThinkingCallback = Callable[[str], None]


class OllamaError(RuntimeError):
    """Raised when the Ollama server returns an error or is unreachable."""


class OllamaCancelled(OllamaError):
    """Raised when the caller cancels an in-flight stream."""


class OllamaClient:
    def __init__(self, config: OllamaConfig) -> None:
        self.config = config
        self._active_resp: Any | None = None
        self._active_lock = threading.Lock()

    def cancel_active(self) -> None:
        """Close any in-flight Ollama HTTP stream so a blocked read can unblock.

        Close on a daemon thread — urllib close() from the HTTP handler thread
        can deadlock with a blocked readline() on Windows.
        """
        with self._active_lock:
            resp = self._active_resp
            self._active_resp = None
        if resp is None:
            return

        def _close() -> None:
            try:
                fp = getattr(resp, "fp", None)
                raw = getattr(fp, "raw", None) if fp is not None else None
                sock = getattr(raw, "_sock", None) if raw is not None else None
                if sock is None and fp is not None:
                    sock = getattr(fp, "_sock", None)
                if sock is not None:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                resp.close()
            except Exception:
                pass

        threading.Thread(target=_close, name="ollama-cancel", daemon=True).start()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        think: bool | None = None,
        timeout_s: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming chat (callers that want a single blob)."""
        return self._chat_request(
            messages,
            tools=tools,
            model=model,
            think=think,
            stream=False,
            timeout_s=timeout_s,
            options=options,
        )

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        think: bool | None = None,
        on_token: TokenCallback | None = None,
        on_thinking: ThinkingCallback | None = None,
        timeout_s: float | None = None,
        options: dict[str, Any] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Stream assistant tokens (+ optional thinking); return final chat shape."""
        return self._chat_request(
            messages,
            tools=tools,
            model=model,
            think=think,
            stream=True,
            on_token=on_token,
            on_thinking=on_thinking,
            timeout_s=timeout_s,
            options=options,
            should_cancel=should_cancel,
        )

    def _chat_request(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        think: bool | None = None,
        stream: bool = False,
        on_token: TokenCallback | None = None,
        on_thinking: ThinkingCallback | None = None,
        timeout_s: float | None = None,
        options: dict[str, Any] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        use_think = bool(self.config.oac_think if think is None else think)
        wait = float(self.config.timeout_s if timeout_s is None else timeout_s)
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "stream": stream,
            # Qwen3: false → /no_think, true → /think
            "think": use_think,
        }
        if tools:
            payload["tools"] = tools
        if options:
            payload["options"] = options

        url = f"{self.config.host}/api/chat"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=wait) as resp:
                with self._active_lock:
                    self._active_resp = resp
                try:
                    if not stream:
                        body = resp.read().decode("utf-8")
                        try:
                            return json.loads(body)
                        except json.JSONDecodeError as exc:
                            raise OllamaError(f"Ollama returned non-JSON: {body[:200]}") from exc
                    return self._consume_stream(
                        resp,
                        on_token=on_token,
                        on_thinking=on_thinking,
                        should_cancel=should_cancel,
                    )
                finally:
                    with self._active_lock:
                        if self._active_resp is resp:
                            self._active_resp = None
        except OllamaCancelled:
            raise
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if self._is_timeout(reason):
                raise OllamaError(
                    f"Ollama timed out after {wait:.0f}s (host={self.config.host})"
                ) from exc
            raise OllamaError(
                f"Cannot reach Ollama at {self.config.host}. Is it running? ({reason})"
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise OllamaError(
                f"Ollama timed out after {wait:.0f}s (host={self.config.host})"
            ) from exc

    @staticmethod
    def _is_timeout(reason: Any) -> bool:
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return True
        return "timed out" in str(reason).lower()

    @staticmethod
    def _set_read_timeout(resp: Any, seconds: float) -> None:
        """Wake blocked stream reads periodically so cancel can be honored."""
        try:
            fp = getattr(resp, "fp", None)
            raw = getattr(fp, "raw", None) if fp is not None else None
            sock = getattr(raw, "_sock", None) if raw is not None else None
            if sock is None and fp is not None:
                sock = getattr(fp, "_sock", None)
            if sock is not None:
                sock.settimeout(seconds)
        except Exception:
            pass

    def _consume_stream(
        self,
        resp: Any,
        *,
        on_token: TokenCallback | None = None,
        on_thinking: ThinkingCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Read NDJSON chat stream; accumulate final message (incl. tool_calls)."""
        role = "assistant"
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        meta: dict[str, Any] = {}
        self._set_read_timeout(resp, 1.0)

        while True:
            if should_cancel and should_cancel():
                try:
                    resp.close()
                except Exception:
                    pass
                raise OllamaCancelled("Cancelled")
            try:
                raw = resp.readline()
            except socket.timeout:
                continue
            except (OSError, ValueError) as exc:
                if should_cancel and should_cancel():
                    raise OllamaCancelled("Cancelled") from exc
                raise OllamaError(f"Ollama stream interrupted: {exc}") from exc
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OllamaError(f"Ollama stream non-JSON: {line[:200]}") from exc

            if chunk.get("error"):
                raise OllamaError(str(chunk["error"]))

            msg = chunk.get("message") or {}
            if msg.get("role"):
                role = msg["role"]

            delta = msg.get("content") or ""
            if delta:
                content_parts.append(delta)
                if on_token:
                    on_token(delta)

            think_delta = msg.get("thinking") or ""
            if think_delta:
                thinking_parts.append(think_delta)
                if on_thinking:
                    on_thinking(think_delta)

            incoming_tools = msg.get("tool_calls")
            if incoming_tools:
                tool_calls = list(incoming_tools)

            if chunk.get("done"):
                for key in (
                    "model",
                    "created_at",
                    "done_reason",
                    "total_duration",
                    "load_duration",
                    "prompt_eval_count",
                    "prompt_eval_duration",
                    "eval_count",
                    "eval_duration",
                ):
                    if key in chunk:
                        meta[key] = chunk[key]

        message: dict[str, Any] = {
            "role": role,
            "content": "".join(content_parts),
        }
        if thinking_parts:
            message["thinking"] = "".join(thinking_parts)
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {"message": message, "done": True, **meta}

    def list_models(self) -> list[str]:
        url = f"{self.config.host}/api/tags"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Failed to list models: {exc}") from exc
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
