"""Minimal Ollama HTTP client (stdlib only — macOS-friendly)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from ollama.config import OllamaConfig


class OllamaError(RuntimeError):
    """Raised when the Ollama server returns an error or is unreachable."""


class OllamaClient:
    def __init__(self, config: OllamaConfig) -> None:
        self.config = config

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        url = f"{self.config.host}/api/chat"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.config.host}. Is it running? ({exc.reason})"
            ) from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Ollama returned non-JSON: {body[:200]}") from exc

    def list_models(self) -> list[str]:
        url = f"{self.config.host}/api/tags"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Failed to list models: {exc}") from exc
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
