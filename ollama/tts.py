"""Client for the local Qwen3-TTS server + sentence-chunk speech pipeline."""

from __future__ import annotations

import base64
import json
import queue
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterator
from urllib.parse import urljoin

AudioCallback = Callable[[bytes, int, str], None]  # wav_bytes, seq, text
TokenCallback = Callable[[str], None]
ErrorCallback = Callable[[str], None]


@dataclass(frozen=True)
class TtsConfig:
    url: str = "http://127.0.0.1:8765"
    enabled: bool = True
    language: str = "English"
    timeout_s: float = 120.0
    min_chars: int = 12


class TtsClient:
    def __init__(self, config: TtsConfig | None = None) -> None:
        self.config = config or TtsConfig()

    @property
    def base(self) -> str:
        return self.config.url.rstrip("/") + "/"

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(
                urljoin(self.base, "health"),
                timeout=min(1.5, self.config.timeout_s),
            ) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return bool(data.get("ok"))
        except Exception:
            return False

    def synthesize_wav(self, text: str, *, language: str | None = None) -> bytes:
        text = (text or "").strip()
        if not text:
            raise ValueError("empty text")
        body = json.dumps(
            {
                "text": text,
                "language": language or self.config.language,
                "return_base64": False,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            urljoin(self.base, "tts/wav"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"TTS HTTP {exc.code}: {detail}") from exc

    def synthesize_b64(self, text: str, *, language: str | None = None) -> tuple[str, int]:
        text = (text or "").strip()
        if not text:
            raise ValueError("empty text")
        body = json.dumps(
            {
                "text": text,
                "language": language or self.config.language,
                "return_base64": True,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            urljoin(self.base, "tts"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"TTS HTTP {exc.code}: {detail}") from exc
        if not data.get("success"):
            raise RuntimeError(str(data))
        b64 = data.get("audio_base64") or ""
        sr = int(data.get("sample_rate") or 24000)
        if not b64:
            raise RuntimeError("TTS response missing audio_base64")
        return b64, sr


def split_speakable(buffer: str, *, final: bool = False, min_chars: int = 12) -> tuple[list[str], str]:
    """Pull completed speakable chunks from a streaming text buffer."""
    if not buffer:
        return [], ""

    chunks: list[str] = []
    rest = buffer

    # Prefer newline / sentence boundaries.
    while True:
        m = re.search(r"[.!?\n]", rest)
        if not m:
            break
        end = m.end()
        # Include trailing closing quotes/brackets after punctuation.
        while end < len(rest) and rest[end] in "\"')]}":
            end += 1
        piece = rest[:end].strip()
        rest = rest[end:]
        if not piece or piece.isspace():
            continue
        # Hold very short fragments unless final flush.
        if len(piece) < min_chars and not final and not piece.endswith("\n"):
            # Put back and wait for more unless we already have a strong end.
            if not re.search(r"[.!?…][\"')\]]*$", piece):
                rest = piece + (" " if rest and not rest[0].isspace() else "") + rest
                break
        chunks.append(re.sub(r"\s+", " ", piece).strip())

    if final:
        tail = rest.strip()
        if tail:
            chunks.append(re.sub(r"\s+", " ", tail))
            rest = ""
    return chunks, rest


class SpeechPipeline:
    """Buffer token deltas into sentences and synthesize them in a worker thread."""

    def __init__(
        self,
        client: TtsClient,
        *,
        on_audio: AudioCallback | None = None,
        on_audio_b64: Callable[[str, int, int, str], None] | None = None,
        on_error: ErrorCallback | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.client = client
        self.on_audio = on_audio
        self.on_audio_b64 = on_audio_b64
        self.on_error = on_error
        self.enabled = self.client.config.enabled if enabled is None else enabled
        self._buf = ""
        self._seq = 0
        self._q: queue.Queue[str | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        if self.enabled:
            self._worker = threading.Thread(target=self._run, name="ainet-tts", daemon=True)
            self._worker.start()

    def feed(self, delta: str) -> None:
        if not delta:
            return
        if not self.enabled:
            return
        self._buf += delta
        chunks, self._buf = split_speakable(
            self._buf, final=False, min_chars=self.client.config.min_chars
        )
        for chunk in chunks:
            self._q.put(chunk)

    def flush(self) -> None:
        if not self.enabled:
            return
        chunks, self._buf = split_speakable(
            self._buf, final=True, min_chars=self.client.config.min_chars
        )
        for chunk in chunks:
            self._q.put(chunk)

    def close(self, timeout: float = 300.0) -> None:
        if not self.enabled:
            return
        self.flush()
        self._q.put(None)
        if self._worker is not None:
            self._worker.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            text = item.strip()
            if not text:
                continue
            seq = self._seq
            self._seq += 1
            try:
                if self.on_audio_b64 is not None:
                    b64, sr = self.client.synthesize_b64(text)
                    self.on_audio_b64(b64, sr, seq, text)
                elif self.on_audio is not None:
                    wav = self.client.synthesize_wav(text)
                    self.on_audio(wav, seq, text)
                else:
                    self.client.synthesize_wav(text)
            except Exception as exc:
                if self.on_error:
                    self.on_error(str(exc))


def iter_speakable(text: str, *, min_chars: int = 12) -> Iterator[str]:
    chunks, rest = split_speakable(text, final=True, min_chars=min_chars)
    yield from chunks
    if rest.strip():
        yield rest.strip()


def wav_to_data_url(wav: bytes) -> str:
    return "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")
