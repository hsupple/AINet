"""Local speech-to-text for ESP32 PCM (faster-whisper, then openai-whisper)."""

from __future__ import annotations

import os
import threading
from typing import Any


def _env_model() -> str:
    return (os.environ.get("AINET_STT_MODEL") or "").strip()


def _want_cuda() -> bool:
    forced = (os.environ.get("AINET_STT_DEVICE") or "").strip().lower()
    if forced in {"cpu", "none"}:
        return False
    if forced in {"cuda", "gpu"}:
        return True
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class SttEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._backend: str | None = None
        self._model: Any = None
        self._error: str | None = None
        self._model_name = _env_model()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ready": self._model is not None,
                "backend": self._backend,
                "model": self._model_name or None,
                "error": self._error,
            }

    def ensure(self) -> None:
        with self._lock:
            if self._model is not None or self._error:
                return
            cuda = _want_cuda()
            name = self._model_name or ("base.en" if cuda else "tiny.en")
            self._model_name = name
            last_err = ""
            try:
                from faster_whisper import WhisperModel

                device = "cuda" if cuda else "cpu"
                compute = "float16" if cuda else "int8"
                self._model = WhisperModel(name, device=device, compute_type=compute)
                self._backend = "faster-whisper"
                self._error = None
                print(f"STT  faster-whisper {name} on {device}", flush=True)
                return
            except Exception as exc:
                last_err = f"faster-whisper: {exc}"
            try:
                import whisper

                self._model = whisper.load_model(name)
                self._backend = "whisper"
                self._error = None
                print(f"STT  openai-whisper {name}", flush=True)
                return
            except Exception as exc:
                last_err = (last_err + " | " if last_err else "") + f"whisper: {exc}"
            self._error = (
                last_err
                + "  Install with: pip install faster-whisper"
            )
            print(f"STT unavailable: {self._error}", flush=True)

    def transcribe(self, pcm16: bytes, sample_rate: int) -> str:
        self.ensure()
        with self._lock:
            model = self._model
            backend = self._backend
            err = self._error
        if model is None:
            raise RuntimeError(err or "STT engine is not available")
        if not pcm16:
            return ""
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required for STT") from exc

        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size < int(sample_rate * 0.25):
            return ""
        if sample_rate != 16000:
            # Linear resample to 16 kHz (whisper native).
            n_out = max(1, int(round(audio.size * 16000.0 / float(sample_rate))))
            x_old = np.linspace(0.0, 1.0, audio.size, endpoint=False)
            x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)

        if backend == "faster-whisper":
            segments, _info = model.transcribe(
                audio,
                language="en",
                beam_size=1,
                vad_filter=False,
                without_timestamps=True,
            )
            parts = [seg.text for seg in segments if getattr(seg, "text", None)]
            text = " ".join(p.strip() for p in parts if p and p.strip())
        else:
            result = model.transcribe(
                audio,
                language="en",
                fp16=bool(_want_cuda()),
                without_timestamps=True,
            )
            text = str(result.get("text") or "")
        return " ".join(text.split())
