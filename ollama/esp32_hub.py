"""ESP32 mic ingest: ping/pong tracking, PCM stream, VAD, STT, chat turns."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable

from ollama.stt import SttEngine

UtteranceHandler = Callable[[str], str]


def _rms_i16(frame: bytes) -> float:
    if len(frame) < 4:
        return 0.0
    n = len(frame) // 2
    total = 0
    # Avoid numpy on the audio hot path so a missing STT install still streams.
    for i in range(0, n * 2, 2):
        sample = int.from_bytes(frame[i : i + 2], "little", signed=True)
        total += sample * sample
    return (total / n) ** 0.5


class Esp32Hub:
    def __init__(self, on_utterance: UtteranceHandler | None = None) -> None:
        self.on_utterance = on_utterance
        self.stt = SttEngine()
        self.lock = threading.RLock()
        self._pcm = bytearray()
        self._rate = 16000
        self._voiced = False
        self._voice_frames = 0
        self._silence_frames = 0
        self._utterance = bytearray()
        self._noise = 250.0
        self._bytes = 0
        self._audio_at = 0.0
        self._ping_at = 0.0
        self._peer = ""
        self._streams = 0
        self._turn_seq = 0
        self.turns: deque[dict[str, Any]] = deque(maxlen=40)
        self._jobs: deque[bytes] = deque()
        self._job_ev = threading.Event()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, name="ainet-stt", daemon=True)
        self._worker.start()
        threading.Thread(target=self.stt.ensure, name="ainet-stt-load", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        self._job_ev.set()

    def ping(self, peer: str = "") -> dict[str, Any]:
        with self.lock:
            self._ping_at = time.monotonic()
            if peer:
                self._peer = peer
        return {
            "ok": True,
            "pong": True,
            "server": "ainet",
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def begin_stream(self, *, peer: str = "", sample_rate: int = 16000) -> None:
        with self.lock:
            self._streams += 1
            self._audio_at = time.monotonic()
            if peer:
                self._peer = peer
            if sample_rate > 0:
                self._rate = int(sample_rate)

    def end_stream(self) -> None:
        leftover = b""
        with self.lock:
            self._streams = max(0, self._streams - 1)
            leftover = bytes(self._utterance)
            self._utterance.clear()
            self._pcm.clear()
            self._voiced = False
            self._voice_frames = 0
            self._silence_frames = 0
        if leftover:
            self._queue_utterance(leftover)

    def feed_pcm(self, data: bytes, sample_rate: int = 0) -> None:
        if not data:
            return
        with self.lock:
            self._bytes += len(data)
            self._audio_at = time.monotonic()
            if sample_rate > 0:
                self._rate = int(sample_rate)
            self._pcm.extend(data)
            rate = self._rate
            pcm = self._pcm
        frame = max(2, int(rate * 0.03) * 2)  # 30 ms
        while True:
            with self.lock:
                if len(pcm) < frame:
                    break
                chunk = bytes(pcm[:frame])
                del pcm[:frame]
            self._vad_frame(chunk, rate)

    def clear_turns(self) -> None:
        with self.lock:
            self.turns.clear()

    def snapshot(self, after_id: int = 0, *, include_turns: bool = True) -> dict[str, Any]:
        now = time.monotonic()
        with self.lock:
            audio_age = now - self._audio_at if self._audio_at else None
            ping_age = now - self._ping_at if self._ping_at else None
            turns = (
                [t for t in self.turns if int(t.get("id") or 0) > after_id]
                if include_turns
                else []
            )
            return {
                "ok": True,
                "connected": bool(self._streams > 0 and audio_age is not None and audio_age < 2.5),
                "streaming": self._streams > 0,
                "ping_ok": bool(ping_age is not None and ping_age < 12.0),
                "peer": self._peer or None,
                "bytes": self._bytes,
                "sample_rate": self._rate,
                "audio_age_s": round(audio_age, 2) if audio_age is not None else None,
                "ping_age_s": round(ping_age, 2) if ping_age is not None else None,
                "stt": self.stt.status(),
                "turns": turns,
                "turn_seq": self._turn_seq,
            }

    def _vad_frame(self, frame: bytes, rate: int) -> None:
        rms = _rms_i16(frame)
        with self.lock:
            if not self._voiced:
                self._noise = (self._noise * 0.97) + (rms * 0.03)
            threshold = max(350.0, self._noise * 3.2)
            speech = rms >= threshold
            if speech:
                self._voice_frames += 1
                self._silence_frames = 0
                if not self._voiced and self._voice_frames >= 3:
                    self._voiced = True
                self._utterance.extend(frame)
                max_bytes = int(rate * 2 * 8)  # 8 s cap
                overflow = b""
                if self._voiced and len(self._utterance) >= max_bytes:
                    overflow = bytes(self._utterance)
                    self._utterance.clear()
                    self._voiced = False
                    self._voice_frames = 0
                else:
                    overflow = b""
            else:
                self._voice_frames = 0
                if self._voiced:
                    self._utterance.extend(frame)
                    self._silence_frames += 1
                    # ~400 ms silence ends the utterance
                    if self._silence_frames >= 14:
                        overflow = bytes(self._utterance)
                        self._utterance.clear()
                        self._voiced = False
                        self._silence_frames = 0
                    else:
                        overflow = b""
                else:
                    overflow = b""
        if overflow:
            self._queue_utterance(overflow)

    def _queue_utterance(self, pcm: bytes) -> None:
        if len(pcm) < 1600:  # < ~50 ms
            return
        with self.lock:
            self._jobs.append(pcm)
        self._job_ev.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._job_ev.wait(timeout=0.25)
            self._job_ev.clear()
            while not self._stop.is_set():
                with self.lock:
                    pcm = self._jobs.popleft() if self._jobs else None
                    rate = self._rate
                if pcm is None:
                    break
                self._transcribe_job(pcm, rate)

    def _transcribe_job(self, pcm: bytes, rate: int) -> None:
        try:
            text = self.stt.transcribe(pcm, rate)
        except Exception as exc:
            with self.lock:
                self._turn_seq += 1
                self.turns.append(
                    {
                        "id": self._turn_seq,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "text": "",
                        "reply": "",
                        "status": "error",
                        "error": str(exc),
                    }
                )
            print(f"STT error: {exc}", flush=True)
            return
        if not text or not any(ch.isalnum() for ch in text):
            return
        with self.lock:
            self._turn_seq += 1
            turn_id = self._turn_seq
            turn = {
                "id": turn_id,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "text": text,
                "reply": "",
                "status": "asking",
                "error": None,
            }
            self.turns.append(turn)
        reply = ""
        err = None
        if self.on_utterance is not None:
            try:
                reply = self.on_utterance(text) or ""
            except Exception as exc:
                err = str(exc)
        with self.lock:
            for row in self.turns:
                if row.get("id") == turn_id:
                    row["reply"] = reply
                    row["status"] = "error" if err else "done"
                    row["error"] = err
                    break
