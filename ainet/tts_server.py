"""Qwen3-TTS local API — run with the Qwen3-TTS venv.

  B:\\AI\\Qwen3-TTS\\venv\\Scripts\\python.exe ainet\\tts_server.py

Optimized for low-latency sentence chunks from AINet chat streaming:
  - model loaded once
  - voice-clone prompt cached at startup
  - returns WAV bytes (no disk round-trip by default)
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(os.environ.get("AINET_TTS_HOME", r"B:\AI\Qwen3-TTS"))
MODEL_DIR = Path(
    os.environ.get(
        "AINET_TTS_MODEL",
        str(BASE_DIR / "models" / "Qwen3-TTS-12Hz-0.6B-Base"),
    )
)
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REF_AUDIO = Path(
    os.environ.get("AINET_TTS_REF_AUDIO", str(BASE_DIR / "ref" / "clone.wav"))
)
REF_TEXT = os.environ.get(
    "AINET_TTS_REF_TEXT",
    "",
).strip()
if not REF_TEXT:
    ref_txt = BASE_DIR / "ref" / "ref_text.txt"
    if ref_txt.is_file():
        REF_TEXT = ref_txt.read_text(encoding="utf-8").strip()
if not REF_TEXT:
    REF_TEXT = (
        "Okay. Yeah. I resent you. I love you. I respect you. "
        "But you know what? You blew it! And thanks to you."
    )

# Prefer CUDA when the installed torch build supports it.
_env_device = os.environ.get("AINET_TTS_DEVICE", "").strip()
if _env_device:
    DEVICE = _env_device
elif torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

XVEC_ONLY = os.environ.get("AINET_TTS_XVEC_ONLY", "0") not in {"0", "false", "False"}
SAVE_FILES = os.environ.get("AINET_TTS_SAVE", "0") not in {"0", "false", "False"}
HOST = os.environ.get("AINET_TTS_HOST", "127.0.0.1")
PORT = int(os.environ.get("AINET_TTS_PORT", "8765"))

_lock = threading.Lock()
_voice_prompt = None
_ready_at = 0.0


def _pick_dtype() -> torch.dtype:
    if DEVICE.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _load_model() -> Qwen3TTSModel:
    dtype = _pick_dtype()
    kwargs: dict = {
        "device_map": DEVICE,
        "dtype": dtype,
    }
    # FlashAttention only helps on CUDA + matching install.
    if DEVICE.startswith("cuda"):
        try:
            import flash_attn  # noqa: F401

            kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            kwargs["attn_implementation"] = "sdpa"

    print(f"Loading Qwen3-TTS from {MODEL_DIR}")
    print(f"  device={DEVICE} dtype={dtype}")
    model = Qwen3TTSModel.from_pretrained(str(MODEL_DIR), **kwargs)
    print("Qwen3-TTS loaded.")
    return model


print("Loading Qwen3-TTS...")
model = _load_model()


def _ensure_voice_prompt():
    global _voice_prompt, _ready_at
    if _voice_prompt is not None:
        return _voice_prompt
    if not REF_AUDIO.is_file():
        raise FileNotFoundError(
            f"Reference audio missing: {REF_AUDIO}. "
            "Set AINET_TTS_REF_AUDIO to a WAV for voice cloning."
        )
    print(f"Caching voice-clone prompt from {REF_AUDIO}")
    items = model.create_voice_clone_prompt(
        ref_audio=str(REF_AUDIO),
        ref_text=REF_TEXT,
        x_vector_only_mode=XVEC_ONLY,
    )
    _voice_prompt = items
    _ready_at = time.time()
    print("Voice-clone prompt ready.")
    return _voice_prompt


try:
    _ensure_voice_prompt()
except Exception as exc:
    print(f"WARNING: voice prompt not cached yet: {exc}")


# ============================================================
# API
# ============================================================

app = FastAPI(title="Qwen3-TTS Local API")


class TTSRequest(BaseModel):
    text: str
    language: str = "English"
    # Optional overrides (re-encode prompt — slower)
    reference_audio: str | None = None
    reference_text: str | None = None
    x_vector_only_mode: bool | None = None
    # Return options
    return_base64: bool = True
    save: bool = False


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return buf.getvalue()


def _synthesize(
    text: str,
    *,
    language: str = "English",
    reference_audio: str | None = None,
    reference_text: str | None = None,
    x_vector_only_mode: bool | None = None,
) -> tuple[np.ndarray, int]:
    text = (text or "").strip()
    if not text:
        raise ValueError("text cannot be empty")

    xvec = XVEC_ONLY if x_vector_only_mode is None else bool(x_vector_only_mode)

    with _lock:
        t0 = time.perf_counter()
        if reference_audio:
            ref = Path(reference_audio)
            if not ref.exists():
                raise FileNotFoundError(f"Reference audio not found: {ref}")
            if not xvec and not (reference_text or "").strip():
                raise ValueError("reference_text is required when using reference_audio")
            wavs, sample_rate = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=str(ref),
                ref_text=reference_text or REF_TEXT,
                x_vector_only_mode=xvec,
            )
        else:
            prompt = _ensure_voice_prompt()
            wavs, sample_rate = model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt,
            )
        elapsed = time.perf_counter() - t0
        print(f"TTS {elapsed:.2f}s  chars={len(text)}  text={text[:80]!r}")
        return wavs[0], int(sample_rate)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Qwen3-TTS",
        "device": DEVICE,
        "model": str(MODEL_DIR),
        "voice_cached": _voice_prompt is not None,
        "ref_audio": str(REF_AUDIO),
        "x_vector_only": XVEC_ONLY,
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "voice_cached": _voice_prompt is not None,
        "ready_at": _ready_at,
        "device": DEVICE,
    }


@app.post("/tts")
def generate_tts(request: TTSRequest):
    try:
        audio, sample_rate = _synthesize(
            request.text,
            language=request.language,
            reference_audio=request.reference_audio,
            reference_text=request.reference_text,
            x_vector_only_mode=request.x_vector_only_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload: dict = {
        "success": True,
        "sample_rate": sample_rate,
        "duration_s": float(len(audio) / max(sample_rate, 1)),
    }

    do_save = request.save or SAVE_FILES
    if do_save:
        filename = f"{uuid.uuid4().hex}.wav"
        output_file = OUTPUT_DIR / filename
        sf.write(str(output_file), audio, sample_rate)
        payload["file"] = str(output_file)

    if request.return_base64:
        payload["audio_base64"] = base64.b64encode(_wav_bytes(audio, sample_rate)).decode(
            "ascii"
        )
        payload["mime"] = "audio/wav"

    return payload


@app.post("/tts/wav")
def generate_tts_wav(request: TTSRequest):
    """Raw WAV bytes — fastest path for clients that can play audio/wav."""
    try:
        audio, sample_rate = _synthesize(
            request.text,
            language=request.language,
            reference_audio=request.reference_audio,
            reference_text=request.reference_text,
            x_vector_only_mode=request.x_vector_only_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    data = _wav_bytes(audio, sample_rate)
    return Response(
        content=data,
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Duration-S": f"{len(audio) / max(sample_rate, 1):.3f}",
            "Cache-Control": "no-store",
        },
    )


class BatchTTSRequest(BaseModel):
    texts: list[str] = Field(default_factory=list)
    language: str = "English"


@app.post("/tts/batch")
def generate_tts_batch(request: BatchTTSRequest):
    """Synthesize multiple short sentences; returns ordered base64 WAVs."""
    out = []
    for i, text in enumerate(request.texts):
        text = (text or "").strip()
        if not text:
            out.append({"index": i, "ok": False, "error": "empty"})
            continue
        try:
            audio, sample_rate = _synthesize(text, language=request.language)
            out.append(
                {
                    "index": i,
                    "ok": True,
                    "sample_rate": sample_rate,
                    "audio_base64": base64.b64encode(
                        _wav_bytes(audio, sample_rate)
                    ).decode("ascii"),
                    "mime": "audio/wav",
                }
            )
        except Exception as exc:
            out.append({"index": i, "ok": False, "error": str(exc)})
    return {"success": True, "items": out}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
