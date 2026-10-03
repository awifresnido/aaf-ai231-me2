"""Audio helpers (PCM16 bytes <-> float32, WAV decode) with no heavy deps."""
from __future__ import annotations

import io
import math
import wave

import numpy as np

PCM16_FULL_SCALE = 32768.0


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / PCM16_FULL_SCALE


def rms_dbfs(x: np.ndarray) -> float:
    """RMS level in dBFS (0 dBFS = full-scale square wave); -120 floor."""
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return max(-120.0, 20.0 * math.log10(rms + 1e-12))


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Decode an integer-PCM WAV to (float32 mono, sample_rate)."""
    with wave.open(io.BytesIO(data), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        i = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
             | (b[:, 2].astype(np.int32) << 16))
        i = np.where(i >= 1 << 23, i - (1 << 24), i)
        x = i.astype(np.float32) / float(1 << 23)
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x.astype(np.float32), sr
