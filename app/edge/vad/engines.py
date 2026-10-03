"""VAD engines: per-chunk speech probability for 16 kHz mono float32 audio.

SileroVAD   -- Silero VAD ONNX model. Works with BOTH model generations:
                 * v3/v4 (inputs: input, sr, h, c) -- the copy openWakeWord ships
                 * v5    (inputs: input, state, sr) -- current Silero release
               The generation is detected from the model's input names.
EnergyVAD   -- dependency-free fallback (adaptive noise floor). Weaker in noise
               or music; used when the ONNX model or onnxruntime is unavailable.

Both expose: chunk_samples, reset(), prob(chunk) -> float in [0, 1].
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512          # 32 ms at 16 kHz; the size Silero v5 requires (v4 accepts it too)
V5_CONTEXT = 64              # v5 expects the last 64 samples of the previous chunk prepended


class VADEngine(Protocol):
    name: str
    chunk_samples: int

    def reset(self) -> None: ...
    def prob(self, chunk: np.ndarray) -> float: ...


def find_silero_model(explicit: Optional[str] = None, search_roots: tuple[str, ...] = ()) -> Path:
    """Locate silero_vad.onnx. Order: explicit path, $VCM_SILERO_VAD, app models/vad/,
    any extra search roots (e.g. an openWakeWord clone), the installed openwakeword package."""
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit))
    if os.environ.get("VCM_SILERO_VAD"):
        cands.append(Path(os.environ["VCM_SILERO_VAD"]))
    app_root = Path(__file__).resolve().parents[2]
    cands.append(app_root / "models" / "vad" / "silero_vad.onnx")
    for root in search_roots:
        r = Path(root)
        cands += [r / "models" / "pretrained" / "silero_vad.onnx",
                  r / "openwakeword" / "resources" / "models" / "silero_vad.onnx",
                  r / "silero_vad.onnx"]
    spec = importlib.util.find_spec("openwakeword")
    if spec and spec.origin:
        cands.append(Path(spec.origin).parent / "resources" / "models" / "silero_vad.onnx")
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "silero_vad.onnx not found. Copy it to models/vad/silero_vad.onnx "
        "(openWakeWord ships it under models/pretrained/ or resources/models/), "
        "or set endpointing.model_path / $VCM_SILERO_VAD. Tried: "
        + ", ".join(str(c) for c in cands))


class SileroVAD:
    name = "silero"

    def __init__(self, model_path: str | Path, n_threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = n_threads
        so.inter_op_num_threads = n_threads
        self.session = ort.InferenceSession(str(model_path), sess_options=so,
                                            providers=["CPUExecutionProvider"])
        names = {i.name for i in self.session.get_inputs()}
        if {"input", "state", "sr"} <= names:
            self.generation = "v5"
        elif {"input", "h", "c", "sr"} <= names:
            self.generation = "v4"
        else:
            raise ValueError(f"unrecognised Silero VAD model inputs: {sorted(names)}")
        self.model_path = str(model_path)
        self.chunk_samples = CHUNK_SAMPLES
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        if self.generation == "v5":
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            self._context = np.zeros((1, V5_CONTEXT), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def prob(self, chunk: np.ndarray) -> float:
        x = np.asarray(chunk, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != self.chunk_samples:
            raise ValueError(f"chunk must be {self.chunk_samples} samples, got {x.shape[1]}")
        if self.generation == "v5":
            xin = np.concatenate([self._context, x], axis=1)
            out, self._state = self.session.run(None, {"input": xin, "state": self._state, "sr": self._sr})
            self._context = xin[:, -V5_CONTEXT:]
        else:
            out, self._h, self._c = self.session.run(
                None, {"input": x, "sr": self._sr, "h": self._h, "c": self._c})
        return float(np.asarray(out).reshape(-1)[0])


class EnergyVAD:
    """Adaptive-noise-floor energy detector, mapped to a pseudo-probability.

    The floor is learned from the quietest recent chunks, so it adapts to room
    tone; speech is ~margin_db above it. Weaker than Silero with music/noise.
    """
    name = "energy"

    def __init__(self, margin_db: float = 12.0, slope_db: float = 3.0):
        self.chunk_samples = CHUNK_SAMPLES
        self.margin_db = margin_db
        self.slope_db = slope_db
        self.reset()

    def reset(self) -> None:
        self._floor: Optional[float] = None

    def prob(self, chunk: np.ndarray) -> float:
        x = np.asarray(chunk, dtype=np.float32)
        db = 10.0 * np.log10(float(np.mean(x * x)) + 1e-10)
        if self._floor is None:
            self._floor = db
        # floor follows quiet chunks quickly, loud chunks very slowly
        rate = 0.3 if db < self._floor else 0.005
        self._floor += rate * (db - self._floor)
        z = (db - self._floor - self.margin_db) / self.slope_db
        return float(1.0 / (1.0 + np.exp(-z)))
