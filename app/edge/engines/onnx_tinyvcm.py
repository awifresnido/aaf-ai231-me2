"""ONNX Runtime VCM engine.

Feature extraction is identical to the torch engine (imported from tiny-vcm);
only the model forward pass runs in ONNX Runtime. The exported model consumes
``(batch=1, 1, n_frames, n_mels)`` mel features and returns logits.

An optional ``deploy.json`` next to the ``.onnx`` (written by
``scripts/export_onnx.py``) supplies the parameter count for the UI; without it
``param_count`` returns None.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from vcm_common.manifest import ModelManifest

from .base import EngineUnavailable, RawOutput, VCMEngine


def _import_tiny_vcm(root: str | None):
    if root:
        root_path = Path(root).expanduser().resolve()
        if not (root_path / "src" / "vcm_data_loader.py").exists():
            raise EngineUnavailable(f"tiny-vcm source not found under {root_path}")
        if str(root_path) not in sys.path:
            sys.path.insert(0, str(root_path))
    try:
        import torch  # noqa: F401
        from src.vcm_data_loader import AudioProcessor, fit_frames  # type: ignore
    except ImportError as exc:  # torch / torchaudio / tiny-vcm missing
        raise EngineUnavailable(f"cannot import tiny-vcm runtime: {exc}") from exc
    return AudioProcessor, fit_frames


class OnnxTinyVCMEngine(VCMEngine):
    def __init__(self, manifest: ModelManifest, tiny_vcm_root: str | None):
        super().__init__(manifest)
        self.tiny_vcm_root = tiny_vcm_root
        self._sess = None
        self._processor = None
        self._fit_frames = None
        self._n_frames = 0
        self._input_name = None
        self._params: int | None = None

    def load(self) -> None:
        weights = self.manifest.weights_path()
        if weights is None or not weights.exists():
            raise EngineUnavailable(f"weights file missing: {weights}")
        AudioProcessor, fit_frames = _import_tiny_vcm(self.tiny_vcm_root)

        f = self.manifest.features
        self._processor = AudioProcessor(
            sample_rate=f.sample_rate, n_mels=f.n_mels, n_fft=f.n_fft,
            hop_length=f.hop_length, f_min=f.f_min, f_max=f.f_max,
            top_db=f.top_db, trim_db=f.trim_db,
        )
        self._fit_frames = fit_frames
        self._n_frames = int(f.max_duration_s * f.sample_rate / f.hop_length) + 1

        import torch
        import onnxruntime as ort

        n_threads = int(self.manifest.engine_options.get("num_threads", 2))
        torch.set_num_threads(n_threads)
        so = ort.SessionOptions()
        so.intra_op_num_threads = n_threads
        self._sess = ort.InferenceSession(str(weights), so, providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name

        sidecar = weights.with_name("deploy.json")
        if sidecar.is_file():
            try:
                self._params = json.loads(sidecar.read_text(encoding="utf-8")).get("params")
            except (OSError, ValueError):
                self._params = None

    def param_count(self) -> int | None:
        return self._params

    def infer(self, waveform: np.ndarray, sample_rate: int) -> RawOutput:
        import torch

        if self._sess is None:
            raise EngineUnavailable("model not loaded")
        t0 = time.perf_counter()
        with torch.no_grad():
            wav = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)).unsqueeze(0)
            wav = self._processor.to_mono_16k(wav, sample_rate)
            wav = self._processor.trim_silence(wav)
            feat = self._processor.process(wav)                     # (1, n_mels, t)
            feat = self._fit_frames(feat, self._n_frames, random_offset=False)
            feat = feat.transpose(1, 2).unsqueeze(0).contiguous()   # (1, 1, time, n_mels)
        t1 = time.perf_counter()
        logits = self._sess.run(None, {self._input_name: feat.numpy()})[0][0]
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        t2 = time.perf_counter()
        return RawOutput(probs=probs.astype(np.float32), feature_ms=(t1 - t0) * 1e3,
                         inference_ms=(t2 - t1) * 1e3)
