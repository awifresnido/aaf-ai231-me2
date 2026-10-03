"""PyTorch runtime for tiny-vcm checkpoints (TC-ResNet / DS-CNN).

Feature extraction is imported from the tiny-vcm repository itself
(``src/vcm_data_loader.py``), not reimplemented, so the edge computes exactly
the features the model was trained on: to_mono_16k -> trim_silence ->
log-mel + per-mel CMVN -> fit_frames(random_offset=False).
"""
from __future__ import annotations

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
        from src.vcm_models_v2 import build_model  # type: ignore
    except ImportError as exc:  # torch / torchaudio / tiny-vcm missing
        raise EngineUnavailable(f"cannot import tiny-vcm runtime: {exc}") from exc
    return AudioProcessor, fit_frames, build_model


class TorchTinyVCMEngine(VCMEngine):
    def __init__(self, manifest: ModelManifest, tiny_vcm_root: str | None):
        super().__init__(manifest)
        self.tiny_vcm_root = tiny_vcm_root
        self._model = None
        self._processor = None
        self._fit_frames = None
        self._n_frames = 0

    def load(self) -> None:
        weights = self.manifest.weights_path()
        if weights is None or not weights.exists():
            raise EngineUnavailable(f"weights file missing: {weights}")
        AudioProcessor, fit_frames, build_model = _import_tiny_vcm(self.tiny_vcm_root)
        import torch

        f = self.manifest.features
        self._processor = AudioProcessor(
            sample_rate=f.sample_rate, n_mels=f.n_mels, n_fft=f.n_fft,
            hop_length=f.hop_length, f_min=f.f_min, f_max=f.f_max,
            top_db=f.top_db, trim_db=f.trim_db,
        )
        self._fit_frames = fit_frames
        self._n_frames = int(f.max_duration_s * f.sample_rate / f.hop_length) + 1

        n_classes = len(self.manifest.labels)
        if n_classes == 0:
            raise EngineUnavailable("manifest has no labels list")
        model = build_model(self.manifest.architecture, n_classes=n_classes, n_mels=f.n_mels)
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:
            raise EngineUnavailable(
                f"checkpoint does not match manifest architecture/labels: {exc}"
            ) from exc
        model.eval()
        torch.set_num_threads(int(self.manifest.engine_options.get("num_threads", 2)))
        self._model = model

    def param_count(self) -> int | None:
        if self._model is None:
            return None
        return sum(p.numel() for p in self._model.parameters())

    def infer(self, waveform: np.ndarray, sample_rate: int) -> RawOutput:
        import torch

        if self._model is None:
            raise EngineUnavailable("model not loaded")
        t0 = time.perf_counter()
        with torch.no_grad():
            wav = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)).unsqueeze(0)
            wav = self._processor.to_mono_16k(wav, sample_rate)
            wav = self._processor.trim_silence(wav)
            feat = self._processor.process(wav)                      # (1, n_mels, t)
            feat = self._fit_frames(feat, self._n_frames, random_offset=False)
            feat = feat.transpose(1, 2).unsqueeze(0)                  # (1, 1, time, n_mels)
            t1 = time.perf_counter()
            logits = self._model(feat)
            probs = torch.softmax(logits, dim=-1)[0].numpy()
        t2 = time.perf_counter()
        return RawOutput(probs=probs, feature_ms=(t1 - t0) * 1e3, inference_ms=(t2 - t1) * 1e3)
