"""VCM engine interface. One implementation per runtime (PyTorch today,
ONNX/TFLite later); the pipeline and UI never know which one is loaded."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from vcm_common.manifest import ModelManifest


class EngineUnavailable(RuntimeError):
    """The engine's runtime or weights are missing on this machine."""


@dataclass
class RawOutput:
    probs: np.ndarray  # (n_classes,) softmax probabilities, same order as manifest.labels
    feature_ms: float
    inference_ms: float


class VCMEngine(ABC):
    def __init__(self, manifest: ModelManifest):
        self.manifest = manifest

    @abstractmethod
    def load(self) -> None:
        """Load weights. Raise EngineUnavailable with a readable reason on failure."""

    @abstractmethod
    def infer(self, waveform: np.ndarray, sample_rate: int) -> RawOutput:
        """Classify one command clip (float32 mono in [-1, 1])."""

    def param_count(self) -> int | None:
        return None


def build_vcm_engine(manifest: ModelManifest, settings: dict) -> VCMEngine:
    if manifest.engine == "torch_tinyvcm":
        from .torch_tinyvcm import TorchTinyVCMEngine

        return TorchTinyVCMEngine(manifest, tiny_vcm_root=settings.get("tiny_vcm_root"))
    if manifest.engine == "onnx_tinyvcm":
        from .onnx_tinyvcm import OnnxTinyVCMEngine

        return OnnxTinyVCMEngine(manifest, tiny_vcm_root=settings.get("tiny_vcm_root"))
    if manifest.engine == "scripted":
        from .scripted import ScriptedEngine

        return ScriptedEngine(manifest)
    if manifest.engine == "agreement":
        from .agreement import AgreementEngine

        return AgreementEngine(manifest)
    raise EngineUnavailable(f"unknown engine '{manifest.engine}'")
