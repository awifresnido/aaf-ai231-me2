"""Scripted engine used by the 'Simulate' buttons.

It ignores the audio and returns the class it was told to return, with
confidence 1.0. Every result it produces is tagged ``source='simulated'`` so
the UI can never pass it off as a real model output.
"""
from __future__ import annotations

import numpy as np

from vcm_common.manifest import ModelManifest
from vcm_common.ontology import get_ontology

from .base import RawOutput, VCMEngine

SCRIPTED_MODEL_ID = "simulator"


def scripted_manifest() -> ModelManifest:
    return ModelManifest(
        id=SCRIPTED_MODEL_ID,
        display_name="Simulator (no model)",
        task="vcm",
        engine="scripted",
        labels=get_ontology().leaf_labels(),
        reject={"min_confidence": 0.0},
    )


class ScriptedEngine(VCMEngine):
    def __init__(self, manifest: ModelManifest):
        super().__init__(manifest)
        self.next_class: str | None = None

    def load(self) -> None:
        return None

    def infer(self, waveform: np.ndarray, sample_rate: int) -> RawOutput:
        labels = self.manifest.labels
        probs = np.zeros(len(labels), dtype=np.float32)
        target = self.next_class or "SILENCE"
        probs[labels.index(target)] = 1.0
        self.next_class = None
        return RawOutput(probs=probs, feature_ms=0.0, inference_ms=0.0)
