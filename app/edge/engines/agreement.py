"""Agreement engine (Double-Check): no weights of its own.

Runs its member models on the same clip and accepts only when every member
predicts the same command with confidence >= threshold. The members themselves
are ordinary torch_tinyvcm engines loaded by the registry.

``infer`` is never called directly: the pipeline special-cases the agreement
engine and runs the members itself (see ``edge.pipeline._run``).
"""
from __future__ import annotations

import numpy as np

from vcm_common.manifest import ModelManifest

from .base import EngineUnavailable, RawOutput, VCMEngine


class AgreementEngine(VCMEngine):
    def __init__(self, manifest: ModelManifest):
        super().__init__(manifest)

    def load(self) -> None:
        return None

    def infer(self, waveform: np.ndarray, sample_rate: int) -> RawOutput:
        raise EngineUnavailable("agreement engine has no direct infer; run its members")

    def param_count(self) -> int | None:
        # members' params are summed by the registry
        return None
