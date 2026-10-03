"""Wake-word engines behind one interface.

* ``manual``       -- no audio model; wake comes from the UI button / Space key.
                      Always available and always usable, even when an audio
                      wake word is active.
* ``openwakeword`` -- an openWakeWord ONNX model (stock ``hey_rhasspy`` today,
                      a custom model later), loaded from explicit file paths so
                      it does not depend on the package's download cache.

Detection = ``threshold`` on the per-frame score, held for ``patience``
consecutive 80 ms frames, with a ``debounce_s`` refractory period.
Status of the stock model on Awi's voice: ../openwakeword/WAKE_WORD_STATUS.md.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

OWW_FRAME_SAMPLES = 1280  # openWakeWord step: 80 ms @ 16 kHz


@dataclass
class WakeInfo:
    id: str
    display_name: str
    kind: str
    status: str                      # ready | unavailable
    detail: Optional[str] = None
    phrase: Optional[str] = None     # what the user should say
    threshold: Optional[float] = None
    patience: Optional[int] = None
    debounce_s: Optional[float] = None


class WakeEngine:
    info: WakeInfo

    def process(self, pcm_int16: np.ndarray) -> Optional[float]:
        """Feed 16 kHz int16 audio; return the score when the wake word fires, else None."""
        return None

    def reset(self) -> None:
        return None

    def pop_peak(self) -> Optional[float]:
        """Highest frame score since the last call (for the live meter), or None."""
        return None

    def summary(self) -> dict:
        return asdict(self.info)


class ManualWake(WakeEngine):
    def __init__(self) -> None:
        self.info = WakeInfo("manual", "Manual trigger (button / Space)", "manual", "ready")


@dataclass
class ScoreGate:
    """threshold + patience + debounce over a stream of per-frame scores."""

    threshold: float = 0.5
    patience: int = 1
    debounce_s: float = 1.5
    _run: int = 0
    _last_fire: float = -1e9

    def update(self, score: float, now: float) -> bool:
        self._run = self._run + 1 if score >= self.threshold else 0
        if self._run >= self.patience and now - self._last_fire >= self.debounce_s:
            self._last_fire = now
            self._run = 0
            return True
        return False

    def reset(self) -> None:
        self._run = 0


class OpenWakeWordEngine(WakeEngine):
    def __init__(self, engine_id: str, model_path: str, melspec_path: Optional[str],
                 embedding_path: Optional[str], threshold: float = 0.5, patience: int = 2,
                 debounce_s: float = 1.5, display_name: Optional[str] = None,
                 phrase: Optional[str] = None,
                 model_factory: Optional[Callable[..., object]] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.gate = ScoreGate(threshold, patience, debounce_s)
        self.clock = clock
        self._buf = np.zeros(0, dtype=np.int16)
        self._peak: Optional[float] = None
        self._model = None
        self.info = WakeInfo(engine_id, display_name or engine_id, "openwakeword", "unavailable",
                             phrase=phrase, threshold=threshold, patience=patience,
                             debounce_s=debounce_s)
        try:
            missing = [p for p in (model_path, melspec_path, embedding_path)
                       if p and not Path(p).exists()]
            if missing:
                raise FileNotFoundError(f"model file(s) missing: {', '.join(missing)}")
            if model_factory is None:
                from openwakeword.model import Model  # type: ignore

                model_factory = Model
            kwargs = {"wakeword_models": [model_path], "inference_framework": "onnx"}
            if melspec_path:
                kwargs["melspec_model_path"] = melspec_path
            if embedding_path:
                kwargs["embedding_model_path"] = embedding_path
            self._model = model_factory(**kwargs)
            self.info.status, self.info.detail = "ready", None
        except Exception as exc:  # noqa: BLE001 -- missing lib/files: stay listed, unavailable
            self.info.detail = str(exc)
            log.warning("wake engine %s unavailable: %s", engine_id, exc)

    def process(self, pcm_int16: np.ndarray) -> Optional[float]:
        if self._model is None:
            return None
        self._buf = np.concatenate([self._buf, pcm_int16.astype(np.int16, copy=False)])
        fired: Optional[float] = None
        while self._buf.size >= OWW_FRAME_SAMPLES:
            frame, self._buf = self._buf[:OWW_FRAME_SAMPLES], self._buf[OWW_FRAME_SAMPLES:]
            scores = self._model.predict(frame)
            score = float(max(scores.values())) if scores else 0.0
            self._peak = score if self._peak is None else max(self._peak, score)
            if self.gate.update(score, self.clock()) and fired is None:
                fired = score
        return fired

    def pop_peak(self) -> Optional[float]:
        p, self._peak = self._peak, None
        return p

    def reset(self) -> None:
        """Clear audio + feature buffers so the command audio cannot re-trigger."""
        self._buf = np.zeros(0, dtype=np.int16)
        self.gate.reset()
        if self._model is not None and hasattr(self._model, "reset"):
            self._model.reset()


def build_wake_engines(cfg: list[dict], resolve: Callable[[Optional[str]], Optional[str]]
                       = lambda p: p) -> dict[str, WakeEngine]:
    engines: dict[str, WakeEngine] = {"manual": ManualWake()}
    for item in cfg or []:
        if item.get("kind") != "openwakeword":
            log.warning("unknown wake engine kind %r ignored", item.get("kind"))
            continue
        eid = f"oww:{item['id']}"
        engines[eid] = OpenWakeWordEngine(
            eid, resolve(item["model"]), resolve(item.get("melspec_model")),
            resolve(item.get("embedding_model")),
            threshold=float(item.get("threshold", 0.5)), patience=int(item.get("patience", 2)),
            debounce_s=float(item.get("debounce_s", 1.5)),
            display_name=item.get("display_name"), phrase=item.get("phrase"),
        )
    return engines
