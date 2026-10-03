"""Config + factory for the VAD endpointing plugin.

demo.yaml (edge section), all keys optional:

    endpointing:
      enabled: true              # false -> classic fixed window (unchanged behaviour)
      engine: silero             # silero | energy
      model_path: null           # null -> auto-locate (models/vad/, $VCM_SILERO_VAD, openWakeWord)
      search_roots: ["../openwakeword"]
      n_threads: 1
      onset_threshold: 0.5
      offset_threshold: 0.35
      min_speech_ms: 200
      hangover_ms: 500
      no_speech_timeout_ms: 2000
      max_window_ms: 4000        # safety cap; keep = the old window_s * 1000
      fallback_to_energy: true   # if Silero can't load, use EnergyVAD instead of failing
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .endpointer import EndpointParams
from .engines import EnergyVAD, SileroVAD, find_silero_model
from .stream import CaptureEndpointer

log = logging.getLogger(__name__)


@dataclass
class EndpointingConfig:
    enabled: bool = False
    engine: str = "silero"
    model_path: Optional[str] = None
    search_roots: list[str] = field(default_factory=list)
    n_threads: int = 1
    fallback_to_energy: bool = True
    params: EndpointParams = field(default_factory=EndpointParams)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "EndpointingConfig":
        d = dict(d or {})
        pkeys = set(EndpointParams.__dataclass_fields__)
        params = EndpointParams(**{k: float(d.pop(k)) for k in list(d) if k in pkeys})
        params.validate()
        known = set(cls.__dataclass_fields__) - {"params"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown endpointing keys: {sorted(unknown)}")
        cfg = cls(params=params, **d)
        if cfg.engine not in ("silero", "energy"):
            raise ValueError(f"endpointing.engine must be silero|energy, got {cfg.engine!r}")
        return cfg


def build_endpointer(cfg: EndpointingConfig) -> tuple[Optional[CaptureEndpointer], str]:
    """-> (endpointer or None, status string for health/UI).
    None means 'use the fixed window' (disabled, or failed without fallback)."""
    if not cfg.enabled:
        return None, "disabled (fixed window)"
    if cfg.engine == "silero":
        try:
            path = find_silero_model(cfg.model_path, tuple(cfg.search_roots))
            eng = SileroVAD(path, n_threads=cfg.n_threads)
            return CaptureEndpointer(eng, cfg.params), f"silero {eng.generation} ({path.name})"
        except Exception as exc:  # noqa: BLE001
            if not cfg.fallback_to_energy:
                log.error("Silero VAD unavailable (%s); using the fixed window", exc)
                return None, f"unavailable: {exc}"
            log.warning("Silero VAD unavailable (%s); falling back to energy VAD", exc)
            return CaptureEndpointer(EnergyVAD(), cfg.params), f"energy (fallback: {exc})"
    return CaptureEndpointer(EnergyVAD(), cfg.params), "energy"
