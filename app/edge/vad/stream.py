"""CaptureEndpointer: streaming glue between the microphone frames and the endpointer.

The pipeline calls ``start()`` when a capture window opens, ``push(frame)`` for
every incoming audio frame (any length, e.g. 20 ms / 320 samples), and reads
``done`` / ``result()``. Frames are re-chunked to the engine's chunk size
(512 samples = 32 ms). The engine's state is reset per capture.

Cost is tracked (``vad_ms_total``, ``vad_ms_max``) so the UI/benchmark can show
the VAD's own CPU time per command.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .endpointer import EndpointParams, Endpointer, EndpointResult
from .engines import SAMPLE_RATE, VADEngine


@dataclass
class CaptureStats:
    chunks: int = 0
    vad_ms_total: float = 0.0
    vad_ms_max: float = 0.0
    probs: list[float] = field(default_factory=list)


class CaptureEndpointer:
    def __init__(self, engine: VADEngine, params: EndpointParams, keep_probs: bool = False):
        self.engine = engine
        self.params = params
        self.ep = Endpointer(params)
        self.keep_probs = keep_probs
        self._buf = np.zeros(0, dtype=np.float32)
        self.stats = CaptureStats()
        self.active = False

    @property
    def chunk_ms(self) -> float:
        return 1000.0 * self.engine.chunk_samples / SAMPLE_RATE

    def start(self) -> None:
        self.engine.reset()
        self.ep.reset()
        self._buf = np.zeros(0, dtype=np.float32)
        self.stats = CaptureStats()
        self.active = True

    def stop(self) -> None:
        self.active = False

    @property
    def done(self) -> bool:
        return self.ep.done

    def result(self) -> EndpointResult:
        return self.ep.result()

    def push(self, frame: np.ndarray) -> list[tuple[str, float]]:
        """Feed float32 mono 16 kHz audio. Returns [(event, offset_ms), ...] that
        happened inside this frame (speech_started / speech_ended / no_speech /
        max_window). Ignored when inactive or already done."""
        if not self.active or self.ep.done:
            return []
        self._buf = np.concatenate([self._buf, np.asarray(frame, dtype=np.float32).reshape(-1)])
        n = self.engine.chunk_samples
        events: list[tuple[str, float]] = []
        while self._buf.size >= n and not self.ep.done:
            chunk, self._buf = self._buf[:n], self._buf[n:]
            t0 = time.perf_counter()
            p = self.engine.prob(chunk)
            dt = (time.perf_counter() - t0) * 1000.0
            s = self.stats
            s.chunks += 1
            s.vad_ms_total += dt
            s.vad_ms_max = max(s.vad_ms_max, dt)
            if self.keep_probs:
                s.probs.append(round(p, 3))
            ev = self.ep.push(p, self.chunk_ms)
            if ev:
                at = (self.ep.speech_start_ms if ev == "speech_started"
                      else self.ep.speech_end_ms if ev == "speech_ended" else self.ep.elapsed_ms)
                events.append((ev, round(float(at or 0.0), 1)))
        if self.ep.done:
            self.active = False
        return events

    def summary(self) -> dict:
        r = self.result()
        s = self.stats
        return {
            "engine": self.engine.name,
            "generation": getattr(self.engine, "generation", None),
            "reason": r.reason,
            "elapsed_ms": round(r.elapsed_ms, 1),
            "speech_start_ms": None if r.speech_start_ms is None else round(r.speech_start_ms, 1),
            "speech_end_ms": None if r.speech_end_ms is None else round(r.speech_end_ms, 1),
            "hangover_ms": self.params.hangover_ms,
            "vad_chunks": s.chunks,
            "vad_ms_mean": round(s.vad_ms_total / s.chunks, 3) if s.chunks else None,
            "vad_ms_max": round(s.vad_ms_max, 3),
        }
