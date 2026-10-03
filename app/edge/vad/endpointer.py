"""Endpointing state machine: decides when the spoken command is over.

Pure Python, no audio, no ML: it consumes one speech probability per chunk
and the chunk duration. Easy to unit-test and to tune.

    WAIT_SPEECH --(speech for >= min_speech_ms)--> SPEECH
    SPEECH --(silence for >= hangover_ms)--> DONE  reason="end_of_speech"
    WAIT_SPEECH --(no speech within no_speech_timeout_ms)--> DONE reason="no_speech"
    any --(elapsed >= max_window_ms)--> DONE reason="max_window"

Hysteresis: a chunk counts as speech when p >= onset_threshold while waiting
for speech, and stays "speech" while p >= offset_threshold once speaking
(offset < onset avoids flapping on soft syllables).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EndpointParams:
    onset_threshold: float = 0.5
    offset_threshold: float = 0.35
    min_speech_ms: float = 200.0
    hangover_ms: float = 500.0
    no_speech_timeout_ms: float = 2000.0
    max_window_ms: float = 4000.0

    def validate(self) -> None:
        if not 0.0 < self.offset_threshold <= self.onset_threshold < 1.0:
            raise ValueError("need 0 < offset_threshold <= onset_threshold < 1")
        for name in ("min_speech_ms", "hangover_ms", "no_speech_timeout_ms", "max_window_ms"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.no_speech_timeout_ms > self.max_window_ms:
            raise ValueError("no_speech_timeout_ms must be <= max_window_ms")


@dataclass
class EndpointResult:
    done: bool
    reason: Optional[str]              # end_of_speech | no_speech | max_window | None
    elapsed_ms: float
    speech_start_ms: Optional[float]   # offsets from capture start
    speech_end_ms: Optional[float]


class Endpointer:
    WAIT, SPEECH, DONE = "wait_speech", "speech", "done"

    def __init__(self, params: EndpointParams):
        params.validate()
        self.p = params
        self.reset()

    def reset(self) -> None:
        self.state = self.WAIT
        self.elapsed_ms = 0.0
        self._run_ms = 0.0          # consecutive speech while waiting
        self._silence_ms = 0.0      # consecutive silence while speaking
        self.speech_start_ms: Optional[float] = None
        self.speech_end_ms: Optional[float] = None
        self.reason: Optional[str] = None

    @property
    def done(self) -> bool:
        return self.state == self.DONE

    def result(self) -> EndpointResult:
        return EndpointResult(self.done, self.reason, self.elapsed_ms,
                              self.speech_start_ms, self.speech_end_ms)

    def push(self, prob: float, chunk_ms: float) -> Optional[str]:
        """Feed one chunk. Returns 'speech_started', 'speech_ended' (endpoint),
        'no_speech', 'max_window' when that happens on this chunk, else None."""
        if self.done:
            return None
        self.elapsed_ms += chunk_ms
        event: Optional[str] = None
        if self.state == self.WAIT:
            if prob >= self.p.onset_threshold:
                self._run_ms += chunk_ms
                if self._run_ms >= self.p.min_speech_ms:
                    self.state = self.SPEECH
                    self.speech_start_ms = self.elapsed_ms - self._run_ms
                    self._silence_ms = 0.0
                    event = "speech_started"
            else:
                self._run_ms = 0.0
                if self.elapsed_ms >= self.p.no_speech_timeout_ms:
                    return self._finish("no_speech")
        elif self.state == self.SPEECH:
            if prob >= self.p.offset_threshold:
                self._silence_ms = 0.0
            else:
                self._silence_ms += chunk_ms
                if self._silence_ms >= self.p.hangover_ms:
                    self.speech_end_ms = self.elapsed_ms - self._silence_ms
                    return self._finish("end_of_speech")
        if self.elapsed_ms >= self.p.max_window_ms:
            if self.state == self.SPEECH:
                self.speech_end_ms = self.elapsed_ms - self._silence_ms
            return self._finish("max_window")
        return event

    def _finish(self, reason: str) -> str:
        self.state = self.DONE
        self.reason = reason
        return "speech_ended" if reason == "end_of_speech" else reason
