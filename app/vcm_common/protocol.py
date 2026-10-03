"""Wire contracts between edge (RPi) and laptop, and laptop and browser.

Edge <-> laptop runs over one WebSocket: binary frames are 16 kHz mono PCM16
little-endian audio; text frames are JSON objects with a ``type`` field.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

AUDIO_SAMPLE_RATE_HZ = 16_000
AUDIO_FRAME_MS = 20
AUDIO_FRAME_SAMPLES = AUDIO_SAMPLE_RATE_HZ * AUDIO_FRAME_MS // 1000  # 320
AUDIO_BYTES_PER_SAMPLE = 2

ResultSource = Literal["live", "replay", "simulated"]


def now_ms() -> float:
    """Wall-clock epoch time in milliseconds."""
    return time.time() * 1000.0


def new_command_id() -> str:
    return uuid.uuid4().hex[:12]


class ModelPrediction(BaseModel):
    """One model's output for one command clip."""

    model_id: str
    class_key: str
    intent: str
    slot: Optional[str] = None
    confidence: float
    top_k: list[tuple[str, float]] = Field(default_factory=list)
    feature_ms: float = 0.0
    inference_ms: float = 0.0


class InferenceResult(BaseModel):
    """What the edge sends after classifying one command clip.

    Only ``primary`` (the active model) can trigger an action; ``compare``
    holds side-by-side predictions from challenger models on the SAME clip.
    """

    command_id: str
    source: ResultSource
    primary: ModelPrediction
    accepted: bool
    reject_reason: Optional[str] = None
    threshold: float
    compare: list[ModelPrediction] = Field(default_factory=list)
    members: list[ModelPrediction] = Field(default_factory=list)  # agreement engine
    clip_ms: float
    clip_rms_dbfs: Optional[float] = None
    media_handled: bool = False
    t_wake_ms: Optional[float] = None
    t_capture_end_ms: float
    t_result_ms: float
    expected_class: Optional[str] = None
    # clock-sync fields, filled in by the laptop hub at ingress (never by the edge)
    t_received_ms: Optional[float] = None
    edge_clock_offset_ms: Optional[float] = None
    timing_valid: bool = True
    edge_times: Optional[dict] = None
    clock_quality: Optional[str] = None   # local | accurate | approximate | syncing (hub-set)


class Event(BaseModel):
    """Timeline event (edge or laptop origin)."""

    type: str
    origin: Literal["edge", "laptop"]
    ts_ms: float = Field(default_factory=now_ms)
    command_id: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)
