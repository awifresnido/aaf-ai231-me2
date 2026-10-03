"""Tests for the VAD endpointing plugin (edge/vad). Silero tests are skipped
when onnxruntime or silero_vad.onnx is unavailable."""
from __future__ import annotations

import numpy as np
import pytest

from edge.vad import (CaptureEndpointer, EndpointingConfig, EndpointParams, Endpointer,
                      EnergyVAD, build_endpointer)

CH = 32.0  # ms per 512-sample chunk


def run(probs, **kw):
    ep = Endpointer(EndpointParams(**kw))
    events = [ep.push(p, CH) for p in probs]
    return ep, [e for e in events if e]


# ---------------------------------------------------------------- endpointer
def test_end_of_speech_after_hangover():
    probs = [0.0] * 5 + [0.9] * 30 + [0.0] * 20          # speech 160..1120 ms
    ep, ev = run(probs, hangover_ms=500)
    assert ev == ["speech_started", "speech_ended"]
    r = ep.result()
    assert r.reason == "end_of_speech"
    assert r.speech_start_ms == pytest.approx(160)
    assert r.speech_end_ms == pytest.approx(1120)
    assert r.elapsed_ms == pytest.approx(1120 + 512)       # first chunk at/after 500 ms silence


def test_short_pause_does_not_end():
    probs = [0.9] * 20 + [0.1] * 10 + [0.9] * 20 + [0.0] * 20   # 320 ms pause < 500 ms
    ep, ev = run(probs, hangover_ms=500)
    assert ev == ["speech_started", "speech_ended"]
    assert ep.result().speech_end_ms == pytest.approx(50 * CH)


def test_hysteresis_keeps_soft_syllables():
    probs = [0.9] * 10 + [0.4] * 20 + [0.0] * 20          # 0.4 >= offset 0.35, < onset 0.5
    ep, _ = run(probs, hangover_ms=300)
    assert ep.result().speech_end_ms == pytest.approx(30 * CH)


def test_blip_shorter_than_min_speech_is_ignored():
    probs = [0.9] * 3 + [0.0] * 80                         # 96 ms < 200 ms
    ep, ev = run(probs, no_speech_timeout_ms=2000)
    assert ev == ["no_speech"]
    assert ep.result().speech_start_ms is None


def test_no_speech_timeout():
    ep, ev = run([0.0] * 100, no_speech_timeout_ms=1500)
    assert ev == ["no_speech"] and ep.result().elapsed_ms >= 1500


def test_max_window_cap_while_talking():
    ep, ev = run([0.9] * 200, max_window_ms=4000)
    assert ev[-1] == "max_window"
    assert ep.result().elapsed_ms == pytest.approx(4000, abs=CH)
    assert ep.result().speech_end_ms is not None


def test_no_events_after_done():
    ep, _ = run([0.0] * 100, no_speech_timeout_ms=500)
    assert ep.push(0.9, CH) is None


def test_param_validation():
    with pytest.raises(ValueError):
        EndpointParams(onset_threshold=0.3, offset_threshold=0.5).validate()
    with pytest.raises(ValueError):
        EndpointParams(no_speech_timeout_ms=5000, max_window_ms=4000).validate()


# ---------------------------------------------------------------- stream
class FakeEngine:
    name, chunk_samples = "fake", 512

    def __init__(self, probs):
        self.probs, self.i, self.resets = probs, 0, 0

    def reset(self):
        self.i, self.resets = 0, self.resets + 1

    def prob(self, chunk):
        assert chunk.shape == (512,)
        p = self.probs[min(self.i, len(self.probs) - 1)]
        self.i += 1
        return p


def test_stream_rechunks_20ms_frames_and_reports_offsets():
    eng = FakeEngine([0.0] * 5 + [0.9] * 30 + [0.0] * 40)
    ce = CaptureEndpointer(eng, EndpointParams(hangover_ms=500))
    ce.start()
    events = []
    for _ in range(400):                                    # 20 ms frames
        events += ce.push(np.zeros(320, np.float32))
        if ce.done:
            break
    assert [e for e, _ in events] == ["speech_started", "speech_ended"]
    assert events[0][1] == pytest.approx(160) and events[1][1] == pytest.approx(1120)
    assert ce.done and not ce.active
    s = ce.summary()
    assert s["reason"] == "end_of_speech" and s["vad_chunks"] == eng.i


def test_stream_inactive_until_start_and_resets_engine():
    eng = FakeEngine([0.9] * 100)
    ce = CaptureEndpointer(eng, EndpointParams())
    assert ce.push(np.zeros(320, np.float32)) == []        # not started: no VAD work
    assert eng.i == 0
    ce.start()
    ce.start()
    assert eng.resets == 2


# ---------------------------------------------------------------- config / factory
def test_config_defaults_disabled_means_fixed_window():
    ce, status = build_endpointer(EndpointingConfig.from_dict({}))
    assert ce is None and "fixed window" in status


def test_config_rejects_unknown_keys_and_engine():
    with pytest.raises(ValueError):
        EndpointingConfig.from_dict({"enabled": True, "hangover": 500})
    with pytest.raises(ValueError):
        EndpointingConfig.from_dict({"enabled": True, "engine": "webrtc"})


def test_missing_silero_falls_back_to_energy():
    cfg = EndpointingConfig.from_dict({"enabled": True, "engine": "silero",
                                       "model_path": "/nonexistent/silero.onnx",
                                       "search_roots": ["/nonexistent"]})
    ce, status = build_endpointer(cfg)
    if ce is not None and ce.engine.name == "silero":
        pytest.skip("a silero model was found via $VCM_SILERO_VAD / models/vad / openwakeword")
    assert ce is not None and ce.engine.name == "energy" and "fallback" in status


def test_missing_silero_without_fallback_disables():
    cfg = EndpointingConfig.from_dict({"enabled": True, "model_path": "/nonexistent.onnx",
                                       "search_roots": ["/nonexistent"],
                                       "fallback_to_energy": False})
    ce, status = build_endpointer(cfg)
    if ce is not None:
        pytest.skip("a silero model was found elsewhere")
    assert "unavailable" in status


def test_energy_vad_detects_tone_burst_over_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1e-3, 16000 * 3).astype(np.float32)
    t = np.arange(16000) / 16000
    x[16000:32000] += 0.2 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    ce = CaptureEndpointer(EnergyVAD(), EndpointParams(hangover_ms=300))
    ce.start()
    for i in range(0, x.size, 320):
        ce.push(x[i:i + 320])
    r = ce.result()
    assert r.reason == "end_of_speech"
    assert 900 <= r.speech_start_ms <= 1100 and 1900 <= r.speech_end_ms <= 2100


# ---------------------------------------------------------------- silero (optional)
def _silero():
    pytest.importorskip("onnxruntime")
    from edge.vad import SileroVAD, find_silero_model
    try:
        return SileroVAD(find_silero_model(search_roots=("../openwakeword",)))
    except FileNotFoundError:
        pytest.skip("silero_vad.onnx not available")


def test_silero_loads_and_is_quiet_on_silence():
    eng = _silero()
    assert eng.generation in ("v4", "v5")
    probs = [eng.prob(np.zeros(512, np.float32)) for _ in range(20)]
    assert max(probs) < 0.3


def test_silero_ignores_steady_tones():
    eng = _silero()
    t = np.arange(512 * 60) / 16000
    x = (0.05 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 330 * t))).astype(np.float32)
    probs = [eng.prob(x[i:i + 512]) for i in range(0, x.size, 512)]
    assert np.mean(probs) < 0.5
