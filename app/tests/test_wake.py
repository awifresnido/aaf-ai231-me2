"""Wake-word engine: gate logic, framing, pipeline trigger, and (when the
openWakeWord bench is present) the real hey_rhasspy model on reference clips."""
import asyncio
import wave
from pathlib import Path

import numpy as np
import pytest

from edge.media import MediaController, SimulatedBackend
from edge.pipeline import EdgePipeline, EdgeSettings
from edge.registry import ModelRegistry
from edge.wake import OWW_FRAME_SAMPLES, ManualWake, OpenWakeWordEngine, ScoreGate

REPO = Path(__file__).resolve().parents[1]
OWW = REPO.parent / "openwakeword"
PRE = OWW / "models" / "pretrained"


class FakeModel:
    """Returns scripted scores, one per 1280-sample frame."""

    def __init__(self, scores, **kwargs):
        self.scores = list(scores)
        self.kwargs = kwargs
        self.frames = 0

    def predict(self, frame):
        assert frame.size == OWW_FRAME_SAMPLES and frame.dtype == np.int16
        self.frames += 1
        return {"m": self.scores.pop(0) if self.scores else 0.0}

    def reset(self):
        pass


def fake_engine(tmp_path, scores, **kw):
    f = tmp_path / "m.onnx"
    f.write_bytes(b"x")
    holder = {}

    def factory(**kwargs):
        holder["m"] = FakeModel(scores, **kwargs)
        return holder["m"]

    t = {"now": 0.0}
    e = OpenWakeWordEngine("oww:test", str(f), None, None, model_factory=factory,
                           clock=lambda: t["now"], **kw)
    return e, holder["m"], t


def test_gate_patience_and_debounce():
    g = ScoreGate(threshold=0.5, patience=2, debounce_s=1.0)
    assert [g.update(s, 0.0) for s in (0.9, 0.2, 0.9)] == [False, False, False]
    assert g.update(0.9, 0.1) is True          # 2 consecutive
    assert g.update(0.9, 0.2) is False
    assert g.update(0.9, 0.3) is False         # patience met again but debounced
    assert g.update(0.2, 0.4) is False         # score drop resets the run
    assert g.update(0.9, 1.2) is False         # debounce over, but patience not yet met
    assert g.update(0.9, 1.3) is True


def test_framing_carries_remainder(tmp_path):
    e, m, _ = fake_engine(tmp_path, [0.0] * 10, patience=1)
    assert e.info.status == "ready"
    e.process(np.zeros(1000, dtype=np.int16))
    assert m.frames == 0
    e.process(np.zeros(1600, dtype=np.int16))   # 2600 total -> 2 frames, 40 carried
    assert m.frames == 2
    assert m.kwargs["inference_framework"] == "onnx"


def test_fires_and_reports_peak(tmp_path):
    e, _, _ = fake_engine(tmp_path, [0.1, 0.7, 0.8, 0.1], threshold=0.5, patience=2)
    out = [e.process(np.zeros(OWW_FRAME_SAMPLES, dtype=np.int16)) for _ in range(4)]
    assert out == [None, None, 0.8, None]
    assert e.pop_peak() == 0.8 and e.pop_peak() is None


def test_missing_files_unavailable_not_crash(tmp_path):
    e = OpenWakeWordEngine("oww:x", str(tmp_path / "nope.onnx"), None, None)
    assert e.info.status == "unavailable" and "missing" in e.info.detail
    assert e.process(np.zeros(2560, dtype=np.int16)) is None


def test_pipeline_wakes_on_audio_and_chime_delay(tmp_path):
    async def run():
        msgs = []

        async def emit(m):
            msgs.append(m)

        e, _, _ = fake_engine(tmp_path, [0.9, 0.9] + [0.0] * 200, patience=2)
        reg = ModelRegistry(tmp_path)
        reg.scan()
        p = EdgePipeline(reg, {"manual": ManualWake(), e.info.id: e},
                         MediaController(backend=SimulatedBackend()),
                         EdgeSettings(window_s=0.2, result_hold_s=0.0, active_vcm="simulator",
                                      active_wake=e.info.id, pre_capture_delay_s=0.1), emit)
        for _ in range(60):   # 60 x 20 ms of audio
            await p.feed_pcm(np.zeros(320, dtype="<i2").tobytes())
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.5)
        evs = [m["event"] for m in msgs if m["type"] == "event"]
        wake = next(ev for ev in evs if ev["type"] == "wake_detected")
        assert wake["payload"]["wake_engine"] == "oww:test" and wake["payload"]["confidence"] == 0.9
        started = next(ev for ev in evs if ev["type"] == "capture_started")
        assert started["ts_ms"] - wake["ts_ms"] >= 90          # chime gap honoured
        assert any("wake_score" in m for m in msgs if m["type"] == "level")
    asyncio.run(run())


def _wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def _real_engine():
    pytest.importorskip("openwakeword")
    if not (PRE / "hey_rhasspy_v0.1.onnx").exists():
        pytest.skip("openWakeWord bench models not present")
    e = OpenWakeWordEngine("oww:hey_rhasspy", str(PRE / "hey_rhasspy_v0.1.onnx"),
                           str(PRE / "melspectrogram.onnx"), str(PRE / "embedding_model.onnx"),
                           threshold=0.5, patience=2)
    assert e.info.status == "ready", e.info.detail
    return e


def _run_clip(e, x):
    pad = np.zeros(16000, dtype=np.int16)
    fired = None
    for chunk in np.array_split(np.concatenate([pad, x, pad]), 60):  # uneven chunks on purpose
        fired = fired or e.process(chunk)
    return fired


def test_real_model_positive_control():
    e = _real_engine()
    clip = OWW / "assets" / "hey_rhasspy_tts_16k.wav"
    if not clip.exists():
        pytest.skip("positive-control clip not present")
    assert _run_clip(e, _wav(clip)) is not None


def test_real_model_negative_control():
    e = _real_engine()
    clip = OWW / "tests" / "data" / "alexa_test.wav"
    if not clip.exists():
        pytest.skip("negative-control clip not present")
    assert _run_clip(e, _wav(clip)) is None
