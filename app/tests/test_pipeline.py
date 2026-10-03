import asyncio
import base64
import io
import wave

import numpy as np

from edge.media import MediaController, SimulatedBackend
from edge.pipeline import EdgePipeline, EdgeSettings
from edge.registry import ModelRegistry
from edge.wake import ManualWake


def make(tmp_path, **kw):
    msgs: list[dict] = []

    async def emit(m):
        msgs.append(m)

    reg = ModelRegistry(tmp_path)
    reg.scan()
    p = EdgePipeline(reg, {"manual": ManualWake()}, MediaController(backend=SimulatedBackend()),
                     EdgeSettings(result_hold_s=0.0, **kw), emit)
    return p, msgs


def results(msgs):
    return [m["result"] for m in msgs if m["type"] == "inference_result"]


def test_simulate_flow_and_ducking(tmp_path):
    async def run():
        p, msgs = make(tmp_path)
        p.media.play()
        r = await p.run_command("simulated", sim_class="VOLUME_UP")
        assert r.accepted and r.source == "simulated" and r.media_handled
        assert p.media.level == 4 and not p.media.ducked
        phases = [m["phase"] for m in msgs if m["type"] == "phase"]
        assert phases == ["wake_detected", "listening", "inferencing", "result", "idle"]
        types = [m["event"]["type"] for m in msgs if m["type"] == "event"]
        assert "media_ducked" in types
    asyncio.run(run())


def test_non_command_is_rejected(tmp_path):
    async def run():
        p, _ = make(tmp_path)
        r = await p.run_command("simulated", sim_class="SILENCE")
        assert not r.accepted and "not a command" in r.reject_reason
    asyncio.run(run())


def test_live_without_audio_fails_cleanly(tmp_path):
    async def run():
        p, msgs = make(tmp_path)
        assert await p.run_command("live") is None
        assert any(m["type"] == "event" and m["event"]["type"] == "capture_failed" for m in msgs)
        assert p.phase.value == "idle"
    asyncio.run(run())


def test_live_capture_uses_streamed_audio(tmp_path):
    async def run():
        p, _ = make(tmp_path, window_s=0.3)
        p.registry.scripted().next_class = None

        async def stream():
            for _ in range(40):
                await p.feed_pcm((np.ones(320) * 1000).astype("<i2").tobytes())
                await asyncio.sleep(0.02)

        feeder = asyncio.create_task(stream())
        await asyncio.sleep(0.05)
        p.s.active_vcm = "simulator"
        r = await p.run_command("live")
        await feeder
        assert r is not None and r.clip_ms > 100
    asyncio.run(run())


def test_unavailable_model_reports_error(tmp_path):
    async def run():
        p, msgs = make(tmp_path, active_vcm="does_not_exist")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
            w.writeframes((np.zeros(16000)).astype("<i2").tobytes())
        await p.handle({"type": "replay", "wav_b64": base64.b64encode(buf.getvalue()).decode()})
        await asyncio.sleep(0.2)
        assert any(m["type"] == "event" and m["event"]["type"] == "inference_error" for m in msgs)
        assert results(msgs) == []
    asyncio.run(run())
