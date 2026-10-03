"""§3e: pipeline wiring tests for the VAD endpointing plugin (VAD_PLUGIN_GUIDE.md)."""
import asyncio
from types import SimpleNamespace

import numpy as np

from edge.media import MediaController, SimulatedBackend
from edge.pipeline import EdgePipeline, EdgeSettings, Phase
from edge.registry import ModelRegistry
from edge.wake import ManualWake, WakeEngine, WakeInfo


class FakeEndpointer:
    """Mimics CaptureEndpointer's public surface with scripted events."""

    def __init__(self, done_after_frames=10, reason="end_of_speech", elapsed_ms=400.0,
                 max_window_ms=4000.0, emit=True):
        self.params = SimpleNamespace(max_window_ms=max_window_ms)
        self._done_after = done_after_frames
        self._reason = reason
        self._elapsed = elapsed_ms
        self._emit = emit
        self.active = False
        self.done = False
        self.n_frames = 0
        self.push_calls = 0

    def start(self):
        self.active = True
        self.done = False
        self.n_frames = 0

    def stop(self):
        self.active = False

    def push(self, frame):
        self.push_calls += 1
        if not self.active or self.done:
            return []
        self.n_frames += 1
        if self.n_frames >= self._done_after:
            self.done = True
            if self._emit:
                ev = "speech_ended" if self._reason == "end_of_speech" else self._reason
                return [(ev, self._elapsed)]
        return []

    def summary(self):
        return {"engine": "fake", "generation": None, "reason": self._reason,
                "elapsed_ms": self._elapsed, "speech_start_ms": 0.0,
                "speech_end_ms": self._elapsed if self._reason != "no_speech" else None,
                "hangover_ms": 500.0, "vad_chunks": self.n_frames,
                "vad_ms_mean": 0.2, "vad_ms_max": 0.3}


class SpyWake(WakeEngine):
    def __init__(self):
        self.info = WakeInfo("spy", "Spy wake", "spy", "ready")
        self.process_calls = 0

    def process(self, pcm_int16):
        self.process_calls += 1
        return None

    def reset(self):
        pass


def make(tmp_path, endpointer=None, **kw):
    msgs: list[dict] = []

    async def emit(m):
        msgs.append(m)

    reg = ModelRegistry(tmp_path)
    reg.scan()
    p = EdgePipeline(reg, {"manual": ManualWake(), "spy": SpyWake()},
                     MediaController(backend=SimulatedBackend()),
                     EdgeSettings(result_hold_s=0.0, pre_capture_delay_s=0.0, **kw),
                     emit, endpointer=endpointer,
                     endpointing_status="fake" if endpointer else "disabled (fixed window)")
    return p, msgs


def events(msgs):
    return [m["event"] for m in msgs if m["type"] == "event"]


def feed(p, n, sleep=0.002):
    async def _feed():
        for _ in range(n):
            await p.feed_pcm((np.ones(320) * 1000).astype("<i2").tobytes())
            await asyncio.sleep(sleep)
    return asyncio.create_task(_feed())


def test_vad_endpoint_closes_capture_early(tmp_path):
    async def run():
        ep = FakeEndpointer(done_after_frames=20, reason="end_of_speech", elapsed_ms=400.0)
        p, msgs = make(tmp_path, endpointer=ep)
        p.s.active_vcm = "simulator"
        p.registry.scripted().next_class = "VOLUME_UP"

        feeder = feed(p, 100)
        await asyncio.sleep(0.02)
        r = await p.run_command("live")
        await feeder
        assert r is not None and r.accepted
        assert ep.done and r.clip_ms < 1000          # closed well before the 4 s window
        types = [e["type"] for e in events(msgs)]
        assert "capture_endpoint" in types and "vad_speech_ended" in types
        listen = [m for m in msgs if m["type"] == "phase" and m["phase"] == "listening"]
        assert listen and listen[0].get("mode") == "vad"
    asyncio.run(run())


def test_vad_no_speech_fails_and_restores_media(tmp_path):
    async def run():
        ep = FakeEndpointer(done_after_frames=20, reason="no_speech", elapsed_ms=2000.0)
        p, msgs = make(tmp_path, endpointer=ep)
        p.s.active_vcm = "simulator"
        p.media.play()

        feeder = feed(p, 100)
        await asyncio.sleep(0.02)
        r = await p.run_command("live")
        await feeder
        assert r is None
        assert any(e["type"] == "capture_failed"
                   and (e.get("payload") or {}).get("reason") == "no speech heard"
                   for e in events(msgs))
        assert not p.media.ducked                     # ducked music restored
        assert p.phase.value == "idle"
    asyncio.run(run())


def test_vad_max_window_still_infers(tmp_path):
    async def run():
        ep = FakeEndpointer(done_after_frames=30, reason="max_window", elapsed_ms=400.0,
                            max_window_ms=400.0)
        p, msgs = make(tmp_path, endpointer=ep)
        p.s.active_vcm = "simulator"
        p.registry.scripted().next_class = "VOLUME_UP"

        feeder = feed(p, 100)
        await asyncio.sleep(0.02)
        r = await p.run_command("live")
        await feeder
        assert r is not None
        assert any(e["type"] == "capture_endpoint"
                   and (e.get("payload") or {}).get("reason") == "max_window"
                   for e in events(msgs))
    asyncio.run(run())


def test_disabled_endpointer_is_fixed_window(tmp_path):
    async def run():
        p, msgs = make(tmp_path, window_s=0.3)        # endpointer=None
        p.s.active_vcm = "simulator"
        p.registry.scripted().next_class = "VOLUME_UP"

        feeder = feed(p, 60, sleep=0.005)
        await asyncio.sleep(0.05)
        r = await p.run_command("live")
        await feeder
        assert r is not None
        types = [e["type"] for e in events(msgs)]
        assert "capture_endpoint" not in types
        assert not any(t.startswith("vad_") for t in types)
        listen = [m for m in msgs if m["type"] == "phase" and m["phase"] == "listening"]
        assert listen and listen[0].get("mode") == "fixed"
    asyncio.run(run())


def test_wake_and_vad_isolation_in_feed_pcm(tmp_path):
    async def run():
        ep = FakeEndpointer(done_after_frames=100, reason="end_of_speech", elapsed_ms=400.0)
        p, _ = make(tmp_path, endpointer=ep)
        p.s.active_wake = "spy"
        spy = p.wake_engines["spy"]
        frame = (np.ones(320) * 1000).astype("<i2").tobytes()

        # idle: wake.process runs, VAD push does not
        p.phase = Phase.IDLE
        p._capture = None
        await p.feed_pcm(frame)
        assert spy.process_calls == 1
        assert ep.push_calls == 0

        # capturing: VAD push runs, wake.process does not
        p.phase = Phase.LISTENING
        p._capture = []
        ep.start()
        await p.feed_pcm(frame)
        assert spy.process_calls == 1          # unchanged
        assert ep.push_calls == 1              # VAD ran
    asyncio.run(run())
