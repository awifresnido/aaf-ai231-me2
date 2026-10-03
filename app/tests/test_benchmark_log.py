import asyncio
import json

import numpy as np

from edge.benchmark_log import BenchmarkLog
from edge.media import MediaController, SimulatedBackend
from edge.pipeline import EdgePipeline, EdgeSettings
from edge.registry import ModelRegistry
from edge.wake import ManualWake


def _lines(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_benchmark_log_wake(tmp_path):
    b = BenchmarkLog(enabled=True, log_dir=tmp_path)
    b.wake()
    b.close()
    files = list(tmp_path.glob("*.log"))
    assert len(files) == 1
    assert _lines(files[0]) == [{"event": "wake"}]


def test_benchmark_log_decision(tmp_path):
    b = BenchmarkLog(enabled=True, log_dir=tmp_path)
    b.decision("TIMER", "30 seconds", 12.34, 1500.0)
    b.close()
    line = _lines(list(tmp_path.glob("*.log"))[0])[0]
    assert line == {"intent": "TIMER", "slot": "30 seconds", "infer_ms": 12.3, "audio_ms": 1500}


def test_benchmark_log_disabled_writes_nothing(tmp_path):
    b = BenchmarkLog(enabled=False, log_dir=tmp_path)
    b.wake()
    b.decision("X", "", 1.0, 1.0)
    b.close()
    assert list(tmp_path.glob("*.log")) == []


def _make(tmp_path, **kw):
    msgs = []

    async def emit(m):
        msgs.append(m)

    reg = ModelRegistry(tmp_path)
    reg.scan()
    p = EdgePipeline(reg, {"manual": ManualWake()}, MediaController(backend=SimulatedBackend()),
                     EdgeSettings(result_hold_s=0.0, **kw), emit)
    return p, msgs


class _Spy:
    def __init__(self):
        self.wakes = 0
        self.decisions = []

    def wake(self):
        self.wakes += 1

    def decision(self, intent, slot, infer_ms, audio_ms):
        self.decisions.append((intent, slot))


def test_benchmark_mode_one_wake_one_decision(tmp_path):
    """Part F 5b: one live wake + one command -> exactly one wake line + one decision line."""
    async def run():
        p, _ = _make(tmp_path, benchmark_mode=True, window_s=0.3)
        spy = _Spy()
        p.benchmark_log = spy
        p.registry.scripted().next_class = "TIMER|30 seconds"

        async def stream():
            for _ in range(40):
                await p.feed_pcm((np.ones(320) * 1000).astype("<i2").tobytes())
                await asyncio.sleep(0.02)

        feeder = asyncio.create_task(stream())
        await asyncio.sleep(0.05)
        p.s.active_vcm = "simulator"
        r = await p.run_command("live")
        await feeder
        assert r is not None and r.accepted
        assert spy.wakes == 1
        assert spy.decisions == [("TIMER", "30 seconds")]
    asyncio.run(run())


def test_benchmark_mode_rejected_is_out_of_scope(tmp_path):
    """Below-threshold / non-command -> OUT_OF_SCOPE decision line."""
    async def run():
        p, _ = _make(tmp_path, benchmark_mode=True, window_s=0.3)
        spy = _Spy()
        p.benchmark_log = spy
        p.registry.scripted().next_class = "SILENCE"

        async def stream():
            for _ in range(40):
                await p.feed_pcm((np.ones(320) * 1000).astype("<i2").tobytes())
                await asyncio.sleep(0.02)

        feeder = asyncio.create_task(stream())
        await asyncio.sleep(0.05)
        p.s.active_vcm = "simulator"
        r = await p.run_command("live")
        await feeder
        assert r is not None and not r.accepted
        assert spy.decisions and spy.decisions[-1][0] == "OUT_OF_SCOPE"
    asyncio.run(run())


def test_benchmark_mode_off_no_log(tmp_path):
    """benchmark_mode off -> the pipeline never emits benchmark lines."""
    async def run():
        p, _ = _make(tmp_path, benchmark_mode=False, window_s=0.3)
        spy = _Spy()
        p.benchmark_log = spy
        p.registry.scripted().next_class = "VOLUME_UP"

        async def stream():
            for _ in range(40):
                await p.feed_pcm((np.ones(320) * 1000).astype("<i2").tobytes())
                await asyncio.sleep(0.02)

        feeder = asyncio.create_task(stream())
        await asyncio.sleep(0.05)
        p.s.active_vcm = "simulator"
        await p.run_command("live")
        await feeder
        assert spy.wakes == 0 and spy.decisions == []
    asyncio.run(run())
