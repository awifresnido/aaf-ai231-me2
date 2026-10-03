"""Benchmark runner state machine, driven with a stub host + synthetic events."""
import asyncio
from dataclasses import replace

import pytest

from laptop.app.benchmark.config import load_benchmark_config
from laptop.app.benchmark.runner import BenchmarkRunner
from laptop.app.benchmark.store import BenchmarkStore
from laptop.app.main import REPO_ROOT
from vcm_common.ontology import get_ontology
from vcm_common.protocol import InferenceResult, ModelPrediction


class StubHost:
    def __init__(self):
        self.edge_state = {"active_vcm": "B2_s0", "threshold_override": None,
                           "models": [{"id": "B2_s0", "min_confidence": 0.8, "params": 67793,
                                       "size_bytes": 274765, "engine": "onnx_tinyvcm"}]}
        self.labels = {}
        self.push_count = 0

    async def push_state(self):
        self.push_count += 1

    @property
    def log(self):
        return self

    def label(self, command_id, expected_class):
        self.labels[command_id] = expected_class


def make(tmp_path):
    o = get_ontology()
    cfg = load_benchmark_config(REPO_ROOT / "config" / "benchmark.yaml", o)
    cfg = replace(cfg, wake_timeout_s=0.05, result_timeout_s=0.05)
    store = BenchmarkStore(tmp_path / "benchmark.sqlite")
    host = StubHost()
    runner = BenchmarkRunner(store, cfg, o, None, host, auto_advance_s=0.02)
    return runner, store, host


def _wake():
    return {"type": "wake_detected", "command_id": "cid_t", "ts_ms": 1000.0,
            "payload": {"source": "live", "wake_engine": "oww:hey_rhasspy", "confidence": 0.9}}


def _result(source="live", class_key="PLAY_MUSIC", accepted=True, confidence=0.9):
    intent, _, slot = class_key.partition("|")
    return InferenceResult(
        command_id="cid_t", source=source,
        primary=ModelPrediction(model_id="B2_s0", class_key=class_key, intent=intent,
                                slot=slot or None, confidence=confidence,
                                feature_ms=12.0, inference_ms=34.0),
        accepted=accepted, threshold=0.8, clip_ms=500.0,
        t_capture_end_ms=1500.0, t_result_ms=1540.0)


async def _started(tmp_path):
    runner, store, host = make(tmp_path)
    sess = await runner.create(set_type="fixed", cond_noise="quiet", cond_distance="near",
                               consent_results=True, consent_audio=False)
    await runner.start(sess["session_id"])
    return runner, store, host, sess


def test_wake_miss(tmp_path):
    async def run():
        runner, store, _, sess = await _started(tmp_path)
        assert runner.phase == "await_wake"
        await asyncio.sleep(0.12)
        t0 = store.trials_for_session(sess["session_id"])[0]
        assert t0["outcome"] == "wake_miss"
        assert runner.idx >= 1
    asyncio.run(run())


def test_no_result_after_wake(tmp_path):
    async def run():
        runner, store, _, sess = await _started(tmp_path)
        await runner.on_wake(_wake())
        assert runner.phase == "await_result"
        await asyncio.sleep(0.12)
        t0 = store.trials_for_session(sess["session_id"])[0]
        assert t0["outcome"] == "no_result"
    asyncio.run(run())


def test_replay_and_simulated_ignored(tmp_path):
    async def run():
        runner, store, _, sess = await _started(tmp_path)
        await runner.on_wake(_wake())
        await runner.on_result(_result(source="replay"))
        await runner.on_result(_result(source="simulated"))
        assert runner.phase == "await_result"
        await runner.on_result(_result(source="live", class_key="PLAY_MUSIC"))
        assert runner.phase == "scored"
        t0 = store.trials_for_session(sess["session_id"])[0]
        assert t0["outcome"] == "correct"
    asyncio.run(run())


def test_wrong_and_negative_false_accept(tmp_path):
    async def run():
        runner, store, _, sess = await _started(tmp_path)
        await runner.on_wake(_wake())
        await runner.on_result(_result(source="live", class_key="LIGHT_OFF"))  # expected PLAY_MUSIC
        t0 = store.trials_for_session(sess["session_id"])[0]
        assert t0["outcome"] == "wrong"
    asyncio.run(run())


def test_retry_supersedes(tmp_path):
    async def run():
        runner, store, _, sess = await _started(tmp_path)
        await runner.on_wake(_wake())
        await runner.on_result(_result(source="live", class_key="PLAY_MUSIC"))
        await runner.retry()
        trials = [x for x in store.trials_for_session(sess["session_id"]) if x["idx"] == 0]
        assert len(trials) == 2
        assert trials[0]["superseded"] == 1
        assert trials[1]["attempt"] == 1
        assert runner.phase == "await_wake"
    asyncio.run(run())


def test_misspoken_excluded(tmp_path):
    async def run():
        runner, store, _, sess = await _started(tmp_path)
        await runner.on_wake(_wake())
        await runner.on_result(_result(source="live", class_key="PLAY_MUSIC"))
        await runner.misspoken()
        t0 = store.trials_for_session(sess["session_id"])[0]
        assert t0["misspoken"] == 1
        assert runner.idx == 1
    asyncio.run(run())


def test_false_wake_recorded(tmp_path):
    async def run():
        runner, store, _, sess = await _started(tmp_path)
        await runner.on_wake(_wake())       # real wake -> await_result
        await runner.on_wake(_wake())       # extra wake -> false wake
        assert store.false_wake_count(sess["session_id"]) == 1
    asyncio.run(run())


def test_lock_captured_at_start(tmp_path):
    async def run():
        runner, _, _, _ = await _started(tmp_path)
        assert runner.running
        assert runner.snapshot()["locked"] == {"model_id": "B2_s0", "threshold": 0.8}
    asyncio.run(run())


def test_second_start_rejected(tmp_path):
    async def run():
        runner, _, _, sess = await _started(tmp_path)
        with pytest.raises(RuntimeError):
            await runner.start(sess["session_id"])
    asyncio.run(run())


def test_trial_labels_command_log(tmp_path):
    async def run():
        runner, _, host, _ = await _started(tmp_path)
        await runner.on_wake(_wake())
        await runner.on_result(_result(source="live", class_key="PLAY_MUSIC"))
        assert host.labels["cid_t"] == "PLAY_MUSIC"
    asyncio.run(run())
