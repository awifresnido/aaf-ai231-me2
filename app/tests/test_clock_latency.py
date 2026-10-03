"""Same-clock latency definitions: scripted round trip with a 13 h edge skew."""
import pytest

from laptop.app.benchmark import metrics
from laptop.app.benchmark.metrics import _e2e_ms
from laptop.app.command_log import CommandLog
from laptop.app.dispatcher import Outcome
from vcm_common.protocol import InferenceResult, ModelPrediction

SKEW = 13 * 3600 * 1000.0
TARGETS = {"correct_min": 0.85, "wrong_max": 0.01, "false_accept_max": 0.05,
           "wake_miss_max": 0.05, "pi_model_p95_ms_max": 50, "pi_features_p95_ms_max": 30,
           "params_max": 100000, "model_size_kb_max": 1024}


def _result(edge_times: bool = True):
    t_received = 1_700_000_000_000.0
    t_result_laptop = t_received - 5.0
    t_capture_laptop = t_result_laptop - 40.0
    kwargs = dict(
        command_id="c1", source="live",
        primary=ModelPrediction(model_id="B2_s0", class_key="LIGHT_ON", intent="LIGHT_ON",
                                confidence=0.9, inference_ms=30.0, feature_ms=12.0),
        accepted=True, threshold=0.8, clip_ms=500.0,
        t_capture_end_ms=t_capture_laptop, t_result_ms=t_result_laptop,
        t_received_ms=t_received, timing_valid=True)
    if edge_times:
        kwargs["edge_times"] = {"t_capture_end_ms": t_capture_laptop - SKEW,
                                "t_result_ms": t_result_laptop - SKEW}
        kwargs["edge_clock_offset_ms"] = -SKEW
    return InferenceResult(**kwargs)


def test_result_to_action_small_positive_with_skew(tmp_path):
    r = _result()
    action_ms = r.t_received_ms + 3.0
    log = CommandLog(tmp_path / "commands.sqlite")
    row = log.record(r, Outcome(True, "light on"), action_ms)
    assert row["result_to_action_ms"] == pytest.approx(3.0, abs=0.01)
    assert row["network_ms"] == pytest.approx(5.0, abs=0.01)
    assert row["capture_to_result_ms"] == pytest.approx(40.0, abs=0.01)
    assert row["timing_valid"] == 1
    assert row["edge_clock_offset_ms"] == pytest.approx(-SKEW)


def test_e2e_is_sum_of_three_parts():
    r = _e2e_ms([{"capture_to_result_ms": 40.0, "network_ms": 5.0,
                  "result_to_action_ms": 3.0, "timing_valid": 1}])
    assert r["n"] == 1
    assert r["p50"] == pytest.approx(48.0)


def test_old_rows_excluded_from_latency_kept_in_accuracy():
    trials = [
        {"session_id": "s1", "alias": "G1", "set_type": "fixed", "model_id": "B2_s0",
         "cond_noise": "quiet", "cond_distance": "near", "created_ms": 1.0, "status": "finished",
         "kind": "command", "idx": 0, "outcome": "correct", "intent_correct": 1,
         "expected_class": "PLAY_MUSIC", "inference_ms": 30.0, "feature_ms": 12.0,
         "capture_to_result_ms": 40.0, "network_ms": 5.0, "result_to_action_ms": 3.0,
         "timing_valid": 1, "misspoken": 0, "superseded": 0},
        {"session_id": "s1", "alias": "G1", "set_type": "fixed", "model_id": "B2_s0",
         "cond_noise": "quiet", "cond_distance": "near", "created_ms": 1.0, "status": "finished",
         "kind": "command", "idx": 1, "outcome": "correct", "intent_correct": 1,
         "expected_class": "WEATHER", "inference_ms": 30.0, "feature_ms": 12.0,
         "capture_to_result_ms": 40.0, "network_ms": None, "result_to_action_ms": 47_000_000.0,
         "timing_valid": 0, "misspoken": 0, "superseded": 0},
    ]
    r = metrics.compute(trials, TARGETS, {"params": 67793, "size_kb": 268.3, "runtime": "ONNX Runtime"},
                        0, {0: "Play music", 1: "Weather"})
    # accuracy keeps both rows
    assert r["pooled"]["correct"]["n"] == 2
    # latency keeps only the timing_valid row
    assert r["pooled"]["e2e_ms"]["n"] == 1
    assert r["pooled"]["e2e_ms"]["p50"] == pytest.approx(48.0)
    # single-clock metrics unaffected by the flag
    assert r["pooled"]["pi_model_ms"]["n"] == 2


def test_e2e_counts_only_trusted_clock_quality():
    base = {"capture_to_result_ms": 40.0, "network_ms": 5.0, "result_to_action_ms": 3.0,
            "timing_valid": 1}
    rows = [dict(base, clock_quality=q) for q in
            ("accurate", "local", None, "approximate", "unknown", "syncing")]
    r = _e2e_ms(rows)
    assert r["n"] == 3          # accurate, local, and pre-quality rows (None)


def test_command_log_network_needs_trusted_clock(tmp_path):
    log = CommandLog(tmp_path / "commands.sqlite")
    r = _result().model_copy(update={"clock_quality": "approximate"})
    row = log.record(r, Outcome(True, "light on"), r.t_received_ms + 3.0)
    assert row["network_ms"] is None and row["clock_quality"] == "approximate"
    r2 = _result().model_copy(update={"command_id": "c2", "clock_quality": "accurate"})
    row2 = log.record(r2, Outcome(True, "light on"), r2.t_received_ms + 3.0)
    assert row2["network_ms"] == pytest.approx(5.0, abs=0.01)


def test_command_log_migration_marks_old_rows_unknown(tmp_path):
    import sqlite3

    db = tmp_path / "commands.sqlite"
    con = sqlite3.connect(str(db))
    con.executescript(
        "CREATE TABLE commands (command_id TEXT PRIMARY KEY, ts_ms REAL, source TEXT, "
        "model_id TEXT, class_key TEXT, confidence REAL, accepted INTEGER, reject_reason TEXT, "
        "threshold REAL, feature_ms REAL, inference_ms REAL, clip_ms REAL, clip_rms_dbfs REAL, "
        "capture_to_result_ms REAL, result_to_action_ms REAL, action TEXT, executed INTEGER, "
        "expected_class TEXT, t_received_ms REAL, network_ms REAL, edge_clock_offset_ms REAL, "
        "timing_valid INTEGER);"
        "INSERT INTO commands (command_id, ts_ms, timing_valid) VALUES ('old', 1.0, 1);")
    con.commit()
    con.close()
    log = CommandLog(db)
    rows = {r["command_id"]: r for r in log.recent(5)}
    assert rows["old"]["clock_quality"] == "unknown"
