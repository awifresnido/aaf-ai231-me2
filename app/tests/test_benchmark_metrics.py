"""Metrics: pooled sums, Wilson, exclusions, amber chip, wake-miss exclusion."""
from laptop.app.benchmark import metrics as M

TARGETS = {
    "correct_min": 0.85, "wrong_max": 0.01, "false_accept_max": 0.05,
    "wake_miss_max": 0.05, "pi_model_p95_ms_max": 50, "pi_features_p95_ms_max": 30,
    "params_max": 100000, "model_size_kb_max": 1024,
}
FACTS = {"params": 67793, "size_kb": 268, "runtime": "ONNX Runtime", "name": "Sweep (TC-ResNet)"}
FIXED = {0: "Play music", 1: "Weather"}


def _trial(**kw):
    base = {"session_id": "s1", "alias": "Guest 01", "set_type": "fixed", "model_id": "B2_s0",
            "cond_noise": "quiet", "cond_distance": "near", "created_ms": 1.0, "status": "finished",
            "kind": "command", "idx": 0, "outcome": "correct", "intent_correct": 1,
            "expected_class": "PLAY_MUSIC", "inference_ms": 34.0, "feature_ms": 12.0,
            "capture_to_result_ms": 40.0, "result_to_action_ms": 5.0, "misspoken": 0,
            "superseded": 0}
    base.update(kw)
    return base


def test_wilson_known_values():
    lo, hi = M.wilson(0, 0)
    assert (lo, hi) == (0.0, 0.0)
    lo, hi = M.wilson(8, 10)
    assert round(lo, 3) == 0.490 and round(hi, 3) == 0.943


def test_pooled_sums_counts():
    trials = [_trial(), _trial(idx=1, expected_class="WEATHER", outcome="wrong", intent_correct=0,
                               session_id="s2", alias="Guest 02", created_ms=2.0)]
    r = M.compute(trials, TARGETS, FACTS, 0, FIXED)
    # pooled over both sessions: 1 correct + 1 wrong, so correct rate = 0.5, not averaged 1.0
    assert r["pooled"]["correct"]["n"] == 2
    assert r["pooled"]["correct"]["k"] == 1
    assert abs(r["pooled"]["correct"]["value"] - 0.5) < 1e-9


def test_wake_miss_excluded_from_vcm_rates():
    trials = [_trial(outcome="correct"),
              _trial(idx=1, expected_class="WEATHER", outcome="wake_miss")]
    r = M.compute(trials, TARGETS, FACTS, 0, FIXED)
    assert r["pooled"]["correct"]["n"] == 1  # wake_miss not counted in VCM denominator
    assert r["pooled"]["wake_miss"]["n"] == 2  # wake_miss denominator includes wake_miss


def test_exclusions_applied():
    trials = [_trial(outcome="correct"),
              _trial(idx=1, outcome="wrong", superseded=1),
              _trial(idx=2, outcome="correct", misspoken=1),
              _trial(idx=3, outcome="correct", session_id="s2", status="abandoned")]
    r = M.compute(trials, TARGETS, FACTS, 0, FIXED)
    assert r["pooled"]["correct"]["n"] == 1  # only the first survives


def test_amber_chip_below_n20():
    trials = [_trial(outcome="correct")]
    r = M.compute(trials, TARGETS, FACTS, 0, FIXED)
    by_key = {c["key"]: c for c in r["chips"]}
    assert by_key["correct"]["status"] == "too_few"
    assert by_key["params"]["status"] == "pass"  # static facts have no 'too few'
