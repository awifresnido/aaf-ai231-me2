import yaml
import pytest
from pydantic import ValidationError

from vcm_common.manifest import ModelManifest, ScoreEntry, Scores

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_all_registered_manifests_load_with_scores():
    for p in sorted((REPO / "models" / "vcm").glob("*/manifest.yaml")):
        m = ModelManifest.load(p.parent)
        assert isinstance(m.scores, Scores)


def test_reported_with_empty_metrics_fails():
    with pytest.raises(ValidationError):
        ScoreEntry(status="reported", metrics={})


def test_not_evaluated_with_empty_metrics_ok():
    e = ScoreEntry(status="not_evaluated", metrics={})
    assert e.status == "not_evaluated"


def test_reported_scores_present_for_leaf_models():
    for mid in ("B2_s0", "E1_s0"):
        m = yaml.safe_load((REPO / "models" / "vcm" / mid / "manifest.yaml").read_text())
        s = m["scores"]
        assert s["training"]["status"] == "reported"
        assert s["validation"]["status"] == "reported"
        assert s["test"]["status"] == "reported"


def test_double_check_scores_not_applicable_training():
    m = yaml.safe_load((REPO / "models" / "vcm" / "AGREE_B2_E1" / "manifest.yaml").read_text())
    assert m["scores"]["training"]["status"] == "not_applicable"
    assert m["scores"]["test"]["status"] == "not_evaluated"
    assert m["scores"]["validation"]["metrics"]["tuning_session_correct"] == 0.918
