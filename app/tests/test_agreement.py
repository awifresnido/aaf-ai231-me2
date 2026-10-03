"""Double-Check agreement engine: accept only when every member agrees."""
import asyncio
from pathlib import Path

import numpy as np
import pytest
import yaml

from edge.engines.base import RawOutput, VCMEngine
from edge.media import MediaController, SimulatedBackend
from edge.pipeline import EdgePipeline, EdgeSettings
from edge.registry import ModelRegistry
from edge.wake import ManualWake
from vcm_common.manifest import ModelManifest
from vcm_common.ontology import get_ontology

REPO = Path(__file__).resolve().parents[1]
O = get_ontology()
LABELS = O.leaf_labels()


class StubMember(VCMEngine):
    """Returns a fixed class with a fixed confidence."""

    def __init__(self, mid: str, class_key: str, confidence: float):
        m = ModelManifest(id=mid, display_name=mid, task="vcm", engine="torch_tinyvcm",
                          labels=LABELS, reject={"min_confidence": 0.8})
        super().__init__(m)
        self._key = class_key
        self._conf = confidence

    def load(self) -> None:
        return None

    def infer(self, waveform, sample_rate):
        probs = np.zeros(len(LABELS), dtype=np.float32)
        probs[LABELS.index(self._key)] = self._conf
        return RawOutput(probs=probs, feature_ms=1.0, inference_ms=2.0)


class StubRegistry(ModelRegistry):
    def __init__(self, agree_members):
        super().__init__(REPO / "models")
        self._agree = agree_members

    def get(self, model_id):
        if model_id == "AGREE_B2_E1":
            raise RuntimeError("pipeline must call members directly")
        if model_id in self._agree:
            return self._agree[model_id]
        raise RuntimeError(f"unexpected member {model_id}")


def _agree_manifest(threshold=0.90):
    return ModelManifest(
        id="AGREE_B2_E1", display_name="Double-Check", task="vcm", engine="agreement",
        members=["B2_s0", "E1_s0"], labels=LABELS,
        reject={"min_confidence": threshold, "non_command_labels": ["UNKNOWN", "SILENCE"]},
    )


def make_pipeline(members, threshold=0.90, **kw):
    reg = StubRegistry(members)
    # inject agreement entry so the pipeline can read its manifest + member names
    ag = _agree_manifest(threshold)
    from edge.registry import Entry
    reg.entries["AGREE_B2_E1"] = Entry(manifest=ag)
    reg.entries["B2_s0"] = Entry(manifest=members["B2_s0"].manifest)
    reg.entries["E1_s0"] = Entry(manifest=members["E1_s0"].manifest)

    msgs: list[dict] = []

    async def emit(m):
        msgs.append(m)

    p = EdgePipeline(reg, {"manual": ManualWake()}, MediaController(backend=SimulatedBackend()),
                     EdgeSettings(active_vcm="AGREE_B2_E1", result_hold_s=0.0, **kw), emit)
    return p, msgs


def _run(p):
    async def go():
        return await p.run_command("replay", clip=np.zeros(16000, dtype=np.float32))
    return asyncio.run(go())


def test_agree_accepts(tmp_path):
    p, _ = make_pipeline({
        "B2_s0": StubMember("B2_s0", "LIGHT_ON", 0.97),
        "E1_s0": StubMember("E1_s0", "LIGHT_ON", 0.95),
    })
    r = _run(p)
    assert r.accepted
    assert r.primary.model_id == "AGREE_B2_E1"
    assert r.primary.class_key == "LIGHT_ON"
    assert len(r.members) == 2
    assert r.threshold == 0.90
    assert r.primary.confidence == pytest.approx(0.95)


def test_agree_one_below_threshold_rejected(tmp_path):
    p, _ = make_pipeline({
        "B2_s0": StubMember("B2_s0", "LIGHT_ON", 0.97),
        "E1_s0": StubMember("E1_s0", "LIGHT_ON", 0.84),
    })
    r = _run(p)
    assert not r.accepted
    assert "E1_s0" in r.reject_reason and "below threshold" in r.reject_reason


def test_disagree_rejected_with_both_predictions(tmp_path):
    p, _ = make_pipeline({
        "B2_s0": StubMember("B2_s0", "LIGHT_ON", 0.95),
        "E1_s0": StubMember("E1_s0", "LIGHT_OFF", 0.91),
    })
    r = _run(p)
    assert not r.accepted
    assert "disagree" in r.reject_reason
    assert "LIGHT_ON" in r.reject_reason and "LIGHT_OFF" in r.reject_reason


def test_both_unknown_rejected(tmp_path):
    p, _ = make_pipeline({
        "B2_s0": StubMember("B2_s0", "UNKNOWN", 0.99),
        "E1_s0": StubMember("E1_s0", "UNKNOWN", 0.99),
    })
    r = _run(p)
    assert not r.accepted
    assert "not a command" in r.reject_reason


def test_threshold_override_applies_to_both(tmp_path):
    p, _ = make_pipeline({
        "B2_s0": StubMember("B2_s0", "LIGHT_ON", 0.88),
        "E1_s0": StubMember("E1_s0", "LIGHT_ON", 0.86),
    }, threshold=0.90, threshold_override=0.80)
    r = _run(p)
    assert r.accepted and r.threshold == 0.80


def test_labels_mismatch_fails_loudly(tmp_path):
    from edge.registry import ModelRegistry
    reg = ModelRegistry(REPO / "models", {"tiny_vcm_root": str(REPO.parent / "tiny-vcm")})
    reg.scan()
    # corrupt E1's labels, then try to load the agreement entry
    reg.entries["E1_s0"].manifest.labels = ["X"]
    from edge.engines.base import EngineUnavailable
    with pytest.raises(EngineUnavailable, match="labels mismatch"):
        reg.get("AGREE_B2_E1")


def test_agreement_registered_manifest(tmp_path):
    m = yaml.safe_load((REPO / "models" / "vcm" / "AGREE_B2_E1" / "manifest.yaml").read_text())
    assert m["engine"] == "agreement"
    assert m["members"] == ["B2_s0", "E1_s0"]
    assert m["labels"] == O.leaf_labels()
    assert m["reject"]["min_confidence"] == 0.90
