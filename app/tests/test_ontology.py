import json
import sys

import pytest
import yaml

from vcm_common.ontology import DEFAULT_ONTOLOGY_PATH, get_ontology


def test_leaf_labels_count_and_decode():
    o = get_ontology()
    leaves = o.leaf_labels()
    assert len(leaves) == 13 + 6 * 3 + 2 == 33
    assert o.decode("BRIGHTNESS|60 percent") == ("BRIGHTNESS", "60 percent")
    assert o.decode("PAUSE") == ("PAUSE", None)


@pytest.mark.parametrize("intent,slot,ok", [
    ("LIGHT_ON", None, True), ("TIMER", "30 seconds", True), ("TIMER", "5 minutes", False),
    ("TIMER", None, False), ("PAUSE", "x", False), ("SILENCE", None, False),
    ("UNKNOWN", None, False), ("DANCE", None, False),
])
def test_validate(intent, slot, ok):
    assert (get_ontology().validate(intent, slot) is None) == ok


def test_committed_manifests_match_ontology():
    """Hand-written manifests must list classes in tiny-vcm index order."""
    from conftest import REPO_ROOT

    o = get_ontology()
    for path in (REPO_ROOT / "models" / "vcm").glob("*/manifest.yaml"):
        m = yaml.safe_load(path.read_text())
        expected = o.leaf_labels() if m["label_scheme"] == "leaf" else o.intent_labels()
        assert m["labels"] == expected, path


def test_vendored_ontology_matches_training(tiny_vcm_root):
    ours = json.loads(DEFAULT_ONTOLOGY_PATH.read_text(encoding="utf-8"))
    theirs = json.loads((tiny_vcm_root / "configs" / "ontology.json").read_text(encoding="utf-8"))
    assert ours == theirs, "run scripts/sync_ontology.py"


@pytest.mark.parametrize("mode", ["leaf", "intent"])
def test_label_order_matches_tiny_vcm(tiny_vcm_root, mode):
    pytest.importorskip("torch")
    sys.path.insert(0, str(tiny_vcm_root))
    from src.vcm_data_loader import create_label_mapping

    mapping = create_label_mapping(str(tiny_vcm_root / "configs" / "ontology.json"), mode)
    theirs = [k for k, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
    o = get_ontology()
    assert (o.leaf_labels() if mode == "leaf" else o.intent_labels()) == theirs
