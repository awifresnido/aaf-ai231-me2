"""Benchmark config: fixed-set resolution, slot validation, random reproducibility."""
from pathlib import Path

import pytest

from laptop.app.benchmark.config import (
    load_benchmark_config,
    generate_random_set,
    normalize_slot,
)
from laptop.app.main import REPO_ROOT
from vcm_common.ontology import get_ontology


@pytest.fixture
def cfg():
    return load_benchmark_config(REPO_ROOT / "config" / "benchmark.yaml")


def test_normalize_slot_clock_and_case():
    assert normalize_slot("6 AM") == normalize_slot("6:00 AM")
    assert normalize_slot("drink water") == normalize_slot("Drink water")
    assert normalize_slot("60 percent") == "60 percent"


def test_fixed_set_resolves_to_leaf_classes(cfg):
    assert len(cfg.fixed_set) == 12
    commands = [p for p in cfg.fixed_set if p.kind == "command"]
    negatives = [p for p in cfg.fixed_set if p.kind == "negative"]
    assert len(commands) == 10
    assert len(negatives) == 2
    leaves = get_ontology().leaf_labels()
    for p in commands:
        assert p.expected_class in leaves, p.expected_class
    # the tricky slots resolved exactly
    by_say = {p.say: p for p in commands}
    assert by_say["Set an alarm for 6 AM"].expected_class == "ALARM|6:00 AM"
    assert by_say["Remind me to drink water"].expected_class == "CREATE_REMINDER|Drink water"


def test_random_set_reproducible(cfg):
    o = get_ontology()
    a = generate_random_set(o, cfg.phrases, 10, 2, cfg.negatives, 42)
    b = generate_random_set(o, cfg.phrases, 10, 2, cfg.negatives, 42)
    assert len(a) == 12
    assert [p.expected_class for p in a] == [p.expected_class for p in b]
    # 10 distinct commands + 2 negatives
    cmds = [p for p in a if p.kind == "command"]
    assert len({p.expected_class for p in cmds}) == 10


def test_bad_slot_fails(monkeypatch):
    import yaml
    from laptop.app.benchmark.config import load_benchmark_config as _load
    bad = Path("/tmp/bench_bad_slot.yaml")
    good = yaml.safe_load((REPO_ROOT / "config" / "benchmark.yaml").read_text())
    good["fixed_set"] = [{"say": "Set alarm for noon", "expect": "ALARM", "slot": "noon"}]
    bad.write_text(yaml.safe_dump(good), encoding="utf-8")
    with pytest.raises(ValueError):
        _load(bad)
