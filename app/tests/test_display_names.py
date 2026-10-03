from pathlib import Path

import yaml
import pytest

from vcm_common.display_names import DisplayNames

REPO = Path(__file__).resolve().parents[1]


def _names() -> DisplayNames:
    return DisplayNames.load(REPO / "config" / "display_names.yaml")


def test_every_registered_model_has_a_name():
    dn = _names()
    registered = [p.parent.name for p in (REPO / "models" / "vcm").glob("*/manifest.yaml")]
    assert registered, "expected at least one registered model"
    assert dn.missing_model_names(registered) == []


def test_display_names_resolve():
    dn = _names()
    assert dn.model_name("B2_s0") == "Sweep (TC-ResNet)"
    assert dn.model_name("E1_s0") == "Focus (CRNN-Attn)"
    assert dn.model_name("AGREE_B2_E1") == "Double-Check (Agreement)"
    assert dn.dataset_name("composite_v1") == "Blend v1"
    assert dn.session_name("s02") == "Tuning Session"


def test_manifests_carry_friendly_display_name():
    dn = _names()
    for p in (REPO / "models" / "vcm").glob("*/manifest.yaml"):
        m = yaml.safe_load(p.read_text())
        assert m["display_name"] == dn.model_name(m["id"]), m["id"]


def test_missing_model_name_reported():
    dn = _names()
    assert dn.missing_model_names(["NOT_REGISTERED"]) == ["NOT_REGISTERED"]


def test_benchmark_role_is_optional_and_marked():
    dn = _names()
    assert dn.model_role("G2_s0") == "benchmark"      # Benchmark (DS-CNN)
    assert dn.model_role("B2_s0") == ""               # no role key
    assert dn.model_role("NOT_REGISTERED") == ""
    assert dn.model("G2_s0")["role"] == "benchmark"
    assert "role" not in dn.model("B2_s0")


def test_dataset_display_id_aliases_mark_optionb():
    dn = _names()
    assert dn.dataset_display_id("mark_optionb") == "synthetic_tts"
    assert dn.dataset_display_id("fsc") == "fsc"
    assert dn.dataset_display_id("NOT_A_DATASET") == "NOT_A_DATASET"
    public = dn.as_dict()["datasets"]
    assert "mark_optionb" not in public
    assert public["synthetic_tts"]["name"] == "Synthetic Voices"


def test_order_models_puts_benchmark_last():
    dn = _names()
    assert dn.order_models(["G2_s0", "B2_s0", "E1_s0"]) == ["B2_s0", "E1_s0", "G2_s0"]
    assert dn.order_models(["B2_s0"]) == ["B2_s0"]
