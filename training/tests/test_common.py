from pathlib import Path

from vcm_data.common import (
    MANIFEST_COLUMNS,
    empty_record,
    load_ontology,
    load_slot_values,
    normalize_text,
    stable_id,
    write_manifest,
)

ONTOLOGY = Path("configs/ontology.json")


def test_stable_id_is_deterministic_and_prefixed() -> None:
    assert stable_id("demo", "a", 1) == stable_id("demo", "a", 1)
    assert stable_id("demo", "a", 1).startswith("demo_")
    assert stable_id("demo", "a", 1) != stable_id("demo", "a", 2)


def test_normalize_text_is_conservative() -> None:
    assert normalize_text("  Turn  ON the Lights ") == "turn on the lights"


def test_ontology_is_optionb_with_21_classes() -> None:
    labels, leaf_map = load_ontology(ONTOLOGY)
    assert len(labels) == 21
    assert {"PLAY_MUSIC", "STOP", "COLOR", "UNKNOWN", "SILENCE"} <= labels
    # leaf and training label are deliberately 1:1
    assert all(leaf_map[label] == label for label in labels)


def test_slot_values_are_closed_and_complete() -> None:
    slot_values = load_slot_values(ONTOLOGY)
    assert set(slot_values) == {
        "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER"
    }
    assert all(len(values) == 3 for values in slot_values.values())
    assert slot_values["COLOR"] == ["Red", "Blue", "Green"]


def test_manifest_columns_include_optionb_slot_fields() -> None:
    assert "slot_type" in MANIFEST_COLUMNS
    assert "slot_value" in MANIFEST_COLUMNS


def test_manifest_write_is_atomic(tmp_path: Path) -> None:
    target = tmp_path / "manifest.csv"
    row = empty_record()
    row["sample_id"] = "demo_1"
    assert write_manifest(target, [row]) == 1
    assert target.exists()
    assert not target.with_suffix(".csv.tmp").exists()
