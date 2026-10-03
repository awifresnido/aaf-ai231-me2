"""The About data snapshot must be generated, sourced, and reproducible.

Checks:
- validates against a Pydantic schema
- every numeric value has a ``source`` entry
- no ``mark_optionb`` and no ``\\bMark\\b`` anywhere
- the model-choice values equal analysis.json (Final Test Session for B2/E1,
  the Sweep tuning sd)
- rebuilding reproduces identical numbers (skipped if tiny-vcm is absent)
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

REPO = Path(__file__).resolve().parents[1]
ABOUT = REPO / "laptop" / "app" / "about" / "about_data.json"

SKIP_KEYS = {"sources", "generated_at", "app_commit", "tvcm_commit", "host",
             "name", "id", "runtime", "quantization", "status", "repo_url",
             "note", "source"}


class Constraints(BaseModel):
    n_classes: int
    n_intents: int
    n_fixed: int
    n_slotted: int
    params_default_k: int
    onnx_kb_default: int
    window_s: float
    n_frames: int


class HeadlineRow(BaseModel):
    id: str
    name: str
    correct: float | None = None
    correct_sd: float | None = None
    wrong: float
    wrong_sd: float | None = None
    random_far: float | None = None
    random_far_sd: float | None = None


class TableRow(BaseModel):
    id: str
    name: str
    correct: float | None = None
    wrong: float
    far: float
    cmd_f1_val: float | None = None
    cmd_f1_test: float | None = None
    final_test: float | None = None


class Deployment(BaseModel):
    name: str
    onnx_kb: int
    runtime: str
    quantization: str


class AboutData(BaseModel):
    generated_at: str
    app_commit: str
    tvcm_commit: str
    host: dict
    constraints: Constraints
    intents: list[str]
    audit: dict
    story: dict
    composition: dict
    headline: dict[str, HeadlineRow]
    table: dict[str, TableRow]
    params: dict[str, int]
    b0: dict
    mcnemar: dict
    final_test: dict[str, float]
    tuning_sd: dict[str, float]
    agree: dict
    parity: dict[str, float]
    deployment: dict[str, Deployment]
    latency_pi: dict
    repo_url: str | None = None
    sources: dict[str, str]


def _load() -> dict:
    assert ABOUT.exists(), "about_data.json missing (run scripts/build_about_data.py)"
    return json.loads(ABOUT.read_text(encoding="utf-8"))


def _numeric_paths(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in SKIP_KEYS:
                continue
            out.update(_numeric_paths(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_numeric_paths(v, f"{prefix}.{i}"))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = obj
    return out


def test_snapshot_validates_against_schema():
    AboutData.model_validate(_load())


def test_every_numeric_value_has_a_source():
    data = _load()
    numbers = _numeric_paths(data)
    assert numbers, "expected numeric values"
    missing = [p for p in numbers if p not in data["sources"]]
    assert not missing, f"unsourced numeric values: {missing}"


def test_no_classmate_name_in_snapshot():
    import re
    text = ABOUT.read_text(encoding="utf-8")
    assert "mark_optionb" not in text
    assert not re.search(r"\bMark\b", text)


def test_model_choice_values_match_analysis_json(tiny_vcm_root):
    data = _load()
    analysis = json.loads((tiny_vcm_root / "results" / "dscnn" / "analysis.json").read_text())
    ts = analysis["test_split"]
    assert data["final_test"]["B2_s0"] == ts["B2_s0"]["slices"]["personal_awi"]["correct"]
    assert data["final_test"]["E1_s0"] == ts["E1_s0"]["slices"]["personal_awi"]["correct"]
    assert data["tuning_sd"]["B2"] == analysis["published_means"]["B2"]["tuning_sd"]


def test_rebuild_reproduces_identical_numbers(tiny_vcm_root, tmp_path):
    before = _load()
    out = tmp_path / "about_data.json"
    subprocess.run([sys.executable, "scripts/build_about_data.py", str(out)],
                   cwd=REPO, check=True, capture_output=True, text=True)
    after = json.loads(out.read_text(encoding="utf-8"))
    for k in ("generated_at", "app_commit", "tvcm_commit"):
        after.pop(k, None)
        before.pop(k, None)
    assert after == before
