#!/usr/bin/env python3
"""Generate ``laptop/app/about/about_data.json`` — the single source of every
number the About view shows.

No metric is typed into React; every value here is read from a tracked file and
carries a ``source`` string. Values that only exist in markdown reports come from
``scripts/about_sources.yaml`` (verified verbatim). Run:

    .venv/bin/python scripts/build_about_data.py

It fails loudly (nonzero exit) if a source file is missing or a markdown quote
does not appear verbatim, so a hand-copied value can never drift silently.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

APP = Path(__file__).resolve().parents[1]          # repo root
TVCM = APP.parent / "tiny-vcm"                       # read-only sibling
ABOUT_DIR = APP / "laptop" / "app" / "about"
OUT = ABOUT_DIR / "about_data.json"
COMPOSITION = ABOUT_DIR / "composition.json"
SOURCES_YAML = APP / "scripts" / "about_sources.yaml"

MODELS = ["B2_s0", "E1_s0", "G2_s0"]                # the three shippable models


def _git(cwd: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _pretty(intent: str) -> str:
    words = intent.lower().split("_")
    return " ".join(w if i else w.capitalize() for i, w in enumerate(words))


def main() -> None:
    # ---- inputs -------------------------------------------------------------
    ontology = json.loads((APP / "vcm_common" / "ontology.json").read_text())
    analysis = json.loads((TVCM / "results" / "dscnn" / "analysis.json").read_text())
    composition = json.loads(COMPOSITION.read_text())
    markdown = yaml.safe_load(SOURCES_YAML.read_text()) or {}
    for key, entry in markdown.items():
        f = (APP / entry["file"]).resolve()
        text = f.read_text(encoding="utf-8")
        assert entry["quote"] in text, f"{key}: quote not found verbatim in {entry['file']}"

    def manifest(mid: str) -> dict:
        return yaml.safe_load((APP / "models" / "vcm" / mid / "manifest.yaml").read_text())

    def deploy(mid: str) -> dict:
        return json.loads((TVCM / "exports" / mid / "deploy.json").read_text())

    def onnx_kb(mid: str) -> int:
        return round((TVCM / "exports" / mid / "model.onnx").stat().st_size / 1024)

    pm = analysis["published_means"]                 # B2, E1, AGREE (5-seed, CRNN-Attn)
    g2 = analysis["models"]["G2"]["agg"]
    gsc = analysis["gsc_far"]
    ts = analysis["test_split"]
    b0 = analysis["models"]["A7"]["seeds"][0]
    mcn = analysis["mcnemar"]
    g2_s0 = analysis["models"]["G2"]["seeds"][0]
    # analysis.json keys models by base id ("B2"), not run id ("B2_s0").
    ANALYSIS_KEY = {"B2_s0": "B2", "E1_s0": "E1", "G2_s0": "G2", "AGREE_B2_E1": "AGREE"}

    src: dict[str, str] = {}
    def S(path: str, source: str) -> str:
        src[path] = source
        return source

    # ---- constraints --------------------------------------------------------
    labels = ontology["labels"]
    fixed = labels["fixed"]
    slotted = labels["slotted"]
    n_fixed, n_slotted = len(fixed), len(slotted)
    n_intents = n_fixed + n_slotted
    n_classes = n_intents + len(labels["local_only"]) + sum(
        len(ontology["slot_values"][s]) - 1 for s in slotted)
    S("constraints.n_fixed", "vcm_common/ontology.json (labels.fixed)")
    S("constraints.n_slotted", "vcm_common/ontology.json (labels.slotted)")
    S("constraints.n_intents", "vcm_common/ontology.json (fixed + slotted)")
    S("constraints.n_classes", "vcm_common/ontology.json (leaf labels incl. UNKNOWN/SILENCE)")
    S("constraints.window_s", f"models/vcm/B2_s0/manifest.yaml (features.max_duration_s: {manifest('B2_s0')['features']['max_duration_s']})")
    S("constraints.n_frames", "tiny-vcm/exports/B2_s0/deploy.json (n_frames)")
    S("constraints.params_default_k", f"tiny-vcm/exports/B2_s0/deploy.json (params: {deploy('B2_s0')['params']})")
    S("constraints.onnx_kb_default", f"tiny-vcm/exports/B2_s0/model.onnx ({onnx_kb('B2_s0')} KB)")

    constraints = {
        "n_classes": n_classes,
        "n_intents": n_intents,
        "n_fixed": n_fixed,
        "n_slotted": n_slotted,
        "params_default_k": round(deploy("B2_s0")["params"] / 1000),
        "onnx_kb_default": onnx_kb("B2_s0"),
        "window_s": float(manifest("B2_s0")["features"]["max_duration_s"]),
        "n_frames": deploy("B2_s0")["n_frames"],
    }

    intents = [_pretty(i) for i in [*fixed, *slotted]]

    # ---- audit / story / mcnemar (markdown-sourced) -------------------------
    def md(key: str) -> object:
        return markdown[key]["value"]

    S("audit.relabeled", "about_sources.yaml -> tiny-vcm/PHASE2_RESULTS.md")
    S("audit.dropped", "about_sources.yaml -> tiny-vcm/PHASE2_RESULTS.md")
    S("story.first_acc", "about_sources.yaml -> VCM_TRAINING_TROUBLESHOOTING.md")
    S("story.fixed_acc", "about_sources.yaml -> tiny-vcm/ABLATION_RESULTS.md")
    S("mcnemar.e1_b2_p", "about_sources.yaml -> tiny-vcm/CRNN_ATTN_RESULTS.md")
    S("mcnemar.n", "tiny-vcm/results/dscnn/analysis.json (mcnemar.n_paired)")

    audit = {"relabeled": md("audit.relabeled"), "dropped": md("audit.dropped")}
    story = {"first_acc": md("story.first_acc"), "fixed_acc": md("story.fixed_acc")}
    mcnemar = {"n": mcn["G2_s0 vs B2_s0"]["n_paired"], "e1_b2_p": md("mcnemar.e1_b2_p")}

    # ---- composition --------------------------------------------------------
    for k, d in composition["datasets"].items():
        for split in ("train", "validation", "test"):
            S(f"composition.datasets.{k}.{split}",
              f"{composition['source']} counted on {composition['host']}")
    S("composition.total_clips", f"{composition['source']} counted on {composition['host']}")
    comp = {"total_clips": composition["total_clips"],
            "datasets": {k: {"name": d["name"], "train": d["train"],
                             "validation": d["validation"], "test": d["test"]}
                         for k, d in composition["datasets"].items()}}

    # ---- headline chart (P3a) + table (P3) ----------------------------------
    def _headline(mid: str) -> dict:
        # Sweep/Focus: correct is single-seed (s0); wrong/FAR are the 5-seed
        # published means. Benchmark (G2) is this benchmark's own 5-seed agg.
        if mid == "AGREE_B2_E1":
            return {"id": mid, "name": "Double-Check", "correct": None, "correct_sd": None,
                    "wrong": pm["AGREE"]["wrong"], "wrong_sd": None,
                    "random_far": None, "random_far_sd": None}
        if mid == "G2_s0":
            return {"id": mid, "name": "Benchmark",
                    "correct": g2["correct"]["mean"], "correct_sd": g2["correct"]["sd"],
                    "wrong": g2["wrong"]["mean"], "wrong_sd": g2["wrong"]["sd"],
                    "random_far": gsc["G2"]["mean"], "random_far_sd": gsc["G2"]["sd"]}
        seed0 = analysis["models"][ANALYSIS_KEY[mid]]["seeds"][0]
        name = "Sweep" if mid == "B2_s0" else "Focus"
        return {"id": mid, "name": name, "correct": seed0["correct"], "correct_sd": None,
                "wrong": pm[ANALYSIS_KEY[mid]]["wrong"], "wrong_sd": pm[ANALYSIS_KEY[mid]]["wrong_sd"],
                "random_far": gsc[ANALYSIS_KEY[mid]]["mean"], "random_far_sd": gsc[ANALYSIS_KEY[mid]]["sd"]}

    def _table_row(mid: str) -> dict:
        if mid == "AGREE_B2_E1":
            return {"id": mid, "name": "Double-Check", "correct": None,
                    "wrong": pm["AGREE"]["wrong"], "far": pm["AGREE"]["FAR"],
                    "cmd_f1_val": None, "cmd_f1_test": None, "final_test": None}
        if mid == "G2_s0":
            m = manifest(mid)
            return {"id": mid, "name": "Benchmark", "correct": g2["correct"]["mean"],
                    "wrong": g2["wrong"]["mean"], "far": g2["FAR"]["mean"],
                    "cmd_f1_val": g2["cmd_f1_tau0"]["mean"],
                    "cmd_f1_test": m["scores"]["test"]["metrics"].get("command_macro_f1"),
                    "final_test": ts[mid]["slices"]["personal_awi"]["correct"]}
        seed0 = analysis["models"][ANALYSIS_KEY[mid]]["seeds"][0]
        m = manifest(mid)
        name = "Sweep" if mid == "B2_s0" else "Focus" if mid == "E1_s0" else "Benchmark"
        cmd_f1_test = m["scores"]["test"]["metrics"].get("command_macro_f1")
        return {"id": mid, "name": name, "correct": seed0["correct"],
                "wrong": pm[ANALYSIS_KEY[mid]]["wrong"], "far": pm[ANALYSIS_KEY[mid]]["FAR"],
                "cmd_f1_val": pm[ANALYSIS_KEY[mid]]["cmd_f1_tau0"], "cmd_f1_test": cmd_f1_test,
                "final_test": ts[mid]["slices"]["personal_awi"]["correct"]}

    for mid in ["B2_s0", "E1_s0", "AGREE_B2_E1", "G2_s0"]:
        for field in ("correct", "correct_sd", "wrong", "wrong_sd", "random_far", "random_far_sd"):
            S(f"headline.{mid}.{field}", "tiny-vcm/results/dscnn/analysis.json (models/published_means/gsc_far)")
        for field in ("correct", "wrong", "far", "cmd_f1_val", "cmd_f1_test", "final_test"):
            S(f"table.{mid}.{field}", "tiny-vcm/results/dscnn/analysis.json + models/vcm manifest scores")

    headline = {m: _headline(m) for m in ["B2_s0", "E1_s0", "AGREE_B2_E1", "G2_s0"]}
    table = {m: _table_row(m) for m in ["B2_s0", "E1_s0", "AGREE_B2_E1", "G2_s0"]}

    # ---- params / b0 / final_test / tuning_sd / agree / parity ---------------
    params = {m: deploy(m)["params"] for m in MODELS}
    for m in MODELS:
        S(f"params.{m}", f"tiny-vcm/exports/{m}/deploy.json (params)")

    S("b0.correct", "tiny-vcm/results/dscnn/analysis.json (A7 seed0 @ tau 0.90)")
    S("b0.wrong", "tiny-vcm/results/dscnn/analysis.json (A7 seed0 @ tau 0.90)")
    S("final_test.B2_s0", "tiny-vcm/results/dscnn/analysis.json (test_split B2_s0 personal_awi)")
    S("final_test.E1_s0", "tiny-vcm/results/dscnn/analysis.json (test_split E1_s0 personal_awi)")
    S("final_test.G2_s0", "tiny-vcm/results/dscnn/analysis.json (test_split G2_s0 personal_awi)")
    S("tuning_sd.B2", "tiny-vcm/results/dscnn/analysis.json (published_means.B2.tuning_sd)")
    S("agree.wrong", "tiny-vcm/results/dscnn/analysis.json (published_means.AGREE.wrong)")
    S("agree.far", "tiny-vcm/results/dscnn/analysis.json (published_means.AGREE.FAR)")

    b0_row = {"correct": b0["correct"], "wrong": b0["wrong"]}
    final_test = {m: ts[m]["slices"]["personal_awi"]["correct"] for m in MODELS}
    tuning_sd = {"B2": pm["B2"]["tuning_sd"]}
    agree = {"wrong": pm["AGREE"]["wrong"], "far": pm["AGREE"]["FAR"]}

    # ---- deployment facts ----------------------------------------------------
    deployment = {}
    for m in MODELS:
        name = "Sweep" if m == "B2_s0" else "Focus" if m == "E1_s0" else "Benchmark"
        deployment[m] = {"name": name, "onnx_kb": onnx_kb(m),
                         "runtime": "ONNX Runtime", "quantization": manifest(m).get("quantization", "fp32")}
        S(f"deployment.{m}.onnx_kb", f"tiny-vcm/exports/{m}/model.onnx ({onnx_kb(m)} KB)")
    S("parity.B2_s0", "tiny-vcm/exports/B2_s0/deploy.json (fp32_parity_max_dprob)")
    parity = {"B2_s0": deploy("B2_s0")["fp32_parity_max_dprob"]}

    snapshot = {
        "generated_at": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                       capture_output=True, text=True, check=True).stdout.strip(),
        "app_commit": _git(APP),
        "tvcm_commit": _git(TVCM),
        "host": {"composition": composition["host"]},
        "constraints": constraints,
        "intents": intents,
        "audit": audit,
        "story": story,
        "composition": comp,
        "headline": headline,
        "table": table,
        "params": params,
        "b0": b0_row,
        "mcnemar": mcnemar,
        "final_test": final_test,
        "tuning_sd": tuning_sd,
        "agree": agree,
        "parity": parity,
        "deployment": deployment,
        "latency_pi": {"status": "pending"},
        "repo_url": None,
        "sources": src,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    out_path.write_text(json.dumps(snapshot, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
