#!/usr/bin/env python3
"""Part 2 crop check: energy vs start crop on 4 slices, for B2_s0 + E1_s0.

Reads the extracted master slices (data/external/me2_crop_check/master_slices.csv),
Awi s02 (data/personal/raw/202453069/manifest.csv) and the v1 Single Words test
(composite_v1.csv gsc_v2 test). Evaluates each model with crop_mode in
{start, energy} and writes results/me2/crop_check.json with the pre-registered
GATE 2 decision.

Convention (documented in the JSON):
  * "correct" = raw accuracy (pred == true) at tau = 0, over command clips.
  * "FAR"     = false-accept rate at tau = 0.9 (v1 operating point), over
                UNKNOWN/SILENCE clips.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
torch.set_num_threads(1)  # per-clip mel is tiny; 64-thread parallelism thrashes
from src.vcm_data_loader import row_label  # noqa: E402
from src.vcm_infer import VCMInferencer  # noqa: E402

TAU_CORRECT = 0.0   # raw argmax accuracy
TAU_FAR = 0.9       # v1 operating threshold


def is_command(key: str) -> bool:
    return key not in ("UNKNOWN", "SILENCE")


def outcome(true_key, pred_key, prob, tau):
    if is_command(true_key):
        if pred_key == true_key and prob >= tau:
            return "correct"
        if (not is_command(pred_key)) or prob < tau:
            return "rejected"
        return "wrong"
    if is_command(pred_key) and prob >= tau:
        return "false_accept"
    return "correct_reject"


def leaf_of(command: str, slot: str) -> str:
    return row_label({"training_label": command, "slot_value": slot or ""}, "leaf")


def build_slices(master_csv: str, awi_manifest: str, awi_mapping: str, composite_csv: str) -> dict:
    slices: dict[str, list[tuple[str, str]]] = defaultdict(list)

    with open(master_csv, newline="") as f:
        for r in csv.DictReader(f):
            slices[r["slice"]].append((r["relative_path"], leaf_of(r["command"], r["slot_value"])))

    # Awi s02 = the validation session only (take 3, source_split == "validation").
    s02_files = set()
    with open(awi_mapping, newline="") as f:
        for r in csv.DictReader(f):
            if r["source_split"] == "validation":
                s02_files.add(r["new_filename"])
    awi_root = Path("data/personal/raw/202453069")
    with open(awi_manifest, newline="") as f:
        for r in csv.DictReader(f):
            if r["filename"] in s02_files:
                p = awi_root / r["filename"]
                slices["awi_s02"].append((str(p), leaf_of(r["label"], r["slot_value"])))

    with open(composite_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["corpus_id"] == "gsc_v2" and r["source_split"] == "test":
                slices["single_words"].append((r["relative_path"], row_label(r, "leaf")))

    return dict(slices)


def run_slice(inf: VCMInferencer, items: list[tuple[str, str]]) -> dict:
    n_cmd = n_unk = 0
    correct0 = correct9 = 0
    false_accept9 = 0
    for path, true_key in items:
        pred_key, prob = inf.predict_wav(path)
        if is_command(true_key):
            n_cmd += 1
            if pred_key == true_key:
                correct0 += 1
                if prob >= TAU_FAR:
                    correct9 += 1
        else:
            n_unk += 1
            if is_command(pred_key) and prob >= TAU_FAR:
                false_accept9 += 1
    return {
        "n_command": n_cmd,
        "n_unknown_silence": n_unk,
        "correct_tau0": correct0 / n_cmd if n_cmd else None,
        "correct_tau0.9": correct9 / n_cmd if n_cmd else None,
        "FAR_tau0.9": false_accept9 / n_unk if n_unk else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-csv", default="data/external/me2_crop_check/master_slices.csv")
    ap.add_argument("--awi-manifest", default="data/personal/raw/202453069/manifest.csv")
    ap.add_argument("--awi-mapping", default="data/personal/raw/202453069/awi01_mapping.csv")
    ap.add_argument("--composite-csv", default="data/manifests/composite_v1.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/me2/crop_check.json")
    args = ap.parse_args()

    models = [
        {"name": "B2_s0",
         "checkpoint": "checkpoints/B2_s0/best.pt",
         "arch": "tcresnet", "width_mult": 1.0},
        {"name": "E1_s0",
         "checkpoint": "checkpoints/E1_s0/best.pt",
         "arch": "crnn_attn", "width_mult": 1.0},
    ]

    slices = build_slices(args.master_csv, args.awi_manifest, args.awi_mapping, args.composite_csv)
    print("slices:", {k: len(v) for k, v in slices.items()})

    result = {
        "tau_correct": TAU_CORRECT,
        "tau_far": TAU_FAR,
        "slices_n": {k: len(v) for k, v in slices.items()},
        "models": {},
        "gate2_decision": {},
    }

    per_model = {}
    for m in models:
        per_model[m["name"]] = {}
        print(f"\n=== {m['name']} ({m['arch']}) ===")
        for cm in ("start", "energy"):
            inf = VCMInferencer(
                m["checkpoint"], device=args.device, arch=m["arch"],
                width_mult=m["width_mult"], label_mode="leaf",
                n_mels=40, max_duration=3.0, crop_mode=cm,
            )
            per_model[m["name"]][cm] = {}
            for sname, items in slices.items():
                st = run_slice(inf, items)
                per_model[m["name"]][cm][sname] = st
                print(f"  {cm:7s} {sname:14s} cmd={st['n_command']:5d} "
                      f"correct0={_f(st['correct_tau0'])} FAR0.9={_f(st['FAR_tau0.9'])}")
            del inf

    # GATE 2 decision
    for m in models:
        nm = m["name"]
        s = per_model[nm]["start"]
        e = per_model[nm]["energy"]
        d_correct = _pt(e["201435283"]["correct_tau0"], s["201435283"]["correct_tau0"])
        d_awi = _pt(e["awi_s02"]["correct_tau0"], s["awi_s02"]["correct_tau0"])
        d_far = _pt(e["single_words"]["FAR_tau0.9"], s["single_words"]["FAR_tau0.9"])
        adopt = (d_correct >= 5.0) and (d_awi >= -1.0) and (d_far <= 1.0)
        result["gate2_decision"][nm] = {
            "delta_201435283_correct_pt": round(d_correct, 3),
            "delta_awi_s02_correct_pt": round(d_awi, 3),
            "delta_single_words_FAR_pt": round(d_far, 3),
            "adopt_energy": adopt,
        }
        print(f"\n{nm}: d_correct(201435283)={d_correct:+.2f}pt "
              f"d_awi_s02={d_awi:+.2f}pt d_FAR={d_far:+.2f}pt -> adopt={adopt}")

    result["models"] = per_model
    both_adopt = all(result["gate2_decision"][m["name"]]["adopt_energy"] for m in models)
    result["gate2_decision"]["overall_crop_mode"] = "energy" if both_adopt else "start"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")
    print(f"GATE 2 overall: adopt 'energy' = {both_adopt} -> use '{result['gate2_decision']['overall_crop_mode']}'")
    return 0


def _f(x):
    return f"{x:.4f}" if x is not None else "-"


def _pt(new, old):
    if new is None or old is None:
        return 0.0
    return (new - old) * 100.0


if __name__ == "__main__":
    sys.exit(main())
