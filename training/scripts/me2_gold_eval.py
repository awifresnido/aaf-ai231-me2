#!/usr/bin/env python3
"""Part C (step-0) + Part E: evaluate checkpoints on the gold slices (Section 8).

Two-phase Part E:
  --infer   : evaluate the named models, save raw predictions per model to
              results/me2_gold/preds_<model>.json  (parallelisable across GPUs)
  --decide  : read all preds_*.json, compute per-arch tau on the tau-set, apply the
              pre-registered decision rule, write eval_*.json + decision.json

Step-0 mode (default): fixed tau 0.9 over the 9 v1 init checkpoints, step0_*.json.

Slices: validation_real (gold test speaker 201435283), validation_synthetic,
validation_paraphrase, validation_oos, awi_s02, holdout_real (202520785),
holdout_syn, holdout_oos, awi_s03, holdout_only (=holdout real+syn+oos+para, NO s03),
near_miss, synthetic_neg_test per neg_kind, single_words (GSC non-command words).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path

import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "training"))
from src.vcm_data_loader import row_label          # noqa: E402
from src.vcm_infer import VCMInferencer            # noqa: E402

MANIFEST = ROOT / "data/manifests/me2_gold_v1.csv"
NEAR_MISS = ROOT / "data/external/me2_gold/near_miss/near_miss.csv"
COMPOSITE = ROOT / "data/manifests/composite_v1.csv"
OUT = ROOT / "results/me2_gold"
OUT.mkdir(parents=True, exist_ok=True)

V1_INIT = {
    "B2_s0": ("checkpoints/B2_s0/best.pt", "tcresnet", 1.0),
    "B2_s1": ("checkpoints/B2_s1/best.pt", "tcresnet", 1.0),
    "B2_s2": ("checkpoints/B2_s2/best.pt", "tcresnet", 1.0),
    "E1_s0": ("checkpoints/E1_s0/best.pt", "crnn_attn", 1.0),
    "E1_s1": ("checkpoints/E1_s1/best.pt", "crnn_attn", 1.0),
    "E1_s2": ("checkpoints/E1_s2/best.pt", "crnn_attn", 1.0),
    "G2_s0": ("checkpoints/G2_s0/best.pt", "dscnn", 1.0),
    "G2_s1": ("checkpoints/G2_s1/best.pt", "dscnn", 1.0),
    "G2_s2": ("checkpoints/G2_s2/best.pt", "dscnn", 1.0),
}
FINETUNED = {
    "B2f_s0": ("checkpoints/B2f_s0/best.pt", "tcresnet", 1.0),
    "B2f_s1": ("checkpoints/B2f_s1/best.pt", "tcresnet", 1.0),
    "B2f_s2": ("checkpoints/B2f_s2/best.pt", "tcresnet", 1.0),
    "E1f_s0": ("checkpoints/E1f_s0/best.pt", "crnn_attn", 1.0),
    "E1f_s1": ("checkpoints/E1f_s1/best.pt", "crnn_attn", 1.0),
    "E1f_s2": ("checkpoints/E1f_s2/best.pt", "crnn_attn", 1.0),
    "G2f_s0": ("checkpoints/G2f_s0/best.pt", "dscnn", 1.0),
    "G2f_s1": ("checkpoints/G2f_s1/best.pt", "dscnn", 1.0),
    "G2f_s2": ("checkpoints/G2f_s2/best.pt", "dscnn", 1.0),
}
ARCH_GROUP = {"B2f": "B2", "E1f": "E1", "G2f": "G2"}
ARCH_SEEDS = {
    "B2": ["B2_s0", "B2_s1", "B2_s2"], "B2f": ["B2f_s0", "B2f_s1", "B2f_s2"],
    "E1": ["E1_s0", "E1_s1", "E1_s2"], "E1f": ["E1f_s0", "E1f_s1", "E1f_s2"],
    "G2": ["G2_s0", "G2_s1", "G2_s2"], "G2f": ["G2f_s0", "G2f_s1", "G2f_s2"],
}
TAU_GRID = [round(x, 2) for x in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95]]


def is_command(k):
    return k not in ("UNKNOWN", "SILENCE")


def leaf(tl, sv):
    return row_label({"training_label": tl, "slot_value": sv}, "leaf")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


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


def _neg_kind(r):
    try:
        return json.loads(r["slots_json"]).get("neg_kind", "?")
    except Exception:
        return "?"


def load_slices():
    s = defaultdict(list)
    for r in csv.DictReader(open(MANIFEST)):
        sp, cid, spk = r["source_split"], r["corpus_id"], r["speaker_id"]
        item = (r["relative_path"], leaf(r["training_label"], r["slot_value"]),
                r["sha256"], spk)
        if sp == "validation":
            s["validation_all"].append(item)
            if cid == "gold_real":
                s["validation_real"].append(item)
            elif cid == "gold_synthetic":
                s["validation_synthetic"].append(item)
            elif cid == "gold_paraphrase":
                s["validation_paraphrase"].append(item)
            elif cid == "gold_oos":
                s["validation_oos"].append(item)
            elif cid == "synthetic_negative":
                s["synthetic_neg_test"].append(item)
                s[f"synthetic_neg_{_neg_kind(r)}"].append(item)
            elif cid == "personal_awi":
                s["awi_s02"].append(item)
        if sp == "test":
            if cid == "gold_real":
                s["holdout_real"].append(item)
            elif cid == "gold_synthetic":
                s["holdout_syn"].append(item)
            elif cid == "gold_oos":
                s["holdout_oos"].append(item)
            elif cid == "gold_paraphrase":
                s["holdout_paraphrase"].append(item)
            elif cid == "personal_awi":
                s["awi_s03"].append(item)
    for r in csv.DictReader(open(NEAR_MISS)):
        s["near_miss"].append((r["relative_path"], "UNKNOWN", "", r["speaker_id"]))
    s["tau_set"] = (s["validation_real"] + s["validation_synthetic"]
                    + s["validation_oos"] + s["synthetic_neg_test"] + s["awi_s02"])
    s["holdout_only"] = (s["holdout_real"] + s["holdout_syn"]
                         + s["holdout_oos"] + s["holdout_paraphrase"])
    return s


def load_single_words(gold_train_sha, gold_train_spk):
    # single_words (v1 GSC FAR slice) is optional: needs composite_v1.csv plus the
    # GSC v0.02 audio (download via reproduce.sh --with-v1-slices).
    if not COMPOSITE.exists():
        return []
    items = []
    for r in csv.DictReader(open(COMPOSITE)):
        if r["corpus_id"] != "gsc_v2" or r["source_split"] != "test":
            continue
        if r["sha256"] in gold_train_sha or r["speaker_id"] in gold_train_spk:
            continue
        if not (ROOT / r["relative_path"]).exists():
            continue  # GSC audio not downloaded
        items.append((r["relative_path"], r["training_label"], r["sha256"], r["speaker_id"]))
    return items


def gold_train_sets():
    sha, spk = set(), set()
    for r in csv.DictReader(open(MANIFEST)):
        if r["source_split"] == "train":
            sha.add(r["sha256"])
            spk.add(r["speaker_id"])
    return sha, spk


def predict_all(inf, slices):
    """Predict every slice, deduplicating identical audio paths (cache)."""
    out = {}
    cache = {}
    for name, items in slices.items():
        preds = []
        for path, true_key, _sha, _spk in items:
            if path not in cache:
                pk, prob = inf.predict_wav(path)
                cache[path] = (pk, prob)
            pk, prob = cache[path]
            preds.append((true_key, pk, prob))
        out[name] = preds
    return out


def slice_stats(preds, tau):
    counts = defaultdict(int)
    for t, p, pr in preds:
        counts[outcome(t, p, pr, tau)] += 1
    n_cmd = counts["correct"] + counts["wrong"] + counts["rejected"]
    n_unk = counts["false_accept"] + counts["correct_reject"]
    return {
        "n": len(preds), "n_command": n_cmd, "n_unknown_silence": n_unk,
        "correct": counts["correct"] / n_cmd if n_cmd else None,
        "wrong": counts["wrong"] / n_cmd if n_cmd else None,
        "rejected": counts["rejected"] / n_cmd if n_cmd else None,
        "FAR": counts["false_accept"] / n_unk if n_unk else None,
        "correct_ci": wilson(counts["correct"], n_cmd),
        "wrong_ci": wilson(counts["wrong"], n_cmd),
        "FAR_ci": wilson(counts["false_accept"], n_unk),
    }


def tau_for_model(seed_preds, slice_name):
    """largest tau with wrong<=1% for EVERY seed (min over seeds)."""
    taus = []
    for preds in seed_preds:
        p = preds[slice_name]
        best = 0.0
        for tau in TAU_GRID:
            st = slice_stats(p, tau)
            if st["wrong"] is not None and st["wrong"] <= 0.01:
                best = tau
        taus.append(best)
    return min(taus)


def per_leaf_accuracy(preds):
    acc = defaultdict(lambda: [0, 0])
    for t, p, pr in preds:
        if is_command(t):
            acc[t][1] += 1
            if p == t:
                acc[t][0] += 1
    return {k: {"correct": v[0], "total": v[1], "rate": v[0] / v[1] if v[1] else None}
            for k, v in sorted(acc.items())}


def run_infer(models, gpu, slices):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for name, (ckpt, arch, wm) in models.items():
        inf = VCMInferencer(ckpt, device="cuda", arch=arch, width_mult=wm, crop_mode="start")
        preds = predict_all(inf, slices)
        (OUT / f"preds_{name}.json").write_text(json.dumps(preds))
        print(f"[infer] {name} done ({len(preds)} slices)", flush=True)


def run_decide():
    preds_by_model = {}
    for f in sorted(OUT.glob("preds_*.json")):
        name = f.stem[len("preds_"):]
        preds_by_model[name] = json.loads(f.read_text())
    # group seeds -> arch
    groups = {arch: [preds_by_model[s] for s in seeds] for arch, seeds in ARCH_SEEDS.items()}

    arch_results = {}
    for arch, spreds in groups.items():
        tau = tau_for_model(spreds, "tau_set")
        pooled = defaultdict(list)
        for p in spreds:
            for k, v in p.items():
                pooled[k].extend(v)
        arch_results[arch] = {
            "tau": tau,
            "per_slice": {k: slice_stats(v, tau) for k, v in pooled.items()},
            "per_slice_tau0": {k: slice_stats(v, 0.0) for k, v in pooled.items()},
            "per_seed_tau": [tau_for_model([p], "tau_set") for p in spreds],
            "per_seed_wrong": [slice_stats(p["tau_set"], tau)["wrong"] for p in spreds],
            "per_leaf_validation_real": per_leaf_accuracy(pooled["validation_real"]),
        }
        print(f"  {arch}: tau={tau} per_seed_wrong={arch_results[arch]['per_seed_wrong']}")

    # per-run eval files (at the arch tau)
    for name, preds in preds_by_model.items():
        arch = name.rsplit("_", 1)[0]
        tau = arch_results[arch]["tau"]
        rec = {"tau": tau,
               "per_slice": {k: slice_stats(v, tau) for k, v in preds.items()},
               "per_slice_tau0": {k: slice_stats(v, 0.0) for k, v in preds.items()}}
        (OUT / f"eval_{name}.json").write_text(json.dumps(rec, indent=2))
    for arch, r in arch_results.items():
        (OUT / f"eval_arch_{arch}.json").write_text(json.dumps(r, indent=2))

    decision = {}
    for farch, varch in ARCH_GROUP.items():
        m, v = arch_results[farch], arch_results[varch]
        cond_seed = all(w is not None and w <= 0.01 for w in m["per_seed_wrong"])
        d_awi = m["per_slice"]["awi_s02"]["correct"] - v["per_slice"]["awi_s02"]["correct"]
        d_test = (m["per_slice"]["validation_real"]["correct"]
                  - v["per_slice"]["validation_real"]["correct"])
        d_far = (m["per_slice"]["single_words"]["FAR"]
                 - v["per_slice"]["single_words"]["FAR"])
        cond_awi = d_awi >= -0.01
        cond_test = d_test >= 0.05
        cond_far = d_far <= 0.01
        decision[farch] = {
            "replace_v1": bool(cond_seed and cond_awi and cond_test and cond_far),
            "v1_counterpart": varch,
            "conditions": {
                "every_seed_wrong_le_1pct_tauset": {"pass": cond_seed, "per_seed_wrong": m["per_seed_wrong"]},
                "awi_s02_ge_v1_minus_1pt": {"pass": cond_awi, "delta": d_awi},
                "gold_test_spk_201435283_ge_v1_plus_5pt": {"pass": cond_test, "delta": d_test},
                "single_words_far_le_v1_plus_1pt": {"pass": cond_far, "delta": d_far},
            },
            "numbers": {
                "v1_tau": v["tau"], "finetuned_tau": m["tau"],
                "v1_awi_s02_correct": v["per_slice"]["awi_s02"]["correct"],
                "ft_awi_s02_correct": m["per_slice"]["awi_s02"]["correct"],
                "v1_gold_test_correct": v["per_slice"]["validation_real"]["correct"],
                "ft_gold_test_correct": m["per_slice"]["validation_real"]["correct"],
                "v1_single_words_far": v["per_slice"]["single_words"]["FAR"],
                "ft_single_words_far": m["per_slice"]["single_words"]["FAR"],
                "v1_holdout_only_correct": v["per_slice"]["holdout_only"]["correct"],
                "ft_holdout_only_correct": m["per_slice"]["holdout_only"]["correct"],
                "v1_holdout_only_far": v["per_slice"]["holdout_only"]["FAR"],
                "ft_holdout_only_far": m["per_slice"]["holdout_only"]["FAR"],
            },
        }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2))
    print("decision:", json.dumps(decision, indent=2))


def run_step0(slices):
    results = {}
    for name, (ckpt, arch, wm) in V1_INIT.items():
        inf = VCMInferencer(ckpt, device="cuda", arch=arch, width_mult=wm, crop_mode="start")
        preds = predict_all(inf, slices)
        results[name] = {"tau": 0.9,
                         "per_slice": {k: slice_stats(v, 0.9) for k, v in preds.items()}}
        (OUT / f"step0_{name}.json").write_text(json.dumps(results[name], indent=2))
        print(f"  {name}: awi_s02={results[name]['per_slice']['awi_s02']['correct']:.4f} "
              f"holdout_real={results[name]['per_slice']['holdout_real']['correct']:.4f}")
    rows = []
    for name, r in results.items():
        row = {"model": name}
        for sl in ["awi_s02", "validation_real", "validation_synthetic",
                   "holdout_real", "holdout_syn", "holdout_oos", "holdout_only",
                   "awi_s03", "near_miss", "single_words"]:
            st = r["per_slice"][sl]
            row[f"{sl}_correct"] = st["correct"]
            row[f"{sl}_wrong"] = st["wrong"]
            row[f"{sl}_far"] = st["FAR"]
        rows.append(row)
    (OUT / "step0_table.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--decide", action="store_true")
    ap.add_argument("--models", default="", help="comma-separated run names (infer mode)")
    ap.add_argument("--gpu", type=int, default=None, help="CUDA_VISIBLE_DEVICES index")
    args = ap.parse_args()

    gsha, gspk = gold_train_sets()
    slices = load_slices()
    slices["single_words"] = load_single_words(gsha, gspk)

    if args.decide:
        run_decide()
        return
    if args.infer:
        all_models = {**V1_INIT, **FINETUNED}
        names = [n for n in args.models.split(",") if n] if args.models else list(all_models)
        models = {n: all_models[n] for n in names}
        run_infer(models, args.gpu, slices)
        return
    # default: step-0
    run_step0(slices)


if __name__ == "__main__":
    main()
