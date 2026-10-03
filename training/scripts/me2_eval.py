#!/usr/bin/env python3
"""Part 5: evaluate v1 (B2/E1/G2 s0) + me2 (B2m/E1m/G2m s0..s4) on the me2 slices.

Computes tau (largest tau with wrong<=1% for every seed), per-slice correct/wrong/
reject/FAR with Wilson 95% CIs, and applies the pre-registered decision rule.
Writes results/me2/eval_<model>.json + results/me2/decision.json.
"""
import csv, json, math, sys
from pathlib import Path
from collections import defaultdict

import torch
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from src.vcm_data_loader import row_label          # noqa: E402
from src.vcm_infer import VCMInferencer            # noqa: E402

MANIFEST = ROOT / "data/manifests/me2_master_v1.csv"
NEAR_MISS = ROOT / "data/external/me2_master/near_miss/near_miss.csv"
COMPOSITE = ROOT / "data/manifests/composite_v1.csv"
OUT = ROOT / "results/me2"
OUT.mkdir(parents=True, exist_ok=True)

V1 = {
    "B2": ("checkpoints/B2_s0/best.pt", "tcresnet", 1.0),
    "E1": ("checkpoints/E1_s0/best.pt", "crnn_attn", 1.0),
    "G2": ("checkpoints/G2_s0/best.pt", "dscnn", 1.0),
}
ME2_ARCH = {"B2m": "tcresnet", "E1m": "crnn_attn", "G2m": "dscnn"}
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


def load_slices():
    s = defaultdict(list)
    for r in csv.DictReader(open(MANIFEST)):
        sp, cid, spk = r["source_split"], r["corpus_id"], r["speaker_id"]
        item = (r["relative_path"], leaf(r["training_label"], r["slot_value"]),
                r["sha256"], spk)
        if sp == "validation" and cid == "me2_real":
            s["validation_real"].append(item)
        if sp == "validation" and cid == "personal_awi":
            s["awi_s02"].append(item)
        if sp == "test" and spk == "202520785":
            s["holdout_real"].append(item)
        if sp == "test" and cid == "me2_syn":
            s["holdout_syn"].append(item)
        if sp == "test" and cid == "personal_awi":
            s["awi_s03"].append(item)
        if spk in ("201435283", "202520785"):
            s["fair_real"].append(item)
    for r in csv.DictReader(open(NEAR_MISS)):
        s["near_miss"].append((r["relative_path"], "UNKNOWN", "", ""))
    return s


def load_single_words_clean(me2_train_sha, me2_train_spk, v1_train_sha, v1_train_spk):
    items = []
    for r in csv.DictReader(open(COMPOSITE)):
        if r["corpus_id"] != "gsc_v2" or r["source_split"] != "test":
            continue
        if r["sha256"] in me2_train_sha or r["sha256"] in v1_train_sha:
            continue
        if r["speaker_id"] in me2_train_spk or r["speaker_id"] in v1_train_spk:
            continue
        items.append((r["relative_path"], r["training_label"], r["sha256"], r["speaker_id"]))
    return items


def training_sets():
    me2_sha, me2_spk = set(), set()
    v1_sha, v1_spk = set(), set()
    for r in csv.DictReader(open(MANIFEST)):
        if r["source_split"] == "train":
            me2_sha.add(r["sha256"])
            me2_spk.add(r["speaker_id"])
    for r in csv.DictReader(open(COMPOSITE)):
        if r["source_split"] == "train":
            v1_sha.add(r["sha256"])
            v1_spk.add(r["speaker_id"])
    return me2_sha, me2_spk, v1_sha, v1_spk


def predict_all(inf, slices):
    """Return {slice_name: [(true_key, pred_key, prob), ...]}."""
    out = {}
    for name, items in slices.items():
        preds = []
        for path, true_key, _sha, _spk in items:
            pk, prob = inf.predict_wav(path)
            preds.append((true_key, pk, prob))
        out[name] = preds
    return out


def slice_stats(preds, tau):
    """preds = [(true, pred, prob)] -> {correct, wrong, rejected, FAR, n, n_cmd, n_unk}."""
    counts = defaultdict(int)
    for t, p, pr in preds:
        counts[outcome(t, p, pr, tau)] += 1
    n_cmd = counts["correct"] + counts["wrong"] + counts["rejected"]
    n_unk = counts["false_accept"] + counts["correct_reject"]
    n = len(preds)
    return {
        "n": n,
        "n_command": n_cmd,
        "n_unknown_silence": n_unk,
        "correct": counts["correct"] / n_cmd if n_cmd else None,
        "wrong": counts["wrong"] / n_cmd if n_cmd else None,
        "rejected": counts["rejected"] / n_cmd if n_cmd else None,
        "FAR": counts["false_accept"] / n_unk if n_unk else None,
        "correct_ci": wilson(counts["correct"], n_cmd),
        "wrong_ci": wilson(counts["wrong"], n_cmd),
        "FAR_ci": wilson(counts["false_accept"], n_unk),
    }


def tau_for_model(all_preds_by_seed, slice_name):
    """largest tau with wrong<=1% for EVERY seed (min over seeds)."""
    taus = []
    for preds in all_preds_by_seed:
        p = preds[slice_name]
        best = 0.0
        for tau in TAU_GRID:
            st = slice_stats(p, tau)
            if st["wrong"] is not None and st["wrong"] <= 0.01:
                best = tau
        taus.append(best)
    return min(taus)


def main():
    me2_sha, me2_spk, v1_sha, v1_spk = training_sets()
    slices = load_slices()
    slices["single_words_clean"] = load_single_words_clean(
        me2_sha, me2_spk, v1_sha, v1_spk)
    print("slices:", {k: len(v) for k, v in slices.items()})

    results = {}          # model_name -> {tau, per_slice_stats}
    me2_seed_preds = {}   # model -> [ {slice: preds} per seed ]

    # ---- v1 models (s0 only) ----
    for name, (ckpt, arch, wm) in V1.items():
        inf = VCMInferencer(ckpt, device="cuda", arch=arch, width_mult=wm)
        preds = predict_all(inf, slices)
        results[name] = {
            "tau": 0.9,  # v1 operating tau (reported separately)
            "per_slice": {k: slice_stats(v, 0.9) for k, v in preds.items()},
            "per_slice_tau0": {k: slice_stats(v, 0.0) for k, v in preds.items()},
        }
        print(f"  v1 {name}: done")

    # ---- me2 models (5 seeds each) ----
    for model in ["B2m", "E1m", "G2m"]:
        arch = ME2_ARCH[model]
        seed_preds = []
        for s in range(5):
            run = f"{model}_s{s}"
            ckpt = f"checkpoints/{run}/best.pt"
            inf = VCMInferencer(ckpt, device="cuda", arch=arch, width_mult=1.0)
            preds = predict_all(inf, slices)
            seed_preds.append(preds)
        me2_seed_preds[model] = seed_preds
        tau = tau_for_model(seed_preds, "validation_real")
        # report the model at its tau (averaged/pooled over seeds)
        pooled = defaultdict(list)
        for preds in seed_preds:
            for k, v in preds.items():
                pooled[k].extend(v)
        results[model] = {
            "tau": tau,
            "per_slice": {k: slice_stats(v, tau) for k, v in pooled.items()},
            "per_slice_tau0": {k: slice_stats(v, 0.0) for k, v in pooled.items()},
            "per_seed_tau": [tau_for_model([p], "validation_real") for p in seed_preds],
            "per_seed_wrong": [
                slice_stats(p["validation_real"], tau)["wrong"] for p in seed_preds],
        }
        print(f"  {model}: tau={tau} per_seed_wrong={results[model]['per_seed_wrong']}")

    # ---- decision rule ----
    decision = {}
    for model, v1 in [("B2m", "B2"), ("E1m", "E1"), ("G2m", "G2")]:
        m, v = results[model], results[v1]
        # v1 numbers are computed at tau 0.9 (its operating point); me2 at its tau.
        cond_seed = all(w is not None and w <= 0.01 for w in m["per_seed_wrong"])
        d_awi = (m["per_slice"]["awi_s02"]["correct"]
                 - v["per_slice"]["awi_s02"]["correct"])
        d_fair = (m["per_slice"]["fair_real"]["correct"]
                  - v["per_slice"]["fair_real"]["correct"])
        d_far = (m["per_slice"]["single_words_clean"]["FAR"]
                 - v["per_slice"]["single_words_clean"]["FAR"])
        cond_awi = d_awi >= -0.01
        cond_fair = d_fair >= 0.05
        cond_far = d_far <= 0.01
        decision[model] = {
            "replace_v1": bool(cond_seed and cond_awi and cond_fair and cond_far),
            "v1_counterpart": v1,
            "conditions": {
                "every_seed_wrong_le_1pct": {"pass": cond_seed,
                                             "per_seed_wrong": m["per_seed_wrong"]},
                "awi_s02_ge_v1_minus_1pt": {"pass": cond_awi, "delta": d_awi},
                "fair_real_ge_v1_plus_5pt": {"pass": cond_fair, "delta": d_fair},
                "single_words_far_le_v1_plus_1pt": {"pass": cond_far, "delta": d_far},
            },
            "numbers": {
                "v1_tau": v["tau"], "me2_tau": m["tau"],
                "v1_awi_s02_correct": v["per_slice"]["awi_s02"]["correct"],
                "me2_awi_s02_correct": m["per_slice"]["awi_s02"]["correct"],
                "v1_fair_real_correct": v["per_slice"]["fair_real"]["correct"],
                "me2_fair_real_correct": m["per_slice"]["fair_real"]["correct"],
                "v1_single_words_far": v["per_slice"]["single_words_clean"]["FAR"],
                "me2_single_words_far": m["per_slice"]["single_words_clean"]["FAR"],
            },
        }

    for name, r in results.items():
        (OUT / f"eval_{name}.json").write_text(json.dumps(r, indent=2))
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2))
    print("decision:", json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
