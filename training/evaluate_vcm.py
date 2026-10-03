#!/usr/bin/env python3
"""Step 4 evaluation harness. Same inference path the RPi will use (vcm_infer).

Usage:
  python evaluate_vcm.py --checkpoint checkpoints/A7_s0/best.pt \
      --manifest data/manifests/composite_v1.csv --split validation [--tau 0.5]

Reports per-corpus-slice outcome rates (correct / rejected / wrong / FAR),
command macro-F1 / intent acc / leaf acc, the tau sweep, and writes
logs/eval_<name>.json + logs/eval_<name>.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vcm_data_loader import decode_label, row_label
from src.vcm_infer import VCMInferencer

TAU_SWEEP = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
SLICES = ["mark_optionb", "fsc", "gsc_v2", "personal_awi", "gsc_noise"]


def is_command_key(key: str) -> bool:
    return key not in ("UNKNOWN", "SILENCE")


def outcome(true_key: str, pred_key: str, prob: float, tau: float) -> str:
    """correct / rejected / wrong / false_accept."""
    if is_command_key(true_key):
        if pred_key == true_key and prob >= tau:
            return "correct"
        if (not is_command_key(pred_key)) or prob < tau:
            return "rejected"
        return "wrong"
    # true is UNKNOWN/SILENCE
    if is_command_key(pred_key) and prob >= tau:
        return "false_accept"
    return "correct_reject"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="data/manifests/composite_v1.csv")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--arch", default="tcresnet")
    ap.add_argument("--width-mult", type=float, default=1.0)
    ap.add_argument("--label-mode", default="leaf")
    ap.add_argument("--n-mels", type=int, default=40)
    ap.add_argument("--max-duration", type=float, default=3.0)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--name", default=None)
    ap.add_argument("--eval-corpora", default=None,
                    help="comma-separated corpus_ids to include (default: all)")
    ap.add_argument("--save-preds", default=None,
                    help="write per-clip preds as JSON (sample_id,true,pred,prob,corpus)")
    args = ap.parse_args()

    corpora = set(args.eval_corpora.split(",")) if args.eval_corpora else None

    inf = VCMInferencer(args.checkpoint, device=args.device, arch=args.arch,
                        width_mult=args.width_mult, label_mode=args.label_mode,
                        n_mels=args.n_mels, max_duration=args.max_duration)

    rows = []
    with open(args.manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source_split"] == args.split and (corpora is None or r["corpus_id"] in corpora):
                rows.append(r)
    print(f"evaluating {len(rows)} rows ({args.split}) on {args.device}")

    recs = []  # (true_key, pred_key, prob, corpus)
    preds = []
    for i, r in enumerate(rows):
        true_key = row_label(r, "leaf")
        pred_key, prob = inf.predict_wav(r["relative_path"])
        recs.append((true_key, pred_key, prob, r["corpus_id"]))
        if args.save_preds:
            preds.append({"sample_id": r["sample_id"], "true": true_key,
                          "pred": pred_key, "prob": prob, "corpus": r["corpus_id"]})
        if (i + 1) % 5000 == 0:
            print(f"  ... {i + 1}/{len(rows)}")

    if args.save_preds:
        Path(args.save_preds).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_preds).write_text(json.dumps(preds))
        print(f"saved {len(preds)} per-clip preds -> {args.save_preds}")

    name = args.name or f"{Path(args.checkpoint).parent.name}_{args.split}"
    result = {"checkpoint": args.checkpoint, "split": args.split, "n": len(rows), "tau": args.tau}

    # ---- per-slice outcomes at the given tau ----
    def slice_stats(subset, tau):
        c = defaultdict(int)
        for t, p, pr, _ in subset:
            c[outcome(t, p, pr, tau)] += 1
        n_cmd = c["correct"] + c["rejected"] + c["wrong"]
        n_unk = c["false_accept"] + c["correct_reject"]
        return {
            "n": len(subset),
            "correct": c["correct"] / n_cmd if n_cmd else None,
            "rejected": c["rejected"] / n_cmd if n_cmd else None,
            "wrong": c["wrong"] / n_cmd if n_cmd else None,
            "FAR": c["false_accept"] / n_unk if n_unk else None,
            "n_command": n_cmd, "n_unknown_silence": n_unk,
        }

    per_slice = {}
    for s in SLICES:
        sub = [x for x in recs if x[3] == s]
        if sub:
            per_slice[s] = slice_stats(sub, args.tau)
    per_slice["combined"] = slice_stats(recs, args.tau)
    result["per_slice"] = per_slice

    # ---- metrics (tau=0 raw) ----
    import numpy as np
    from sklearn.metrics import f1_score, confusion_matrix
    yt = [t for t, _, _, _ in recs]
    yp = [p for _, p, _, _ in recs]
    cmd_idx = [i for i, t in enumerate(yt) if is_command_key(t)]
    yt_c = [yt[i] for i in cmd_idx]
    yp_c = [yp[i] for i in cmd_idx]
    present_cmd = sorted(set(yt_c))
    macro_f1 = f1_score(yt_c, yp_c, labels=present_cmd, average="macro", zero_division=0) if yt_c else None
    leaf_acc = float(np.mean([t == p for t, p in zip(yt, yp)]))
    intent_true = np.array([decode_label(t)[0] for t in yt])
    intent_pred = np.array([decode_label(p)[0] for p in yp])
    intent_acc = float((intent_true == intent_pred).mean())
    result["metrics"] = {
        "command_macro_f1": float(macro_f1) if macro_f1 is not None else None,
        "leaf_acc": leaf_acc,
        "intent_acc": intent_acc,
        "confusion": confusion_matrix(yt_c, yp_c, labels=present_cmd).tolist() if yt_c else [],
        "labels": present_cmd,
    }

    # ---- tau sweep ----
    sweep = {}
    for tau in TAU_SWEEP:
        c = defaultdict(int)
        for t, p, pr, _ in recs:
            c[outcome(t, p, pr, tau)] += 1
        n_cmd = c["correct"] + c["rejected"] + c["wrong"]
        n_unk = c["false_accept"] + c["correct_reject"]
        sweep[tau] = {
            "correct": c["correct"] / n_cmd if n_cmd else None,
            "rejected": c["rejected"] / n_cmd if n_cmd else None,
            "wrong": c["wrong"] / n_cmd if n_cmd else None,
            "FAR": c["false_accept"] / n_unk if n_unk else None,
        }
    result["tau_sweep"] = sweep

    Path("logs").mkdir(exist_ok=True)
    out = f"logs/eval_{name}.json"
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")

    # ---- console summary ----
    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "-"

    print(f"\n=== {name} @ tau={args.tau} ===")
    print(f"  command_macro_f1={macro_f1:.4f}  intent_acc={intent_acc:.4f}  leaf_acc={leaf_acc:.4f}")
    for s, st in per_slice.items():
        if st["n_command"]:
            print(f"  {s:14s} correct={fmt(st['correct'])} rejected={fmt(st['rejected'])} "
                  f"wrong={fmt(st['wrong'])} FAR={fmt(st['FAR'])} "
                  f"(cmd={st['n_command']} unk={st['n_unknown_silence']})")
    print("\n=== tau sweep (correct/rejected/wrong/FAR) ===")
    for tau in TAU_SWEEP:
        st = sweep[tau]
        print(f"  tau={tau:.2f}: correct={fmt(st['correct'])} rejected={fmt(st['rejected'])} "
              f"wrong={fmt(st['wrong'])} FAR={fmt(st['FAR'])}")


if __name__ == "__main__":
    main()
