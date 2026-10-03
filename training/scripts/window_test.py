#!/usr/bin/env python3
"""4-s window robustness test (Step 4 / decision rule).

For each personal-command validation clip: place the trimmed speech at the
START / MIDDLE / END of a 4.0 s buffer filled with room tone (Awi's SILENCE
train + gsc_noise), run predict_wav on the full buffer, and report
correct / rejected / wrong per position. This is the demo capture path.

Usage:
  python scripts/window_test.py --checkpoint checkpoints/<run>/best.pt --tau 0.5
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchaudio

from src.vcm_data_loader import AudioProcessor, row_label
from src.vcm_infer import VCMInferencer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="data/manifests/composite_v1.csv")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--arch", default="tcresnet")
    ap.add_argument("--width-mult", type=float, default=1.0)
    ap.add_argument("--label-mode", default="leaf")
    ap.add_argument("--n-mels", type=int, default=40)
    ap.add_argument("--max-duration", type=float, default=3.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    inf = VCMInferencer(args.checkpoint, device=args.device, arch=args.arch,
                        width_mult=args.width_mult, label_mode=args.label_mode,
                        n_mels=args.n_mels, max_duration=args.max_duration)
    proc = inf.processor
    sr = inf.sample_rate
    buf_len = 4 * sr

    # room tone: personal SILENCE (quiet) + gsc_noise scaled to match the SILENCE RMS
    silences, noises = [], []
    with open(args.manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source_split"] == "train" and (
                r["corpus_id"] == "gsc_noise"
                or (r["corpus_id"] == "personal_awi" and r["training_label"] == "SILENCE")
            ):
                w, s = torchaudio.load(r["relative_path"])
                w = proc.to_mono_16k(w, s)
                (noises if r["corpus_id"] == "gsc_noise" else silences).append(w)
    sil_rms = float(torch.cat(silences, dim=-1).pow(2).mean().sqrt()) if silences else 1.0
    tone_clips = list(silences)
    for n in noises:
        nr = float(n.pow(2).mean().sqrt()) + 1e-8
        tone_clips.append(n * (sil_rms / nr))
    tone = torch.cat([c for c in tone_clips if c.shape[-1] > 0], dim=-1)
    if tone.shape[-1] < buf_len:
        reps = (buf_len + tone.shape[-1] - 1) // tone.shape[-1]
        tone = tone.repeat(1, reps)
    tone = tone[..., :buf_len]
    print(f"room tone built from {len(tone_clips)} clips ({buf_len} samples)")

    # personal command clips in the eval split
    clips = []
    with open(args.manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["source_split"] == args.split and r["corpus_id"] == "personal_awi" \
                    and r["training_label"] not in ("UNKNOWN", "SILENCE"):
                clips.append(r)
    print(f"personal command clips ({args.split}): {len(clips)}")

    pos_counts = defaultdict(Counter)
    for r in clips:
        true_key = row_label(r, "leaf")
        w, s = torchaudio.load(r["relative_path"])
        w = proc.to_mono_16k(w, s)
        speech = proc.trim_silence(w)
        if speech.shape[-1] == 0 or speech.shape[-1] >= buf_len:
            continue
        L = speech.shape[-1]
        positions = {"start": 0, "middle": (buf_len - L) // 2, "end": buf_len - L}
        for pos, off in positions.items():
            buf = tone.clone()
            buf[..., off:off + L] = speech
            pred_key, prob = inf.predict_wav(buf)
            if pred_key == true_key and prob >= args.tau:
                o = "correct"
            elif pred_key in ("UNKNOWN", "SILENCE") or prob < args.tau:
                o = "rejected"
            else:
                o = "wrong"
            pos_counts[pos][o] += 1

    print(f"\n=== 4-s window test @ tau={args.tau} ===")
    for pos in ("start", "middle", "end"):
        c = pos_counts[pos]
        n = sum(c.values())
        print(f"  {pos:7s} correct={c['correct']/n:.3f} rejected={c['rejected']/n:.3f} "
              f"wrong={c['wrong']/n:.3f}  (n={n})")


if __name__ == "__main__":
    main()
