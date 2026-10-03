#!/usr/bin/env python3
"""Step 3 unit check: save before/after wave-aug samples + report SNR.

Saves logs/aug_samples/NN_<label>_{before,after}.wav (Awi can listen) and
logs/aug_samples/aug_report.json. SNR is 10*log10(P_before / P_diff) on the
length-aligned difference (speed-perturb clips are compared on the overlap).
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torchaudio

from src.vcm_data_loader import AudioProcessor, WaveAugment

WAVE_CFG = {
    "noise_mix": {"p": 0.6, "snr_db": [5, 25]},
    "reverb": {"p": 0.3, "rt60_s": [0.15, 0.6]},
    "speed_perturb": {"p": 0.5, "factors": [0.9, 1.0, 1.1]},
    "gain": {"p": 0.5, "db": 6.0},
    "mic_bandlimit": {"p": 0.2, "cutoff_hz": [3500, 7500]},
}


def main() -> None:
    out = Path("logs/aug_samples")
    out.mkdir(parents=True, exist_ok=True)
    proc = AudioProcessor(sample_rate=16000, n_mels=40, n_fft=512, hop_length=160,
                          f_min=50, f_max=7600, top_db=80.0, trim_db=40.0)

    with open("data/manifests/composite_v1.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["source_split"] == "train"]

    bank = []
    for r in rows:
        if r["corpus_id"] == "gsc_noise" or (r["corpus_id"] == "personal_awi" and r["training_label"] == "SILENCE"):
            w, sr = torchaudio.load(r["relative_path"])
            bank.append(proc.to_mono_16k(w, sr))
    print(f"noise bank size: {len(bank)}")

    random.seed(42)
    by_corpus = {}
    for r in rows:
        if r["training_label"] in ("UNKNOWN", "SILENCE"):
            continue
        by_corpus.setdefault(r["corpus_id"], []).append(r)
    picked = []
    for corpus in ("personal_awi", "fsc", "mark_optionb", "gsc_v2"):
        if corpus in by_corpus:
            picked += random.sample(by_corpus[corpus], min(5, len(by_corpus[corpus])))

    np.random.seed(0)
    aug = WaveAugment(WAVE_CFG, bank, 16000)
    report, snrs = [], []
    for i, r in enumerate(picked):
        w, sr = torchaudio.load(r["relative_path"])
        w = proc.to_mono_16k(w, sr)
        before = w.clone()
        after = aug(w.clone())
        n = min(before.shape[-1], after.shape[-1])
        ps = (before[..., :n] ** 2).mean().item() + 1e-10
        pn = ((after[..., :n] - before[..., :n]) ** 2).mean().item() + 1e-10
        snr = 10 * np.log10(ps / pn)
        snrs.append(snr)
        tag = r["training_label"].replace("|", "_")
        torchaudio.save(str(out / f"{i:02d}_{tag}_before.wav"), before, 16000)
        torchaudio.save(str(out / f"{i:02d}_{tag}_after.wav"), after, 16000)
        report.append({"idx": i, "label": r["training_label"], "corpus": r["corpus_id"],
                       "snr_db": round(float(snr), 2)})

    (out / "aug_report.json").write_text(json.dumps(report, indent=2))
    print(f"saved {len(picked)} before/after pairs to {out}")
    print(f"mean SNR: {np.mean(snrs):.2f} dB | min {min(snrs):.2f} | max {max(snrs):.2f}")


if __name__ == "__main__":
    main()
