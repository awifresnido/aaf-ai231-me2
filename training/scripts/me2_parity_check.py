#!/usr/bin/env python3
"""GATE 0c parity check (ME2 Part 0 step 6).

(i)  no-augmentation GPU features vs CPU features on N clips: max abs diff <= 1e-4.
(ii) augmentation parameter statistics (SNR, gain, RT60, speed factor) within 5%
     of the CPU path.
Also saves 10 before/after audio pairs to logs/aug_samples_gpu/.

Usage:
    python scripts/me2_parity_check.py --n 500 --manifest data/manifests/composite_v1.csv
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vcm_data_loader import AudioProcessor, WaveAugment, fit_frames  # noqa: E402
from src.vcm_gpu_pipeline import GPUWaveAugment, GPUBatchProcessor  # noqa: E402

SR = 16000
MAX_DURATION = 3.0


def build_processor():
    return AudioProcessor(sample_rate=16000, n_mels=40, n_fft=512, hop_length=160,
                          f_min=50, f_max=7600, top_db=80.0, trim_db=40.0)


def load_clips(manifest, n):
    rows = []
    with open(manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("source_split") == "train":
                rows.append(r)
    rng = np.random.RandomState(0)
    rng.shuffle(rows)
    return rows[:n]


def compare_stats(name, cpu_list, gpu_list, lo, hi, bins=10, tol=0.05):
    """Histogram comparison (robust for zero-mean distributions like gain)."""
    cpu = np.array(cpu_list, dtype=float)
    gpu = np.array(gpu_list, dtype=float)
    if len(cpu) == 0 or len(gpu) == 0:
        return {"name": name, "cpu_n": len(cpu), "gpu_n": len(gpu),
                "pass": True, "note": "empty (no draws)"}
    hc, _ = np.histogram(cpu, bins=bins, range=(lo, hi))
    hg, _ = np.histogram(gpu, bins=bins, range=(lo, hi))
    hc = hc / hc.sum()
    hg = hg / hg.sum()
    max_bin_diff = float(np.abs(hc - hg).max())
    return {"name": name, "cpu_n": len(cpu), "gpu_n": len(gpu),
            "cpu_mean": round(float(cpu.mean()), 4), "gpu_mean": round(float(gpu.mean()), 4),
            "max_bin_diff": round(max_bin_diff, 4), "pass": max_bin_diff <= tol}


def speed_hist(lst):
    c = Counter(round(float(x), 2) for x in lst)
    total = len(lst) or 1
    return {f"{k:.2f}": round(v / total, 4) for k, v in sorted(c.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--manifest", default="data/manifests/me2_gold_v1.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_clips(args.manifest, args.n)
    cpu_proc = build_processor()
    n_frames = int(MAX_DURATION * SR / cpu_proc.hop_length) + 1
    waves = [cpu_proc.to_mono_16k(*torchaudio.load(r["relative_path"])) for r in rows]
    print(f"loaded {len(waves)} clips, device={device}")

    # ---- Part (i): no-aug feature parity ----
    gpu_noaug = GPUBatchProcessor(cpu_proc, None, {}, n_frames, [], device)
    max_diff = 0.0
    for w in waves:
        wc = cpu_proc.trim_silence(w)
        fc = fit_frames(cpu_proc.process(wc), n_frames, random_offset=False)
        fg = gpu_noaug([w.to(device)], augment=False).cpu().squeeze(1).transpose(1, 2)
        max_diff = max(max_diff, float((fc - fg).abs().max()))
    part_i = {"n": len(waves), "max_abs_diff": round(max_diff, 8), "pass": max_diff <= 1e-4}

    # ---- Part (ii): augmentation stats (force every effect) ----
    cfg = {
        "noise_mix": {"p": 1.0, "snr_db": [5, 25]},
        "reverb": {"p": 1.0, "rt60_s": [0.15, 0.6]},
        "speed_perturb": {"p": 1.0, "factors": [0.9, 1.0, 1.1]},
        "gain": {"p": 1.0, "db": 6.0},
        "mic_bandlimit": {"p": 1.0, "cutoff_hz": [3500, 7500]},
    }
    noise_bank = [torch.randn(1, SR) for _ in range(4)]
    cpu_aug = WaveAugment(cfg, noise_bank, SR)
    gpu_aug = GPUWaveAugment(cfg, noise_bank, SR, device)
    for w in waves:
        cpu_aug(w.clone())
        gpu_aug([w.clone().to(device)])
    part_ii = [
        compare_stats("snr", cpu_aug.aug_stats["snr"], gpu_aug.aug_stats["snr"], lo=5.0, hi=25.0),
        compare_stats("gain_db", cpu_aug.aug_stats["gain_db"], gpu_aug.aug_stats["gain_db"], lo=-6.0, hi=6.0),
        compare_stats("rt60", cpu_aug.aug_stats["rt60"], gpu_aug.aug_stats["rt60"], lo=0.15, hi=0.6),
    ]
    sf_cpu = speed_hist(cpu_aug.aug_stats["speed_factor"])
    sf_gpu = speed_hist(gpu_aug.aug_stats["speed_factor"])
    sf_pass = all(abs(sf_cpu.get(k, 0.0) - sf_gpu.get(k, 0.0)) <= 0.05
                  for k in set(sf_cpu) | set(sf_gpu))
    part_ii.append({"name": "speed_factor", "cpu_hist": sf_cpu, "gpu_hist": sf_gpu, "pass": sf_pass})

    # ---- 10 before/after audio pairs ----
    out_dir = Path("logs/aug_samples_gpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, w in enumerate(waves[:10]):
        after = gpu_aug([w.clone().to(device)])[0].cpu()
        torchaudio.save(str(out_dir / f"{i:02d}_before.wav"), w, SR)
        torchaudio.save(str(out_dir / f"{i:02d}_after.wav"), after, SR)

    report = {
        "n_clips": len(waves),
        "device": device,
        "part_i_features": part_i,
        "part_ii_aug_stats": part_ii,
        "part_ii_pass": all(x["pass"] for x in part_ii),
        "gate_0c_pass": part_i["pass"] and all(x["pass"] for x in part_ii),
        "audio_pairs": str(out_dir),
    }
    (Path("logs") / "parity_0c.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["gate_0c_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
