#!/usr/bin/env python3
"""Latency benchmark for the Raspberry Pi 4 (ME2 Part 0 step 7).

Runs the app's ONNX engine end-to-end — feature extraction (16 kHz -> trim ->
log-mel 40 + CMVN -> fit 301 frames) then the ONNX model — on a 4.0 s capture,
with 30 warm-up and >= 200 timed runs. Writes results/<run>/latency_pi4.json.

Dependencies (same as the app): torch, torchaudio, onnxruntime, numpy.

Usage:
    python scripts/bench_pi.py --deploy exports/E1_s0 --run E1_s0
    python scripts/bench_pi.py --deploy exports/G2_s0 --run G2_s0 --threads 4 --runs 300
    python scripts/bench_pi.py --deploy exports/G2_s0 --run G2_s0 --wav /path/to/4s.wav

The ONNX input is named ``mel`` (float32 [1, 1, n_frames, n_mels]) and the output
``logits`` (float32 [1, num_classes]); both are read dynamically from the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vcm_data_loader import AudioProcessor, fit_frames  # noqa: E402

CAPTURE_SECONDS = 4.0


def read_sys_text(*paths) -> str:
    for p in paths:
        try:
            v = Path(p).read_text().strip()
            if v:
                return v
        except OSError:
            continue
    return "unknown"


def cpu_governor() -> str:
    return read_sys_text(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
        "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor",
    )


def soc_temperature_c() -> float | None:
    for p in ("/sys/class/thermal/thermal_zone0/temp", "/sys/class/thermal/thermal_zone0/hwmon/temp1_input"):
        raw = read_sys_text(p)
        if raw != "unknown" and raw.lstrip("-").isdigit():
            val = int(raw) / 1000.0  # millidegrees C on the Pi
            return val if val < 200 else val / 1000.0
    return None


def synth_capture(sample_rate: int, seconds: float) -> torch.Tensor:
    """A 4.0 s 16 kHz tone+noise so trim_silence keeps the full clip."""
    n = int(sample_rate * seconds)
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    tone = 0.35 * torch.sin(2 * np.pi * 220.0 * t) + 0.25 * torch.sin(2 * np.pi * 440.0 * t)
    tone = tone + 0.05 * torch.randn(n)
    return tone.unsqueeze(0)  # (1, n)


def load_capture(wav_path: str | None, sample_rate: int, seconds: float) -> torch.Tensor:
    if wav_path:
        wav, sr = torchaudio.load(wav_path)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        wav = wav.mean(dim=0, keepdim=True) if wav.shape[0] > 1 else wav  # mono
    else:
        wav = synth_capture(sample_rate, seconds)
    target = int(sample_rate * seconds)
    if wav.shape[-1] < target:
        wav = torch.nn.functional.pad(wav, (0, target - wav.shape[-1]))
    return wav[:, :target]


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(sorted_ms: list[float], q: float) -> float:
    if not sorted_ms:
        return float("nan")
    k = (len(sorted_ms) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_ms) - 1)
    frac = k - lo
    return sorted_ms[lo] * (1 - frac) + sorted_ms[hi] * frac


def main() -> int:
    ap = argparse.ArgumentParser(description="Pi4 latency benchmark for the VCM ONNX engine")
    ap.add_argument("--deploy", required=True, help="export dir containing deploy.json + model.onnx")
    ap.add_argument("--run", required=True, help="run name (also the results/<run> directory)")
    ap.add_argument("--wav", default=None, help="optional 4.0 s 16 kHz capture; synthesized if omitted")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--threads", type=int, default=None, help="onnxruntime intra-op threads (default: auto)")
    args = ap.parse_args()

    deploy_dir = Path(args.deploy)
    deploy = json.loads((deploy_dir / "deploy.json").read_text())
    n_mels = int(deploy.get("n_mels", 40))
    n_frames = int(deploy.get("n_frames", 301))
    sample_rate = int(deploy.get("sample_rate", 16000))
    crop_mode = deploy.get("crop_mode", "start")
    if crop_mode != "start":
        raise SystemExit(f"crop_mode={crop_mode!r} not supported yet (only 'start' in Part 0)")

    onnx_path = deploy_dir / deploy.get("onnx", "model.onnx")

    # ---- feature front-end (same as training / app) ----
    processor = AudioProcessor(
        sample_rate=sample_rate, n_mels=n_mels, n_fft=512, hop_length=160,
        f_min=50, f_max=7600, top_db=80.0, trim_db=40.0,
    )

    def extract_features(wav: torch.Tensor) -> np.ndarray:
        wav = processor.trim_silence(wav)
        feat = processor.process(wav)                     # (1, n_mels, t)
        feat = fit_frames(feat, n_frames, random_offset=False)  # (1, n_mels, n_frames)
        return feat.transpose(1, 2).unsqueeze(0).contiguous().numpy()  # (1, 1, n_frames, n_mels)

    # ---- ONNX session ----
    sess_opts = ort.SessionOptions()
    threads = args.threads if args.threads is not None else int(os.environ.get("OMP_NUM_THREADS", 0)) or None
    if threads:
        sess_opts.intra_op_num_threads = threads
        sess_opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_opts, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    out_names = [o.name for o in session.get_outputs()]

    wav = load_capture(args.wav, sample_rate, CAPTURE_SECONDS)

    # ---- warm-up ----
    for _ in range(args.warmup):
        feat = extract_features(wav)
        session.run(out_names, {in_name: feat})

    # ---- timed runs ----
    feat_lat, model_lat, total_lat = [], [], []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        feat = extract_features(wav)
        t1 = time.perf_counter()
        session.run(out_names, {in_name: feat})
        t2 = time.perf_counter()
        feat_lat.append((t1 - t0) * 1000.0)
        model_lat.append((t2 - t1) * 1000.0)
        total_lat.append((t2 - t0) * 1000.0)

    total_lat.sort()
    p50, p95, p99 = percentile(total_lat, 0.50), percentile(total_lat, 0.95), percentile(total_lat, 0.99)

    # ---- model size incl. external weight data ----
    data_files = sorted(set(deploy_dir.glob("*.onnx.data")))
    size_bytes = onnx_path.stat().st_size + sum(f.stat().st_size for f in data_files)

    result = {
        "run": args.run,
        "capture_seconds": CAPTURE_SECONDS,
        "n_warmup": args.warmup,
        "n_runs": args.runs,
        "crop_mode": crop_mode,
        "onnx": str(onnx_path),
        "onnx_md5": md5_of(onnx_path),
        "onnx_external_data": [{"file": f.name, "md5": md5_of(f)} for f in data_files],
        "model_size_bytes": size_bytes,
        "model_size_mb": round(size_bytes / (1024 * 1024), 3),
        "onnxruntime_version": ort.__version__,
        "threads": threads if threads else (session.get_session_options().intra_op_num_threads or 0),
        "cpu_governor": cpu_governor(),
        "soc_temperature_c": soc_temperature_c(),
        "latency_total_ms": {
            "p50": round(p50, 3),
            "p95": round(p95, 3),
            "p99": round(p99, 3),
            "mean": round(statistics.mean(total_lat), 3),
        },
        "latency_features_ms": {
            "p50": round(percentile(sorted(feat_lat), 0.50), 3),
            "p95": round(percentile(sorted(feat_lat), 0.95), 3),
        },
        "latency_model_ms": {
            "p50": round(percentile(sorted(model_lat), 0.50), 3),
            "p95": round(percentile(sorted(model_lat), 0.95), 3),
        },
        "rtf": {
            "p50": round(p50 / (CAPTURE_SECONDS * 1000.0), 4),
            "p95": round(p95 / (CAPTURE_SECONDS * 1000.0), 4),
            "p99": round(p99 / (CAPTURE_SECONDS * 1000.0), 4),
        },
    }

    out_dir = Path("results") / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "latency_pi4.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
