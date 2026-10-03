#!/usr/bin/env python
"""Isolate why a registered model is slow in the app's torch engine.

Times ONLY the model forward pass (features computed once, outside the loop)
on a fixed 3.0 s input, under several runtime settings, plus ONNX Runtime if an
exported model is given. Same numbers the app shows as ``inference_ms``.

Usage (from $APP, app venv active):
    python scripts/bench_engine.py --model E1_s0 --onnx ../tiny-vcm/exports/E1_s0/model.onnx
    python scripts/bench_engine.py --model B2_s0
    python scripts/bench_engine.py --model E1_s0 --wav some_command.wav

For models whose manifest ships ONNX (engine: onnx_tinyvcm) the torch baseline is
taken from <tiny_vcm_root>/checkpoints/<model>/best.pt automatically.

Reads models/vcm/<id>/manifest.yaml and config/demo.yaml (tiny_vcm_root).
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edge.engines.torch_tinyvcm import TorchTinyVCMEngine  # noqa: E402
from vcm_common.manifest import ModelManifest  # noqa: E402


def tiny_vcm_root() -> str | None:
    cfg = yaml.safe_load((ROOT / "config" / "demo.yaml").read_text())

    def find(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "tiny_vcm_root":
                    return v
                r = find(v)
                if r:
                    return r
        return None

    root = find(cfg)
    if root is None:
        return None
    path = Path(root)
    return str(path if path.is_absolute() else (ROOT / path).resolve())


def timeit(fn, n: int, warmup: int = 5) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return statistics.median(ts), ts[int(0.95 * (len(ts) - 1))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model id under models/vcm/")
    ap.add_argument("--wav", default=None, help="optional real clip; default = 3 s of low noise + tone")
    ap.add_argument("--onnx", default=None, help="optional exported model.onnx for comparison")
    ap.add_argument("-n", type=int, default=30)
    args = ap.parse_args()

    import torch

    m = ModelManifest.load(ROOT / "models" / "vcm" / args.model)
    w = m.weights_path()
    if w is not None and w.suffix == ".onnx":
        # ONNX-engine manifests (B2_s0 / E1_s0 / G2_s0) point `weights` at the
        # exported graph, which the torch engine cannot load. The torch baseline
        # is the training-workspace checkpoint for the same run id.
        root = Path(os.environ.get("TINY_VCM_ROOT", ROOT.parent / "tiny-vcm")).resolve()
        ckpt = root / "checkpoints" / args.model / "best.pt"
        if not ckpt.exists():
            raise SystemExit(
                f"{args.model}: manifest ships ONNX ({w}); no torch checkpoint at {ckpt}.\n"
                f"Sync it from the training workspace, or bench the ONNX engine alone."
            )
        m.weights = str(ckpt)
    eng = TorchTinyVCMEngine(m, tiny_vcm_root=tiny_vcm_root())
    eng.load()
    print(f"{args.model}: arch={m.architecture.get('arch')} params={eng.param_count():,} "
          f"torch {torch.__version__} | default threads after load: {torch.get_num_threads()}")

    # features once, outside the timed loop
    if args.wav:
        import torchaudio
        wav, sr = torchaudio.load(args.wav)
        wave = wav.mean(0).numpy()
    else:
        sr = 16000
        t = np.arange(int(3.0 * sr)) / sr
        wave = (0.01 * np.random.default_rng(0).standard_normal(t.size)
                + 0.2 * np.sin(2 * np.pi * 220 * t) * (t > 0.5) * (t < 2.0)).astype(np.float32)
    with torch.no_grad():
        w = torch.from_numpy(wave).unsqueeze(0)
        w = eng._processor.trim_silence(eng._processor.to_mono_16k(w, sr))
        feat = eng._fit_frames(eng._processor.process(w), eng._n_frames, random_offset=False)
        feat = feat.transpose(1, 2).unsqueeze(0).contiguous()
    print(f"input {tuple(feat.shape)}\n")
    model = eng._model

    rows = []

    def run_torch(label, threads, flush, mode):
        torch.set_num_threads(threads)
        torch.set_flush_denormal(flush)
        ctx = torch.inference_mode if mode == "inference_mode" else torch.no_grad

        def f():
            with ctx():
                model(feat)
        p50, p95 = timeit(f, args.n)
        rows.append((label, threads, flush, mode, p50, p95))

    for threads in (1, 2, 4):
        run_torch("torch", threads, False, "no_grad")        # = the app today at threads=2
        run_torch("torch", threads, True, "no_grad")
    run_torch("torch", 2, True, "inference_mode")
    torch.set_flush_denormal(False)

    # denormal census: how many GRU activations are subnormal on this input?
    if hasattr(model, "gru"):
        cap = {}
        h = model.gru.register_forward_hook(lambda mod, i, o: cap.__setitem__("out", o[0]))
        with torch.no_grad():
            model(feat)
        h.remove()
        out = cap["out"].float().abs()
        tiny = ((out > 0) & (out < torch.finfo(torch.float32).tiny)).float().mean().item()
        print(f"GRU output subnormal fraction: {tiny:.2%}")

    print(f"{'runtime':8} {'thr':>3} {'flush_denormal':>14} {'mode':>15} {'p50 ms':>8} {'p95 ms':>8}")
    for r in rows:
        print(f"{r[0]:8} {r[1]:>3} {str(r[2]):>14} {r[3]:>15} {r[4]:8.2f} {r[5]:8.2f}")

    if args.onnx:
        try:
            import onnxruntime as ort
        except ImportError:
            print("\nonnxruntime not installed: pip install onnxruntime")
            return
        for threads in (1, 2):
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
            name = sess.get_inputs()[0].name
            x = feat.numpy()
            p50, p95 = timeit(lambda: sess.run(None, {name: x}), args.n)
            print(f"{'onnx':8} {threads:>3} {'-':>14} {'-':>15} {p50:8.2f} {p95:8.2f}")
        # parity with torch on this input
        with torch.no_grad():
            pt = torch.softmax(model(feat), -1).numpy()
        ox = sess.run(None, {name: feat.numpy()})[0]
        is_probs = bool((ox >= 0).all() and np.allclose(ox.sum(-1), 1.0, atol=1e-4))
        if not is_probs:  # exported graph returns logits
            ox = np.exp(ox - ox.max(-1, keepdims=True)); ox /= ox.sum(-1, keepdims=True)
        print(f"onnx vs torch max |dprob| = {np.abs(pt - ox).max():.2e}")


if __name__ == "__main__":
    main()

