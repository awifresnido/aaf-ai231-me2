#!/usr/bin/env python
"""Export a registered VCM model to ONNX (embedded weights, static batch=1).

    python scripts/export_onnx.py --model B2_s0
    python scripts/export_onnx.py --model E1_s0

Reads models/vcm/<id>/manifest.yaml, builds the same architecture the torch
engine would, loads the checkpoint at <tiny-vcm>/checkpoints/<id>/best.pt, and
writes into <tiny-vcm>/exports/<id>/:

    model.onnx    ONNX graph with weights embedded (input "feat" (1,1,T,40))
    deploy.json   arch / params / tau / label info + fp32 parity
    labels.json   the label list (leaf or intent order)

Notes:
  * batch is fixed at 1 and time at n_frames (3.0 s) -- exactly what the edge
    feeds -- which also sidesteps the GRU variable-length export warning.
  * torch.onnx.export defaults to external_data=True from torch>=2.9; we pass
    external_data=False so the weights stay inside the .onnx file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vcm_common.manifest import ModelManifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model id under models/vcm/")
    ap.add_argument("--tiny-vcm", default=str(ROOT.parent / "tiny-vcm"), type=Path)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import torch

    tv = args.tiny_vcm.resolve()
    sys.path.insert(0, str(tv))
    from src.vcm_models_v2 import build_model  # type: ignore

    m = ModelManifest.load(ROOT / "models" / "vcm" / args.model)
    n_classes = len(m.labels)
    n_mels = m.features.n_mels
    n_frames = int(m.features.max_duration_s * m.features.sample_rate / m.features.hop_length) + 1

    checkpoint = tv / "checkpoints" / args.model / "best.pt"
    if not checkpoint.exists():
        sys.exit(f"checkpoint not found: {checkpoint}")

    model = build_model(m.architecture, n_classes=n_classes, n_mels=n_mels)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    params = sum(p.numel() for p in model.parameters())

    out_dir = tv / "exports" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    dummy = torch.randn(1, 1, n_frames, n_mels)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["feat"], output_names=["logits"],
        opset_version=args.opset, dynamo=False, external_data=False,
    )

    # fp32 parity against torch on the same input
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = dummy.numpy()
    with torch.no_grad():
        pt = torch.softmax(model(dummy), -1).numpy()
    ox = sess.run(None, {"feat": x})[0]
    ox = np.exp(ox - ox.max(-1, keepdims=True)); ox /= ox.sum(-1, keepdims=True)
    parity = float(np.abs(pt - ox).max())

    deploy = {
        "run": m.id,
        "arch": m.architecture.get("arch"),
        "params": params,
        "tau": m.reject.min_confidence,
        "onnx": "model.onnx",
        "n_mels": n_mels,
        "n_frames": n_frames,
        "sample_rate": m.features.sample_rate,
        "label_mode": m.label_scheme,
        "num_classes": n_classes,
        "fp32_parity_max_dprob": parity,
    }
    (out_dir / "deploy.json").write_text(json.dumps(deploy, indent=2) + "\n", encoding="utf-8")
    (out_dir / "labels.json").write_text(json.dumps(m.labels, indent=2) + "\n", encoding="utf-8")

    print(f"{m.id}: {params:,} params, {n_classes} classes, arch={m.architecture.get('arch')}, "
          f"onnx={onnx_path} ({onnx_path.stat().st_size} bytes), fp32 parity {parity:.2e}")


if __name__ == "__main__":
    main()
