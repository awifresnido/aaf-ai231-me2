#!/usr/bin/env python3
"""Part 5 step 5: export winning me2 seeds to ONNX opset 17, run parity check,
and write deploy packages under exports/me2/<run>/.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from src.vcm_infer import VCMInferencer
from src.vcm_data_loader import fit_frames
EXPORTS = ROOT / "exports/me2"
EXPORTS.mkdir(parents=True, exist_ok=True)
DECISION = json.loads((ROOT / "results/me2/decision.json").read_text())

WINNERS = [
    ("B2m_s2", "tcresnet", 1.0, "sweep", "Sweep (TC-ResNet me2)"),
    ("E1m_s0", "crnn_attn", 1.0, "focus", "Focus (CRNN-Attn me2)"),
    ("G2m_s4", "dscnn", 1.0, "benchmark", "Benchmark (DS-CNN me2)"),
]

OPSET = 17


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


report = {}

for run, arch, wm, role, display in WINNERS:
    model_family = run.split("_")[0]
    tau = DECISION[model_family]["numbers"]["me2_tau"]
    ckpt = ROOT / f"checkpoints/{run}/best.pt"
    out = EXPORTS / run
    out.mkdir(parents=True, exist_ok=True)

    inf = VCMInferencer(ckpt, device="cpu", arch=arch, width_mult=wm)
    model = inf.model.eval()
    n_frames, n_mels = inf.n_frames, 40
    dummy = torch.zeros(1, 1, n_frames, n_mels)
    onnx_path = out / "model.onnx"

    torch.onnx.export(
        model, dummy, str(onnx_path), opset_version=OPSET,
        input_names=["mel"], output_names=["logits"],
        dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False, external_data=False,
    )

    import onnx
    proto = onnx.load(str(onnx_path))
    opsets = {i.domain or "ai.onnx": int(i.version) for i in proto.opset_import}
    got_opset = opsets.get("ai.onnx")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # fp32 parity (64 random tensors)
    torch.manual_seed(0)
    worst = 0.0
    for _ in range(64):
        x = torch.randn(1, 1, n_frames, n_mels)
        with torch.no_grad():
            prob_pt = torch.softmax(model(x), -1)[0].numpy()
        logits = sess.run(None, {"mel": x.numpy().astype(np.float32)})[0]
        worst = max(worst, float(np.abs(prob_pt - softmax(logits[0])).max()))

    # latency (single-thread)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess1 = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    torch.set_num_threads(1)
    wav = torch.randn(1, 48000)

    def features():
        w = inf.processor.to_mono_16k(wav, 16000)
        w = inf.processor.trim_silence(w)
        f = inf.processor.process(w)
        f = fit_frames(f, n_frames, random_offset=False)
        return f.transpose(1, 2).unsqueeze(0).numpy().astype(np.float32)

    x = features()
    for _ in range(10):
        features(); sess1.run(None, {"mel": x})
    ft, mt = [], []
    for _ in range(100):
        t = time.perf_counter(); x = features(); ft.append(time.perf_counter() - t)
        t = time.perf_counter(); sess1.run(None, {"mel": x}); mt.append(time.perf_counter() - t)

    def pct(a, p):
        return sorted(a)[int(p * len(a))] * 1000

    idx_to_key = {v: k for k, v in inf.mapping.items()}
    labels = [idx_to_key[i] for i in range(len(inf.mapping))]
    (out / "labels.json").write_text(json.dumps(labels, indent=2))

    sidecar = json.loads(ckpt.with_name("model_config.json").read_text())
    deploy = {
        "run": run,
        "arch": inf.arch,
        "params": sum(p.numel() for p in model.parameters()),
        "tau": tau,
        "crop_mode": "start",
        "onnx": "model.onnx",
        "labels": "labels.json",
        "model_config": sidecar["model_cfg"],
        "n_mels": n_mels,
        "n_frames": n_frames,
        "sample_rate": inf.sample_rate,
        "label_mode": inf.label_mode,
        "num_classes": len(inf.mapping),
        "opset": OPSET,
        "opset_exported": got_opset,
        "fp32_parity_max_dprob": worst,
        "fp32_onnx_kb": round(onnx_path.stat().st_size / 1024, 1),
        "latency_model_p50_ms": round(pct(mt, 0.5), 2),
        "latency_model_p95_ms": round(pct(mt, 0.95), 2),
        "latency_features_p50_ms": round(pct(ft, 0.5), 2),
        "latency_features_p95_ms": round(pct(ft, 0.95), 2),
        "role": role,
        "display_name": display,
    }
    (out / "deploy.json").write_text(json.dumps(deploy, indent=2))
    report[run] = {
        "arch": inf.arch,
        "params": deploy["params"],
        "onnx_kb": deploy["fp32_onnx_kb"],
        "opset": got_opset,
        "parity_max_dprob": worst,
        "parity_pass": bool(worst <= 1e-4),
        "latency_model_p95_ms": deploy["latency_model_p95_ms"],
        "deploy_path": str(out / "deploy.json"),
    }
    print(f"Exported {run}: {deploy['params']:,} params, {deploy['fp32_onnx_kb']} KB, "
          f"parity={worst:.2e} ({'PASS' if worst<=1e-4 else 'FAIL'}), "
          f"model p95={deploy['latency_model_p95_ms']}ms")

(ROOT / "results/me2/onnx_parity_report.json").write_text(json.dumps(report, indent=2))
print("ALL EXPORTS COMPLETE")
