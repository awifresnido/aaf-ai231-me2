#!/usr/bin/env python3
"""C4: export CRNN-Attn to ONNX, parity check, latency benchmark, deploy.json."""
import json, sys, time
from pathlib import Path
import torch, numpy as np

sys.path.insert(0, "src")
from src.vcm_infer import VCMInferencer
from src.vcm_data_loader import fit_frames

def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

RUN = sys.argv[1] if len(sys.argv) > 1 else "E1_s0"
CKPT = Path(f"checkpoints/{RUN}/best.pt")
OUT = Path("exports") / RUN
OUT.mkdir(parents=True, exist_ok=True)

inf = VCMInferencer(CKPT, device="cpu")
model = inf.model
n_frames, n_mels = inf.n_frames, 40
dummy = torch.zeros(1, 1, n_frames, n_mels)

# 1. export fp32 ONNX (opset 17, dynamic batch)
onnx_path = OUT / "model.onnx"
torch.onnx.export(model, dummy, str(onnx_path), opset_version=17,
                  input_names=["mel"], output_names=["logits"],
                  dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}})
print(f"exported {onnx_path}")

import onnxruntime as ort
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

# 2. fp32 parity
rng = np.random.default_rng(0); torch.manual_seed(0)
worst = 0.0
for _ in range(64):
    x = torch.randn(1, 1, n_frames, n_mels)
    with torch.no_grad():
        prob_pt = torch.softmax(model(x), -1)[0].numpy()
    logits = sess.run(None, {"mel": x.numpy().astype(np.float32)})[0]
    prob_on = softmax(logits[0])
    worst = max(worst, float(np.abs(prob_pt - prob_on).max()))
print(f"fp32 parity max|dprob| = {worst:.2e} (budget 1e-4) -> {'PASS' if worst <= 1e-4 else 'FAIL'}")

# 3. latency (single-thread, 200 runs)
opts = ort.SessionOptions(); opts.intra_op_num_threads = 1; opts.inter_op_num_threads = 1
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
for _ in range(200):
    t = time.perf_counter(); x = features(); ft.append(time.perf_counter() - t)
    t = time.perf_counter(); sess1.run(None, {"mel": x}); mt.append(time.perf_counter() - t)

def pct(a, p): return sorted(a)[int(p * len(a))] * 1000
print(f"features p50={pct(ft,0.5):.1f}ms p95={pct(ft,0.95):.1f}ms (budget 30ms)")
print(f"model    p50={pct(mt,0.5):.1f}ms p95={pct(mt,0.95):.1f}ms (budget 50ms)")

deploy = {
    "run": RUN, "arch": inf.arch, "params": sum(p.numel() for p in model.parameters()),
    "tau": 0.9, "onnx": "model.onnx",
    "n_mels": n_mels, "n_frames": n_frames, "sample_rate": inf.sample_rate,
    "label_mode": inf.label_mode, "num_classes": len(inf.mapping),
    "fp32_parity_max_dprob": worst,
    "latency_features_p95_ms": round(pct(ft, 0.95), 2),
    "latency_model_p95_ms": round(pct(mt, 0.95), 2),
}
(OUT / "deploy.json").write_text(json.dumps(deploy, indent=2))
print(json.dumps(deploy, indent=2))
