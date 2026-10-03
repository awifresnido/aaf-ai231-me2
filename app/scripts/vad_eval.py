#!/usr/bin/env python3
"""Evaluate VAD endpointing on real recordings before enabling it in the demo.

Replays each WAV in 20 ms frames (exactly like the live microphone stream)
through the CaptureEndpointer, for several hangover values, and reports:

  * endpoint reason mix   (end_of_speech / no_speech / max_window)
  * stop time             (ms after capture start) and time saved vs the fixed window
  * truncation rate       capture closed before the speech actually ended
                          (reference speech end = energy trim at -40 dB re. clip peak and
                          >= 10 dB above the noise floor; same idea tiny-vcm uses to trim)
  * optional VCM check    --tiny-vcm/--checkpoint: predict on the FULL clip and on the
                          ENDPOINTED clip; report agreement and accuracy vs the label

Inputs: a class-format folder (manifest.csv + WAVs, e.g. AI231/202453069 or a
classmate folder) or any folder of WAVs (no labels -> no accuracy).

    python scripts/vad_eval.py --folder ../202453069 --hangovers 300,500,700
    python scripts/vad_eval.py --folder ../202520785 --tiny-vcm ../tiny-vcm \
        --checkpoint ../tiny-vcm/checkpoints/B2_s0/best.pt --arch tcresnet

Pre-registered choice rule (docs/VAD_PLUGIN_GUIDE.md §5): the smallest hangover with
truncation <= 1 % and VCM accuracy within 1 pt of the full-window clip.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from edge.vad import EndpointingConfig, build_endpointer  # noqa: E402

SR = 16000
FRAME = 320


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError(f"{path}: only 16-bit PCM supported")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        n = int(round(x.size * SR / sr))
        x = np.interp(np.arange(n) / SR, np.arange(x.size) / sr, x).astype(np.float32)
    return x


def ref_speech_end_ms(x: np.ndarray, trim_db: float = 40.0) -> float | None:
    frame, hop = int(0.025 * SR), int(0.0125 * SR)
    if x.size <= frame:
        return None
    fr = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]
    db = 10 * np.log10(np.mean(fr * fr, axis=1) + 1e-10)
    # speech must be within trim_db of the peak AND clearly above the noise floor
    # (10th-percentile frame level + 10 dB); otherwise room noise near the
    # -40 dB line would count as speech to the end of the clip.
    floor = np.percentile(db, 10)
    act = np.nonzero(db > max(db.max() - trim_db, floor + 10.0))[0]
    return None if act.size == 0 else 1000.0 * (act[-1] * hop + frame) / SR


def load_items(folder: Path) -> list[dict]:
    man = folder / "manifest.csv"
    if man.is_file():
        rows = list(csv.DictReader(open(man, newline="", encoding="utf-8")))
        items = []
        for r in rows:
            label = r.get("label", "")
            slot = (r.get("slot_value") or "").strip()
            if label in ("UNKNOWN", "SILENCE") or not label:
                continue
            items.append({"path": folder / r["filename"],
                          "label": f"{label}|{slot}" if slot else label})
        return items
    return [{"path": p, "label": None} for p in sorted(folder.glob("*.wav"))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True, action="append", help="repeatable")
    ap.add_argument("--hangovers", default="300,500,700")
    ap.add_argument("--engine", default="silero", choices=["silero", "energy"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--window-ms", type=float, default=4000.0, help="the fixed window being replaced")
    ap.add_argument("--tiny-vcm", default=None, help="path to tiny-vcm (enables the VCM check)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--arch", default="tcresnet")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runtime/vad_eval.json")
    args = ap.parse_args()

    items = [it for f in args.folder for it in load_items(Path(f))]
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} clips")

    vcm = None
    if args.tiny_vcm and args.checkpoint:
        import os
        tvcm = Path(args.tiny_vcm).resolve()
        ckpt = Path(args.checkpoint).resolve()
        sys.path.insert(0, str(tvcm))
        import torch  # noqa: F401
        from src.vcm_infer import VCMInferencer
        prev = os.getcwd()
        os.chdir(tvcm)            # VCMInferencer reads configs/ontology.json relative to cwd
        try:
            vcm = VCMInferencer(str(ckpt), arch=args.arch)   # model_config.json sidecar wins if present
        finally:
            os.chdir(prev)

    def predict(x: np.ndarray) -> str:
        import torch
        return vcm.predict_wav(torch.from_numpy(x).unsqueeze(0))[0]

    full_pred = {}
    if vcm:
        for it in items:
            full_pred[str(it["path"])] = predict(read_wav(it["path"]))

    report = {"n_clips": len(items), "engine": args.engine, "results": []}
    for hang in [float(h) for h in args.hangovers.split(",")]:
        cfg = EndpointingConfig.from_dict({"enabled": True, "engine": args.engine,
                                           "model_path": args.model_path,
                                           "search_roots": ["../openwakeword"],
                                           "hangover_ms": hang, "max_window_ms": args.window_ms})
        ce, status = build_endpointer(cfg)
        reasons, stops, trunc, agree, acc_full, acc_ep, n_lab = {}, [], 0, 0, 0, 0, 0
        vad_ms = []
        for it in items:
            x = read_wav(it["path"])[: int(args.window_ms * SR / 1000)]
            ce.start()
            n_used = x.size
            for i in range(0, x.size, FRAME):
                ce.push(x[i:i + FRAME])
                if ce.done:
                    n_used = i + FRAME
                    break
            r = ce.result()
            reasons[r.reason or "audio_ended"] = reasons.get(r.reason or "audio_ended", 0) + 1
            stop_ms = 1000.0 * n_used / SR
            stops.append(stop_ms)
            ref = ref_speech_end_ms(x)
            if r.reason != "no_speech" and ref is not None and stop_ms < ref - 50:
                trunc += 1
            s = ce.summary()
            if s["vad_ms_mean"] is not None:
                vad_ms.append(s["vad_ms_mean"])
            if vcm and r.reason != "no_speech":
                p_ep = predict(x[:n_used])
                p_full = full_pred[str(it["path"])]
                agree += int(p_ep == p_full)
                if it["label"]:
                    n_lab += 1
                    acc_full += int(p_full == it["label"])
                    acc_ep += int(p_ep == it["label"])
        n = len(items)
        res = {
            "hangover_ms": hang, "status": status, "reasons": reasons,
            "stop_ms_median": float(np.median(stops)), "stop_ms_p90": float(np.percentile(stops, 90)),
            "saved_ms_median": float(args.window_ms - np.median(stops)),
            "truncation_rate": trunc / n if n else None,
            "vad_ms_per_chunk_mean": float(np.mean(vad_ms)) if vad_ms else None,
        }
        if vcm:
            spoken = n - reasons.get("no_speech", 0)
            res.update({"vcm_agreement_full_vs_endpointed": agree / spoken if spoken else None,
                        "vcm_acc_full": acc_full / n_lab if n_lab else None,
                        "vcm_acc_endpointed": acc_ep / n_lab if n_lab else None, "n_labelled": n_lab})
        report["results"].append(res)
        print(json.dumps(res))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
