"""Package a tiny-vcm checkpoint as a model the demo app can load.

    python scripts/register_model.py \
        --checkpoint ../tiny-vcm/checkpoints/B2_s1/best.pt \
        --id B2_s1 --name "Phase 2 TC-ResNet, seed 1" \
        --phase phase2 --data "composite_v1 + wave aug" --min-confidence 0.80 --no-copy

``architecture`` is pruned to the keys the run's own arch reads (DS-CNN gets channels /
n_blocks, not the TC-ResNet width_mult / kernel that the copied config also carries).

Reads architecture, features and label mode from tiny-vcm's train_config.yaml
(or the config stored in the checkpoint, if any), derives the class list with
tiny-vcm's own create_label_mapping, copies (or references) the weights,
writes manifest.yaml, then loads the model once to prove manifest and weights agree.

If ``model_config.json`` sits next to the checkpoint (written by train_vcm.py for
every run from the CRNN-Attn work onward), it is the single source of truth: the
whole ``model_cfg`` dict becomes ``architecture`` (including the crnn keys), and
``label_mode`` / the mel front-end come from it too.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vcm_common.arch_keys import prune_architecture  # noqa: E402


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tiny-vcm", default=str(REPO_ROOT.parent / "tiny-vcm"), type=Path)
    ap.add_argument("--train-config", type=Path, default=None,
                    help="default: <tiny-vcm>/configs/train_config.yaml")
    ap.add_argument("--label-mode", choices=["leaf", "intent"], default=None,
                    help="override the label mode read from the config")
    ap.add_argument("--phase", default=None)
    ap.add_argument("--data", default=None, help="training data description")
    ap.add_argument("--mlflow-run", default=None)
    ap.add_argument("--min-confidence", type=float, default=0.70)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-copy", action="store_true",
                    help="reference the checkpoint in place (dev) instead of copying it")
    args = ap.parse_args()

    tv = args.tiny_vcm.resolve()
    sys.path.insert(0, str(tv))
    import torch
    from src.vcm_data_loader import create_label_mapping  # type: ignore

    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)

    sidecar = args.checkpoint.with_name("model_config.json")
    if sidecar.is_file():
        sc = json.loads(sidecar.read_text(encoding="utf-8"))
        model_cfg = dict(sc["model_cfg"])          # whole dict, including crnn keys
        label_mode = sc["label_mode"]
        features = {
            "sample_rate": int(sc["sample_rate"]),
            "n_mels": int(sc["n_mels"]),
            "n_fft": int(sc["n_fft"]),
            "hop_length": int(sc["hop_length"]),
            "f_min": int(sc["f_min"]),
            "f_max": int(sc["f_max"]),
            "top_db": float(sc["top_db"]),
            "trim_db": sc["trim_db"],
            "max_duration_s": float(sc["max_duration"]),
        }
        print("using sidecar model_config.json")
    else:
        cfg_path = args.train_config or tv / "configs" / "train_config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict) \
                and "model" in ckpt["config"]:
            cfg = ckpt["config"]  # trust the run's own config over today's YAML
            print("using config stored in checkpoint")
        model_cfg, data_cfg = cfg["model"], cfg["data"]
        label_mode = args.label_mode or cfg.get("training", {}).get("label_mode", "leaf")
        features = {
            "sample_rate": int(data_cfg.get("sample_rate", 16000)),
            "n_mels": int(data_cfg.get("n_mels", 40)),
            "n_fft": int(data_cfg.get("n_fft", 512)),
            "hop_length": int(data_cfg.get("hop_length", 160)),
            "f_min": int(data_cfg.get("f_min", 50)),
            "f_max": int(data_cfg.get("f_max", 7600)),
            "top_db": float(data_cfg.get("top_db", 80.0)),
            "trim_db": data_cfg.get("trim_db", 40.0),
            "max_duration_s": float(data_cfg.get("max_duration", 3.0)),
        }

    # the model block of a train_config.yaml carries every arch's keys; keep only
    # the ones this architecture actually reads (the E1 manifest kept width_mult/kernel)
    model_cfg, dropped = prune_architecture(model_cfg)
    if dropped:
        print(f"architecture: dropped keys not used by arch={model_cfg.get('arch')!r}: {dropped}")

    mapping = create_label_mapping(str(tv / "configs" / "ontology.json"), label_mode)
    labels = [k for k, _ in sorted(mapping.items(), key=lambda kv: kv[1])]

    out = REPO_ROOT / "models" / "vcm" / args.id
    if out.exists() and not args.force:
        sys.exit(f"{out} exists (use --force to overwrite)")
    out.mkdir(parents=True, exist_ok=True)
    if args.no_copy:
        weights = os.path.relpath(args.checkpoint.resolve(), out.resolve())
    else:
        shutil.copy2(args.checkpoint, out / "model.pt")
        weights = "model.pt"

    metrics = {k: ckpt[k] for k in ("epoch", "val_acc", "best_val_acc", "val_loss")
               if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], (int, float))}
    manifest = {
        "id": args.id,
        "display_name": args.name,
        "task": "vcm",
        "engine": "torch_tinyvcm",
        "weights": weights,
        "architecture": dict(model_cfg),
        "label_scheme": label_mode,
        "labels": labels,
        "features": features,
        "reject": {"min_confidence": args.min_confidence,
                   "non_command_labels": ["UNKNOWN", "SILENCE"]},
        "quantization": "fp32",
        "lineage": {"training_phase": args.phase, "training_data": args.data,
                    "git_commit": git_commit(tv), "mlflow_run_id": args.mlflow_run,
                    "notes": f"registered from {args.checkpoint}"},
        "offline_metrics": metrics,
        "engine_options": {"num_threads": 2},
    }
    (out / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False,
                                                      allow_unicode=True), encoding="utf-8")

    from edge.registry import ModelRegistry

    reg = ModelRegistry(REPO_ROOT / "models", {"tiny_vcm_root": str(tv)})
    reg.scan()
    engine = reg.get(args.id)
    print(f"registered {args.id}: {len(labels)} classes, {engine.param_count():,} params -> {out}")


if __name__ == "__main__":
    main()
