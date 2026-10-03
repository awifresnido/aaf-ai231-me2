"""
VCM Main Training Script (Hardware-Aware)
Works on both Jetson Orin Nano and DGX A100

Usage:
    python train_vcm.py --phase 1 --device cuda
    python train_vcm.py --phase 2 --device cuda            # warm-starts from phase 1 best.pt
    # ablation overrides (see ARCH_REFACTOR_V2.md):
    python train_vcm.py --phase 1 --arch legacy --max-duration 4.0 --run-name A0_legacy
    python train_vcm.py --phase 1 --arch tcresnet --seed 1 --run-name A2_tc_s1

Revised 2026-09-25: model from src.vcm_models_v2.build_model (config
`model.arch`), label_mode intent|leaf, configurable class weighting,
warm_start actually implemented, data: block actually passed to the loader,
seeded runs, end-of-run macro-F1 / intent-level accuracy / confusion report.

ME2 Part 0 (2026-10): explicit --batch-size/--grad-accum/--amp/--no-amp that
override src/vcm_hardware.py, configurable --select-corpora, and a run-summary
JSON with the full environment/optimiser/timing block required by
ME2_MASTER_INSTRUCTIONS.md Part 0 step 5.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

from src.vcm_data.common import check_intent_contract
from src.vcm_data_loader import (
    compute_class_weights, create_dataloaders, create_label_mapping, decode_label,
)
from src.vcm_hardware import setup_hardware
from src.vcm_models_v2 import build_model, count_parameters
from src.vcm_trainer import Trainer

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
ONTOLOGY = "configs/ontology.json"


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    """Current HEAD commit sha (for the run summary). Never fails the run."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def git_dirty() -> bool:
    """True if the working tree has uncommitted tracked or untracked changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return bool(out)
    except Exception:
        return False


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def evaluate(model, loader, device, idx_to_key):
    """Accuracy, macro-F1 at class level and at intent level (slots collapsed)."""
    from sklearn.metrics import confusion_matrix, f1_score

    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        y_pred.extend(model(x.to(device)).argmax(1).cpu().tolist())
        y_true.extend(y.tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    intent_true = np.array([decode_label(idx_to_key[i])[0] for i in y_true])
    intent_pred = np.array([decode_label(idx_to_key[i])[0] for i in y_pred])
    present = sorted(set(y_true.tolist()))
    return {
        "n": int(len(y_true)),
        "acc": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "intent_acc": float((intent_true == intent_pred).mean()),
        "intent_macro_f1": float(f1_score(intent_true, intent_pred, average="macro", zero_division=0)),
        "labels": [idx_to_key[i] for i in present],
        "confusion": confusion_matrix(y_true, y_pred, labels=present).tolist(),
    }


def main():
    p = argparse.ArgumentParser(description="VCM Training Script (Hardware-Aware)")
    p.add_argument("--config", default="configs/train_config.yaml")
    p.add_argument("--phase", type=int, choices=[1, 2], default=1)
    p.add_argument("--device", choices=["cuda", "cpu"],
                   default="cuda" if torch.cuda.is_available() else "cpu")
    # ablation overrides -- None means "use the config"
    p.add_argument("--arch", choices=["tcresnet", "dscnn", "crnn_attn", "legacy"])
    p.add_argument("--width-mult", type=float)
    # crnn_attn options (ignored by other archs)
    p.add_argument("--gru-hidden", type=int)
    p.add_argument("--unidirectional", action="store_true", help="crnn_attn: uni-GRU instead of bi-GRU")
    p.add_argument("--pool", choices=["attention", "meanmax"], help="crnn_attn pooling")
    # throughput (GPU-utilisation) options -- do not change the model or the maths
    p.add_argument("--batch-size", type=int, help="override phase batch size (default keeps B2 parity: 64)")
    p.add_argument("--num-workers", type=int, help="DataLoader workers for THIS run")
    p.add_argument("--waveform-cache", default=None, help="dir built by scripts/build_waveform_cache.py")
    # Part 0 tooling: explicit hardware overrides (must win over src/vcm_hardware.py)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="gradient accumulation steps (overrides the hardware profile)")
    p.add_argument("--amp", dest="use_amp", action="store_true", default=None,
                   help="force mixed precision ON")
    p.add_argument("--no-amp", dest="use_amp", action="store_false",
                   help="force mixed precision OFF")
    p.add_argument("--gpu-features", action="store_true",
                   help="GPU augmentation + log-mel pipeline (workers return raw waveforms)")
    # Part 0 tooling: configurable validation corpora (default keeps v1 behaviour)
    p.add_argument("--select-corpora", nargs="*", default=None,
                   help="validation corpus filter (default: personal_awi + fsc for command_macro_f1)")
    p.add_argument("--crop-mode", choices=["start", "energy"], default="start",
                   help="crop window selection (GATE 2 result)")
    # Part 0 tooling: dataset provenance recorded in the summary JSON
    p.add_argument("--dataset-name", default=None, help="e.g. me2_master_v1 (else derived from manifest)")
    p.add_argument("--dataset-revision", default=None, help="HF dataset revision")
    p.add_argument("--max-duration", type=float)
    p.add_argument("--n-mels", type=int)
    p.add_argument("--label-mode", choices=["intent", "leaf"])
    p.add_argument("--class-weighting", choices=["none", "inverse", "sqrt_inverse"])
    p.add_argument("--no-trim", action="store_true", help="disable silence trimming")
    p.add_argument("--no-wave-aug", action="store_true", help="disable waveform augmentation (D1 ablation)")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None)
    p.add_argument("--init-checkpoint", default=None, help="override phase-2 warm start source")
    p.add_argument("--train-manifest", default=None, help="override the phase train manifest")
    p.add_argument("--val-manifest", default=None, help="override the phase val manifest")
    p.add_argument("--select-metric", choices=["val_loss", "command_macro_f1"], default=None,
                   help="checkpoint selection metric (default: val_loss)")
    args = p.parse_args()

    # Hard rule: real runs refuse a dirty git tree (reproducibility).
    if git_dirty():
        logger.error("Refusing to start: git working tree is dirty. Commit before launching.")
        return 1

    profiler, hardware_config = setup_hardware()
    config = load_config(args.config)
    phase_cfg = config["phase1"] if args.phase == 1 else config["phase2"]
    model_cfg = dict(config.get("model", {}))
    data_cfg = dict(config.get("data", {}))
    train_cfg = dict(config.get("training", {}))

    # dataset_revision is mandatory for the gold dataset (never run without it).
    _manifest = args.train_manifest or phase_cfg["train_manifest"]
    _dname = args.dataset_name or Path(_manifest).stem
    if _dname == "me2_gold_v1" and not args.dataset_revision:
        logger.error("--dataset-revision is mandatory for me2_gold_v1; pass it (HF revision).")
        return 1

    # ---- apply overrides ----
    if args.arch: model_cfg["arch"] = args.arch
    if args.width_mult: model_cfg["width_mult"] = args.width_mult
    if args.gru_hidden: model_cfg["gru_hidden"] = args.gru_hidden
    if args.unidirectional: model_cfg["bidirectional"] = False
    if args.pool: model_cfg["pool"] = args.pool
    if args.waveform_cache: data_cfg["waveform_cache"] = args.waveform_cache
    if args.max_duration: data_cfg["max_duration"] = args.max_duration
    if args.n_mels: data_cfg["n_mels"] = args.n_mels
    if args.no_trim: data_cfg["trim_db"] = None
    if args.no_wave_aug: data_cfg["wave_augmentation"] = None
    label_mode = args.label_mode or train_cfg.get("label_mode", "intent")
    weighting = args.class_weighting or train_cfg.get("class_weighting", "sqrt_inverse")
    epochs = args.epochs or phase_cfg["epochs"]
    lr = args.lr or float(phase_cfg["learning_rate"])
    n_mels = int(data_cfg.get("n_mels", 40))

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True   # fixed input size (3.0 s window) -> fastest kernels
    run_name = args.run_name or f"p{args.phase}_{model_cfg.get('arch', 'tcresnet')}_{label_mode}_s{args.seed}"
    save_dir = Path("checkpoints") / run_name
    device = torch.device(args.device)

    # ---- labels ----
    mapping = create_label_mapping(ONTOLOGY, label_mode)
    if label_mode == "intent":
        check_intent_contract(mapping, Path(ONTOLOGY))
    idx_to_key = {v: k for k, v in mapping.items()}
    command_indices = [i for k, i in mapping.items() if k not in ("UNKNOWN", "SILENCE")]
    select_metric = args.select_metric or "val_loss"
    logger.info(f"✓ {len(mapping)} classes (label_mode={label_mode}, select_metric={select_metric})")

    # ---- model ----
    model = build_model(model_cfg, n_classes=len(mapping), n_mels=n_mels).to(device)
    logger.info(f"✓ Model {model_cfg.get('arch', 'tcresnet')}: {count_parameters(model):,} params")

    init_ckpt_path = None
    init_ckpt_sha256 = None
    if args.phase == 2 and phase_cfg.get("warm_start", False):
        init = Path(args.init_checkpoint or phase_cfg.get("init_checkpoint", ""))
        if not init.is_file():
            raise FileNotFoundError(
                f"phase2.warm_start is true but init checkpoint {init} does not exist. "
                f"Run phase 1 first or pass --init-checkpoint."
            )
        state = torch.load(init, map_location=device)["model_state_dict"]
        model.load_state_dict(state)  # strict: arch/n_classes/n_mels must match phase 1
        init_ckpt_path = str(init.resolve())
        init_ckpt_sha256 = sha256_file(init)
        logger.info(f"✓ Warm start from {init} (sha256 {init_ckpt_sha256})")

    # ---- data ----
    batch_size = args.batch_size or hardware_config.get_memory_safe_batch_size(
        phase_cfg.get("batch_size", hardware_config.config["batch_size"]))
    num_workers = args.num_workers if args.num_workers is not None else hardware_config.get_num_workers(batch_size)
    grad_accum = args.grad_accum
    use_amp = args.use_amp if args.use_amp is not None else hardware_config.config["mixed_precision"]
    val_corpora = args.select_corpora if args.select_corpora is not None else (
        ["personal_awi", "fsc"] if select_metric == "command_macro_f1" else None)
    train_loader, val_loader = create_dataloaders(
        train_manifest=args.train_manifest or phase_cfg["train_manifest"],
        val_manifest=args.val_manifest or phase_cfg["val_manifest"],
        intent_mapping=mapping,
        audio_root=data_cfg.get("audio_root", "."),
        batch_size=batch_size,
        num_workers=num_workers,
        sample_rate=int(data_cfg.get("sample_rate", 16000)),
        train_split=phase_cfg.get("train_split", "train"),
        val_split=phase_cfg.get("val_split", "validation"),
        data_cfg=data_cfg,
        label_mode=label_mode,
        seed=args.seed,
        val_corpora=val_corpora,
        gpu_pipeline=args.gpu_features,
        crop_mode=args.crop_mode,
        device=str(device),
    )
    logger.info(f"✓ Train {len(train_loader.dataset)} | Val {len(val_loader.dataset)} | "
                f"batch {batch_size} | workers {num_workers} | max_duration {data_cfg.get('max_duration')} s | "
                f"waveform_cache {data_cfg.get('waveform_cache') or 'OFF'}")

    # sidecar: everything needed to rebuild this exact model + input pipeline
    # (read by src/vcm_infer.py so the app / evaluator never guess arch or window)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "model_cfg": model_cfg, "n_classes": len(mapping), "label_mode": label_mode,
        "n_mels": n_mels, "max_duration": float(data_cfg.get("max_duration", 2.5)),
        "sample_rate": int(data_cfg.get("sample_rate", 16000)),
        "hop_length": int(data_cfg.get("hop_length", 160)), "n_fft": int(data_cfg.get("n_fft", 512)),
        "f_min": int(data_cfg.get("f_min", 50)), "f_max": int(data_cfg.get("f_max", 7600)),
        "top_db": float(data_cfg.get("top_db", 80.0)), "trim_db": data_cfg.get("trim_db", 40.0),
        "params": count_parameters(model), "seed": args.seed, "run_name": run_name,
        "crop_mode": args.crop_mode,
    }
    (save_dir / "model_config.json").write_text(json.dumps(model_config, indent=2))
    class_weights = compute_class_weights(train_loader.dataset.samples, mapping, weighting)
    logger.info(f"✓ Class weighting: {weighting}")

    # ---- train ----
    start_time = time.time()
    start_iso = datetime.now().isoformat(timespec="seconds")
    trainer = Trainer(
        model=model,
        device=device,
        mixed_precision=use_amp,
        gradient_clip=float(model_cfg.get("gradient_clip", 1.0)),
        gradient_accumulation_steps=grad_accum,
        class_weights=class_weights.to(device),
        command_indices=command_indices,
        select_metric=select_metric,
    )
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        scheduler_type=phase_cfg["scheduler_type"],
        early_stopping_patience=phase_cfg["early_stopping_patience"],
        save_dir=str(save_dir),
        weight_decay=float(phase_cfg.get("weight_decay", 1e-5)),
    )
    end_iso = datetime.now().isoformat(timespec="seconds")
    wall_clock = round(time.time() - start_time, 3)
    trainer.save_history(Path("logs") / f"history_{run_name}.json")

    # ---- final evaluation on the BEST checkpoint ----
    trainer.load_checkpoint(save_dir / "best.pt")
    metrics = evaluate(model, val_loader, device, idx_to_key)

    train_manifest = args.train_manifest or phase_cfg["train_manifest"]
    dataset_name = args.dataset_name or Path(train_manifest).stem
    if torch.cuda.is_available():
        gpu_name = "jetson-orin-nano" if profiler.is_jetson() else torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
    else:
        gpu_name, gpu_count = "cpu", 0
    optimiser_info = None
    if getattr(trainer, "optimizer", None) is not None:
        pg = trainer.optimizer.param_groups[0]
        optimiser_info = {
            "class": type(trainer.optimizer).__name__,
            "lr": lr,
            "weight_decay": float(pg.get("weight_decay", 0.0)),
            "betas": [float(b) for b in pg.get("betas", (0.9, 0.999))],
            "eps": float(pg.get("eps", 1e-8)),
        }
    summary = {
        "run_name": run_name, "phase": args.phase, "seed": args.seed,
        "arch": model_cfg.get("arch", "tcresnet"), "params": count_parameters(model),
        "label_mode": label_mode, "class_weighting": weighting,
        "max_duration": data_cfg.get("max_duration"), "n_mels": n_mels,
        "trim_db": data_cfg.get("trim_db"), "epochs_run": len(history["val_acc"]),
        "model_cfg": model_cfg, "batch_size": batch_size, "num_workers": num_workers,
        "train_uncached_rows": getattr(train_loader.dataset, "uncached_rows", None),
        "val": metrics,
        # ---- Part 0 step 5: environment / optimiser / timing block ----
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "hostname": socket.gethostname(),
        "gpu": {"name": gpu_name, "count": gpu_count},
        "gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
        "dataset_name": dataset_name,
        "dataset_revision": args.dataset_revision,
        "manifest_sha256": sha256_file(train_manifest),
        "init_checkpoint": init_ckpt_path,
        "init_checkpoint_sha256": init_ckpt_sha256,
        "optimiser": optimiser_info,
        "effective_batch_size": batch_size,
        "grad_accumulation": grad_accum,
        "amp": use_amp,
        "crop_mode": args.crop_mode,
        "start_time": start_iso,
        "end_time": end_iso,
        "wall_clock_seconds": wall_clock,
        "global_steps": getattr(trainer, "global_steps", None),
        "best_epoch": getattr(trainer, "best_epoch", None),
        "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else None,
        "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else None,
    }
    out = Path("logs") / f"summary_{run_name}.json"
    out.write_text(json.dumps(summary, indent=2))
    logger.info(
        f"RESULT {run_name}: val acc {metrics['acc']:.4f} | macro-F1 {metrics['macro_f1']:.4f} | "
        f"intent acc {metrics['intent_acc']:.4f} -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
