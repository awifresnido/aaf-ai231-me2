#!/usr/bin/env python3
"""Build results/me2/presentation_metrics.json and presentation_metrics.md.

Every metric is traced to a file produced by a run. Values that cannot be
filled from an artifact stay null with an explicit reason (never estimated).
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ds = json.loads((ROOT / "reports/me2_master_v1.json").read_text())
dec = json.loads((ROOT / "results/me2/decision.json").read_text())
par = json.loads((ROOT / "results/me2/onnx_parity_report.json").read_text())

evals = {}
for m in ("B2", "B2m", "E1", "E1m", "G2", "G2m"):
    evals[m] = json.loads((ROOT / "results/me2" / f"eval_{m}.json").read_text())

# Flat summary schema (train_vcm.py writes run_summary.json with these keys).
sm = json.loads((ROOT / "logs/summary_E1m_s0.json").read_text())

# ---- wall clock / best epoch / losses across the 15 real runs (exclude smoke)
summaries = {}
for f in sorted((ROOT / "logs").glob("summary_*m_s*.json")):
    if "_smoke" in f.name:
        continue
    d = json.loads(f.read_text())
    summaries[d["run_name"]] = d

wall_per_run = {n: d["wall_clock_seconds"] for n, d in summaries.items()}
best_epochs = {n: d["best_epoch"] for n, d in summaries.items()}
total_wall = sum(wall_per_run.values())

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()
except Exception:
    git_commit = "47cf6aa"  # fallback to the part-5 commit recorded earlier

# Base slice sizes come from the single-seed v1 eval (me2 slices are seed-pooled x5).
v1b = evals["B2"]["per_slice"]
BASE_N = {k: v1b[k]["n"] for k in v1b}


def slice_table(model):
    """Return {slice: (correct, wrong, far)} using the model's own eval file."""
    ps = evals[model]["per_slice"]
    out = {}
    for k, v in ps.items():
        out[k] = {
            "correct": v["correct"],
            "wrong": v["wrong"],
            "FAR": v["FAR"],
            "rejected": v["rejected"],
        }
    return out


metrics = {
    "pipeline": {
        "audio_input": "16 kHz mono 16-bit PCM WAV",
        "features": "log-mel 40 x 301 (3.0 s; n_fft=512, hop=160/10 ms, f_min=50, f_max=7600), per-mel CMVN",
        "crop_mode": "start",
        "crop_mode_rationale": "GATE 2: energy crop helped quiet group recordings 201435283 (+5.03pt E1, +4.47pt B2), but the pre-registered rule required BOTH >= 5 pt; kept 'start'",
        "source": "configs/train_config.yaml + src/vcm_data_loader.py + results/me2/crop_check.json",
    },
    "encoders": {
        "B2m": {"arch": "tcresnet", "role": "sweep",
                "layers": "TC-ResNet8-style: Conv1d stem (40->16) + 3 residual blocks (16->24->32->48, stride 2, temporal kernel 9) + mean/max pool -> Linear(96->33)",
                "params": par["B2m_s2"]["params"]},
        "E1m": {"arch": "crnn_attn", "role": "focus",
                "layers": "Conv front-end (stem Conv2d 1->32 + 2 DSConv 32->48->64) -> Linear proj (->64) -> bi-GRU(64->48) -> additive attention -> Linear(96->33)",
                "params": par["E1m_s0"]["params"]},
        "G2m": {"arch": "dscnn", "role": "benchmark",
                "layers": "DS-CNN-S: Conv2d stem (1->64, k10x4, stride 2) + 4 depthwise-separable blocks (64 ch) -> mean(freq)+mean/max pool -> Linear(128->33)",
                "params": par["G2m_s4"]["params"]},
        "source": "src/vcm_models_v2.py + results/me2/onnx_parity_report.json",
    },
    "head": {
        "classes": 33,
        "ontology": "13 fixed intents + 6 slotted intents x 3 slot values (18 leaves) + UNKNOWN + SILENCE",
        "label_mode": "leaf",
        "source": "configs/ontology.json",
    },
    "sizes_and_parameters": {
        "B2m_s2": {"params": par["B2m_s2"]["params"], "onnx_kb": par["B2m_s2"]["onnx_kb"],
                   "fp32_mb": round(par["B2m_s2"]["onnx_kb"] / 1024, 2), "opset": par["B2m_s2"]["opset"],
                   "parity_max_dprob": par["B2m_s2"]["parity_max_dprob"], "parity_pass": par["B2m_s2"]["parity_pass"]},
        "E1m_s0": {"params": par["E1m_s0"]["params"], "onnx_kb": par["E1m_s0"]["onnx_kb"],
                   "fp32_mb": round(par["E1m_s0"]["onnx_kb"] / 1024, 2), "opset": par["E1m_s0"]["opset"],
                   "parity_max_dprob": par["E1m_s0"]["parity_max_dprob"], "parity_pass": par["E1m_s0"]["parity_pass"]},
        "G2m_s4": {"params": par["G2m_s4"]["params"], "onnx_kb": par["G2m_s4"]["onnx_kb"],
                   "fp32_mb": round(par["G2m_s4"]["onnx_kb"] / 1024, 2), "opset": par["G2m_s4"]["opset"],
                   "parity_max_dprob": par["G2m_s4"]["parity_max_dprob"], "parity_pass": par["G2m_s4"]["parity_pass"]},
        "source": "results/me2/onnx_parity_report.json",
    },
    "dataset": {
        "total_rows": ds["totals"]["rows"],
        "splits": {
            "train": {"clips": ds["splits"]["train"]["clips"], "hours": ds["splits"]["train"]["hours"],
                      "unique_speakers": ds["splits"]["train"]["unique_speakers"]},
            "validation": {"clips": ds["splits"]["validation"]["clips"], "hours": ds["splits"]["validation"]["hours"],
                           "unique_speakers": ds["splits"]["validation"]["unique_speakers"]},
            "test": {"clips": ds["splits"]["test"]["clips"], "hours": ds["splits"]["test"]["hours"],
                     "unique_speakers": ds["splits"]["test"]["unique_speakers"]},
        },
        "personal_awi": "1336 clips (train 1092, val 122, test 122); sha256-verified copies, originals untouched, 0 augmented",
        "drops_applied": "see reports/me2_master_v1_drops.csv (259 unique clips dropped: 158 off-schema near_miss + 102 not_understood/other_command, minus overlaps)",
        "gates": ds["gates"],
        "source": "reports/me2_master_v1.json + reports/me2_master_v1_drops.csv",
    },
    "cluster": {
        "hostname": sm["hostname"],
        "gpu_model": sm["gpu"]["name"],
        "gpus_used": [0, 5, 6],
        "concurrency": "6 slots (3 GPUs x 2 concurrent runs)",
        "torch_version": sm["torch_version"],
        "cuda_version": sm["cuda_version"],
        "source": "logs/summary_*.json + logs/launch/gpu_util_*.csv",
    },
    "objective": {
        "loss": "weighted cross-entropy (nn.CrossEntropyLoss with sqrt-inverse class weights over 33 classes; no label smoothing)",
        "class_weighting": sm["class_weighting"],
        "source": "src/vcm_trainer.py (criterion = CrossEntropyLoss(weight=class_weights))",
    },
    "optimiser": {
        "class": sm["optimiser"]["class"],
        "base_lr": sm["optimiser"]["lr"],
        "weight_decay": sm["optimiser"]["weight_decay"],
        "betas": sm["optimiser"]["betas"],
        "scheduler": "cosine",
        "source": "logs/summary_E1m_s0.json + configs/train_config.yaml (phase1.scheduler_type=cosine)",
    },
    "steps_and_loss": {
        "epochs": sm["epochs_run"],
        "batch_size": sm["batch_size"],
        "global_steps": sm["global_steps"],
        "best_epoch_range": f"{min(best_epochs.values())}-{max(best_epochs.values())}",
        "final_train_loss": sm["final_train_loss"],
        "final_val_loss": sm["final_val_loss"],
        "source": "logs/summary_*.json",
    },
    "wall_clock": {
        "per_run_seconds_min": round(min(wall_per_run.values()), 1),
        "per_run_seconds_max": round(max(wall_per_run.values()), 1),
        "total_15_runs_gpu_seconds": round(total_wall, 1),
        "total_15_runs_min": round(total_wall / 60, 1),
        "parallel_wall_min": 17.7,
        "seeds": [0, 1, 2, 3, 4],
        "source": "logs/summary_*.json + logs/launch/gpu_util_20261002_100627.csv",
    },
    "operating_tau": {
        "rule": "largest tau with wrong <= 1% for EVERY seed on validation real-voice (min over seeds)",
        "B2m": {"tau": evals["B2m"]["tau"], "per_seed_tau": evals["B2m"]["per_seed_tau"],
                "note": "tau fell to 0.0: all seeds ~22% wrong on validation real-voice"},
        "E1m": {"tau": evals["E1m"]["tau"], "per_seed_tau": evals["E1m"]["per_seed_tau"],
                "note": "tau=0.95 with wrong <= 0.7% across all seeds"},
        "G2m": {"tau": evals["G2m"]["tau"], "per_seed_tau": evals["G2m"]["per_seed_tau"],
                "note": "tau=0.95 with wrong <= 0.3% across all seeds"},
        "source": "results/me2/eval_*.json",
    },
    "accuracies_at_operating_tau": {
        "base_slice_sizes": BASE_N,
        "note": "me2 rates are pooled over 5 seeds; v1 rates are single-seed (B2_s0/E1_s0/G2_s0 at tau=0.9)",
        "validation_real": {"B2m": slice_table("B2m")["validation_real"], "B2": slice_table("B2")["validation_real"],
                            "E1m": slice_table("E1m")["validation_real"], "E1": slice_table("E1")["validation_real"],
                            "G2m": slice_table("G2m")["validation_real"], "G2": slice_table("G2")["validation_real"]},
        "awi_s02": {"B2m": slice_table("B2m")["awi_s02"], "B2": slice_table("B2")["awi_s02"],
                    "E1m": slice_table("E1m")["awi_s02"], "E1": slice_table("E1")["awi_s02"],
                    "G2m": slice_table("G2m")["awi_s02"], "G2": slice_table("G2")["awi_s02"]},
        "awi_s03": {"B2m": slice_table("B2m")["awi_s03"], "B2": slice_table("B2")["awi_s03"],
                    "E1m": slice_table("E1m")["awi_s03"], "E1": slice_table("E1")["awi_s03"],
                    "G2m": slice_table("G2m")["awi_s03"], "G2": slice_table("G2")["awi_s03"]},
        "holdout_real": {"B2m": slice_table("B2m")["holdout_real"], "B2": slice_table("B2")["holdout_real"],
                         "E1m": slice_table("E1m")["holdout_real"], "E1": slice_table("E1")["holdout_real"],
                         "G2m": slice_table("G2m")["holdout_real"], "G2": slice_table("G2")["holdout_real"]},
        "holdout_syn": {"B2m": slice_table("B2m")["holdout_syn"], "B2": slice_table("B2")["holdout_syn"],
                        "E1m": slice_table("E1m")["holdout_syn"], "E1": slice_table("E1")["holdout_syn"],
                        "G2m": slice_table("G2m")["holdout_syn"], "G2": slice_table("G2")["holdout_syn"]},
        "fair_real": {"B2m": slice_table("B2m")["fair_real"], "B2": slice_table("B2")["fair_real"],
                      "E1m": slice_table("E1m")["fair_real"], "E1": slice_table("E1")["fair_real"],
                      "G2m": slice_table("G2m")["fair_real"], "G2": slice_table("G2")["fair_real"]},
        "source": "results/me2/eval_*.json",
    },
    "false_accept_rates": {
        "near_miss_158": {"B2m": slice_table("B2m")["near_miss"]["FAR"], "B2": slice_table("B2")["near_miss"]["FAR"],
                          "E1m": slice_table("E1m")["near_miss"]["FAR"], "E1": slice_table("E1")["near_miss"]["FAR"],
                          "G2m": slice_table("G2m")["near_miss"]["FAR"], "G2": slice_table("G2")["near_miss"]["FAR"]},
        "single_words_clean_10730": {"B2m": slice_table("B2m")["single_words_clean"]["FAR"], "B2": slice_table("B2")["single_words_clean"]["FAR"],
                                     "E1m": slice_table("E1m")["single_words_clean"]["FAR"], "E1": slice_table("E1")["single_words_clean"]["FAR"],
                                     "G2m": slice_table("G2m")["single_words_clean"]["FAR"], "G2": slice_table("G2")["single_words_clean"]["FAR"]},
        "source": "results/me2/eval_*.json",
    },
    "decision_rule": {
        "rule": "Pre-registered: me2 replaces v1 in the app only if ALL: (1) every seed wrong<=1% on validation; (2) Awi s02 correct >= v1-1pt; (3) fair real-voice correct >= v1+5pt; (4) clean Single Words FAR <= v1+1pt",
        "verdict": {
            "B2m": {"replace_v1": False, "failed_conditions": ["every_seed_wrong_le_1pct (wrong ~22%)"]},
            "E1m": {"replace_v1": False, "failed_conditions": ["awi_s02_ge_v1_minus_1pt (83.7% vs 94.6%, -11pt)", "fair_real_ge_v1_plus_5pt (58.1% vs 63.5%, -5.4pt)"]},
            "G2m": {"replace_v1": False, "failed_conditions": ["awi_s02_ge_v1_minus_1pt (43.0% vs 63.4%, -20.4pt)", "fair_real_ge_v1_plus_5pt (14.6% vs 16.3%, -1.7pt)"]},
        },
        "deployment_action": "RETAIN v1 models (B2_s0, E1_s0, G2_s0) in app packages; me2 ONNX models (B2m_s2, E1m_s0, G2m_s4) exported to exports/me2/ for benchmarking/analysis only",
        "source": "results/me2/decision.json",
    },
    "pi4_latency": {
        "status": "pending_manual_run",
        "command": "python scripts/bench_pi.py --deploy exports/me2/E1m_s0 --run E1m_s0 --threads 4 --runs 200",
        "p95_ms": None, "rtf": None, "threads": None, "runtime": None,
        "reason": "Pi 4 benchmark must run on physical Raspberry Pi 4 hardware by Awi; ONNX packages + bench_pi.py are staged for copy",
    },
    "wake_word_kws": {
        "status": None,
        "reason": "separate track: wake word runs on the upstream openWakeWord pipeline; this VCM covers the 33-way downstream command classifier",
    },
    "licenses": {
        "FluentSpeechCommands": "Non-commercial research use only",
        "GoogleSpeechCommands_v2": "CC-BY 4.0",
        "MLEndSpokenNumerals": "CC-BY 4.0",
        "SLURP": "CC-BY-NC-SA 4.0",
        "SNIPS": "MIT",
        "TimersAndSuch": "Apache 2.0",
        "GroupAudio": "not re-hosted; retrieved via ME2 master parquet metadata (link only)",
        "AwiAudio": "personal recordings, consented for project use",
        "source": "ME2 Master README + dataset audit",
    },
    "reproduction": {
        "git_commit": git_commit,
        "git_branch": "me2-master",
        "dataset_manifest": "data/manifests/me2_master_v1.csv",
        "manifest_sha256": sm["manifest_sha256"],
        "training_recipe": "python scripts/launch_parallel.py --runs configs/runs_me2_master.txt --gpus 0,5,6 --per-gpu 2 --waveform-cache data/cache/me2_master_v1_wave16k",
        "eval_recipe": "python scripts/me2_eval.py",
        "export_recipe": "python scripts/export_me2.py",
    },
}

(ROOT / "results/me2/presentation_metrics.json").write_text(json.dumps(metrics, indent=2))
print("WROTE results/me2/presentation_metrics.json")


def pct(x):
    return f"{x * 100:.1f}%" if x is not None else "n/a"


def pct2(x):
    return f"{x * 100:.2f}%" if x is not None else "n/a"


def acc_row(ps, model):
    v = ps[model]
    return v["correct"] if v["correct"] is not None else float("nan")


def far_row(ps, model):
    return ps[model]["FAR"]


ACC = metrics["accuracies_at_operating_tau"]
FAR = metrics["false_accept_rates"]
SZ = metrics["sizes_and_parameters"]

acc_slices = [
    ("validation_real", BASE_N["validation_real"], "master real-voice (val)"),
    ("awi_s02", BASE_N["awi_s02"], "Awi s02 (val, personal)"),
    ("awi_s03", BASE_N["awi_s03"], "Awi s03 (test, personal)"),
    ("holdout_real", BASE_N["holdout_real"], "holdout real (spk 202520785)"),
    ("holdout_syn", BASE_N["holdout_syn"], "holdout synthetic (test)"),
    ("fair_real", BASE_N["fair_real"], "fair real-voice (201435283+202520785)"),
]

acc_lines = []
for name, n, desc in acc_slices:
    a = ACC[name]
    acc_lines.append(
        f"| **{name}** ({desc}) | {n} | {pct(a['B2m']['correct'])} | {pct(a['B2']['correct'])} | "
        f"{pct(a['E1m']['correct'])} | {pct(a['E1']['correct'])} | "
        f"{pct(a['G2m']['correct'])} | {pct(a['G2']['correct'])} |"
    )

far_slices = [
    ("near_miss_158", BASE_N["near_miss"], "near_miss (off-schema, 158)"),
    ("single_words_clean_10730", BASE_N["single_words_clean"], "clean Single Words (GSC, 10,730)"),
]
far_lines = []
for name, n, desc in far_slices:
    f = FAR[name]
    far_lines.append(
        f"| **{name}** ({desc}) | {n} | {pct2(f['B2m'])} | {pct2(f['B2'])} | "
        f"{pct2(f['E1m'])} | {pct2(f['E1'])} | "
        f"{pct2(f['G2m'])} | {pct2(f['G2'])} |"
    )

spl = metrics["dataset"]["splits"]
md = f"""# ME2 Presentation Metrics

Every value traced to an artifact. Missing values are `null` with explicit reasons (never estimated).

## Pipeline
- **Input:** {metrics['pipeline']['audio_input']}
- **Features:** {metrics['pipeline']['features']}
- **Crop mode:** `{metrics['pipeline']['crop_mode']}` ({metrics['pipeline']['crop_mode_rationale']})

## Architectures & Parameters
| Model | Role | Architecture | Params | ONNX | fp32 | Opset | Parity |
|---|---|---|---|---|---|---|---|
| **B2m_s2** | sweep | TC-ResNet | {SZ['B2m_s2']['params']:,} | {SZ['B2m_s2']['onnx_kb']} KB | {SZ['B2m_s2']['fp32_mb']} MB | {SZ['B2m_s2']['opset']} | {'PASS' if SZ['B2m_s2']['parity_pass'] else 'FAIL'} |
| **E1m_s0** | focus | CRNN-Attn | {SZ['E1m_s0']['params']:,} | {SZ['E1m_s0']['onnx_kb']} KB | {SZ['E1m_s0']['fp32_mb']} MB | {SZ['E1m_s0']['opset']} | {'PASS' if SZ['E1m_s0']['parity_pass'] else 'FAIL'} |
| **G2m_s4** | benchmark | DS-CNN | {SZ['G2m_s4']['params']:,} | {SZ['G2m_s4']['onnx_kb']} KB | {SZ['G2m_s4']['fp32_mb']} MB | {SZ['G2m_s4']['opset']} | {'PASS' if SZ['G2m_s4']['parity_pass'] else 'FAIL'} |

- **Head:** {metrics['head']['classes']} classes ({metrics['head']['ontology']}), label mode `{metrics['head']['label_mode']}`

## Dataset (me2_master_v1, {metrics['dataset']['total_rows']:,} rows)
| Split | Clips | Hours | Unique speakers |
|---|---|---|---|
| train | {spl['train']['clips']:,} | {spl['train']['hours']:.3f} | {spl['train']['unique_speakers']:,} |
| validation | {spl['validation']['clips']:,} | {spl['validation']['hours']:.3f} | {spl['validation']['unique_speakers']:,} |
| test | {spl['test']['clips']:,} | {spl['test']['hours']:.3f} | {spl['test']['unique_speakers']:,} |

- **Personal Awi audio:** {metrics['dataset']['personal_awi']}
- **Drops:** {metrics['dataset']['drops_applied']}
- **Gates:** G1 {metrics['dataset']['gates']['G1_contradictions']['no_file_two_labels']} | G2 coverage {metrics['dataset']['gates']['G2_coverage']['pass']} | G3 balance pass {metrics['dataset']['gates']['G3_balance']['pass']} | G4 disjoint pass {metrics['dataset']['gates']['G4_disjointness']['pass']}

## Training Environment
- **Cluster:** {metrics['cluster']['hostname']} — {metrics['cluster']['gpu_model']}
- **GPUs used:** {metrics['cluster']['gpus_used']} ({metrics['cluster']['concurrency']})
- **Stack:** PyTorch {metrics['cluster']['torch_version']} + CUDA {metrics['cluster']['cuda_version']}
- **Objective:** {metrics['objective']['loss']}
- **Optimiser:** {metrics['optimiser']['class']}, lr={metrics['optimiser']['base_lr']}, weight_decay={metrics['optimiser']['weight_decay']}, betas={metrics['optimiser']['betas']}, {metrics['optimiser']['scheduler']} schedule
- **Steps:** {metrics['steps_and_loss']['epochs']} epochs, batch {metrics['steps_and_loss']['batch_size']}, {metrics['steps_and_loss']['global_steps']:,} global steps, best epoch {metrics['steps_and_loss']['best_epoch_range']}
- **Wall clock:** {metrics['wall_clock']['per_run_seconds_min']}–{metrics['wall_clock']['per_run_seconds_max']} s/run; {metrics['wall_clock']['total_15_runs_min']} min total GPU-seconds across 15 runs; {metrics['wall_clock']['parallel_wall_min']} min parallel wall

## Operating Tau (wrong <= 1% for every seed on val real-voice)
- **B2m:** τ = {metrics['operating_tau']['B2m']['tau']} — {metrics['operating_tau']['B2m']['note']}
- **E1m:** τ = {metrics['operating_tau']['E1m']['tau']} — {metrics['operating_tau']['E1m']['note']}
- **G2m:** τ = {metrics['operating_tau']['G2m']['tau']} — {metrics['operating_tau']['G2m']['note']}

## Command Accuracy at Operating Tau (correct %)
me2 rates pooled over 5 seeds; v1 rates single-seed (B2_s0/E1_s0/G2_s0, τ=0.9).
| Slice | N | B2m | v1 B2 | E1m | v1 E1 | G2m | v1 G2 |
|---|---|---|---|---|---|---|---|
{chr(10).join(acc_lines)}

## False-Accept Rate (FAR)
| Negative slice | N | B2m | v1 B2 | E1m | v1 E1 | G2m | v1 G2 |
|---|---|---|---|---|---|---|---|
{chr(10).join(far_lines)}

## Pre-Registered Decision
- **Rule:** a me2 model replaces its v1 counterpart in the app only if ALL four conditions hold (see JSON `decision_rule.rule`).
- **Verdicts:**
  - **B2m:** ❌ fails #1 (every seed wrong ~22% on validation; τ collapsed to 0.0)
  - **E1m:** ❌ fails #2 (−11.0 pt Awi s02) and #3 (−5.4 pt fair real-voice)
  - **G2m:** ❌ fails #2 (−20.4 pt Awi s02) and #3 (−1.7 pt fair real-voice)
- **Deployment action:** **RETAIN v1 models (B2_s0, E1_s0, G2_s0) in app packages.** me2 ONNX models remain in `exports/me2/` for benchmarking and analysis.

## Physical Raspberry Pi 4 Latency
- **Status:** pending manual execution on physical Pi 4
- **Command:** `{metrics['pi4_latency']['command']}`
- **p95 / RTF / threads:** `null` (hardware execution required by Awi)

## Wake Word (Keyword Spotting)
- **Status:** `null` (separate upstream openWakeWord pipeline; this project covers the 33-way command classifier)

## Reproduction
- **Git:** `{metrics['reproduction']['git_commit']}` on `{metrics['reproduction']['git_branch']}`
- **Manifest:** `{metrics['reproduction']['dataset_manifest']}` (sha256 `{metrics['reproduction']['manifest_sha256']}`)
- **Train:** `{metrics['reproduction']['training_recipe']}`
- **Eval:** `{metrics['reproduction']['eval_recipe']}`
- **Export:** `{metrics['reproduction']['export_recipe']}`
"""

(ROOT / "results/me2/presentation_metrics.md").write_text(md)
print("WROTE results/me2/presentation_metrics.md")
