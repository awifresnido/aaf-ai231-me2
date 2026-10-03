#!/usr/bin/env bash
# reproduce.sh — one-command reproduction of the ME2 submission pipeline.
#
#   git clone --recursive https://github.com/awifresnido/aaf-ai231-me2.git
#   cd aaf-ai231-me2
#   ./reproduce.sh                                   # setup → data → weights → eval → export → app
#   ./reproduce.sh --train dgx --gpus 0,1,3          # + manifest → train → eval (fresh checkpoints)
#   ./reproduce.sh --train jetson                    # same, sequential, on a Jetson Orin Nano 8 GB
#   ./reproduce.sh --pi <user@host>                  # deploy edge to the Pi + run the class benchmark
#   ./reproduce.sh --dry-run [...]                   # print every command and URL, run nothing
#
# Stages are idempotent: each skips when its outputs already exist (and, where
# applicable, their checksums match).

set -euo pipefail

# ---------------------------------------------------------------------------
# Repository metadata
# ---------------------------------------------------------------------------
REPO="awifresnido/aaf-ai231-me2"
RELEASE_TAG="v0"
DATASET="airimonda/ai231-me2-voice-commands"
DATASET_REVISION="6947f13073e57eb6ae67e7e2fc3680700b82aa13"
DGX_DATASET_DIR="/data/ai231"          # shared class cache on the DGX (owner-provided load.py)
# Parquet form of the same HF dataset (train/test/holdout/numerals +
# supplemental_synth + synthetic_negatives). The manifest stage reads this
# directly with pyarrow (read-only). Override via env if yours lives elsewhere.
export ME2_GOLD_DIR="${ME2_GOLD_DIR:-$HOME/projects/ME2_GOLD}"
MANIFEST_SHA256="f81867392d20e3a23cbbcd2b77559c3b357e943f10e3095b65f7aa1b6171cdb2"

# ---------------------------------------------------------------------------
# Command dispatch (dry-run aware)
# ---------------------------------------------------------------------------
DRY_RUN=0
run() {
    # print a single command; in dry-run mode only echo it
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  $ %s\n' "$*"
    else
        printf '  $ %s\n' "$*" >&2
        "$@"
    fi
}

note() { printf '\n[%s] %s\n' "$1" "$2"; }
url() { printf '  ↳ %s\n' "$1"; }

usage() {
    sed -n '2,12p' "$0"
    exit 1
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
MODE="default"          # default | train | pi
TRAIN_TARGET="dgx"
GPUS="0,1,3"
PI_HOST=""
NO_PERSONAL=0
WITH_V1_SLICES=0
ONLY_STAGE=""
WEIGHTS_LOCAL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)          DRY_RUN=1; shift ;;
        --train)            MODE="train"; shift
                            if [[ $# -gt 0 && "$1" != --* ]]; then TRAIN_TARGET="$1"; shift; fi ;;
        --gpus)             GPUS="$2"; shift 2 ;;
        --pi)               MODE="pi"; PI_HOST="$2"; shift 2 ;;
        --no-personal)      NO_PERSONAL=1; shift ;;
        --with-v1-slices)   WITH_V1_SLICES=1; shift ;;
        --weights-local)    WEIGHTS_LOCAL="$2"; shift 2 ;;
        --stage)            ONLY_STAGE="$2"; shift 2 ;;
        -h|--help)          usage ;;
        *)                  echo "unknown arg: $1" >&2; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
stage_setup() {
    note "setup" "create venv from lock files; check Python / ffmpeg / Node"
    run python3 --version
    run python3 -m venv .venv
    run .venv/bin/pip install -U pip
    run .venv/bin/pip install -r env/requirements-train.txt
    run .venv/bin/pip install -r env/requirements-app.txt
    run .venv/bin/pip install -r env/requirements-pi.txt
    run ffmpeg -version
    run node --version
    run npm --version
}

stage_data() {
    note "data" "verify the gold dataset parquet is readable (ME2_GOLD_DIR)"
    url "Hugging Face: ${DATASET} @ ${DATASET_REVISION}"
    # The manifest stage reads the parquet directly (pyarrow, read-only). The
    # shared /data/ai231 HF arrow cache is read-only for non-owners, so its
    # owner-provided load.py cannot acquire the datasets cache lock. Verify the
    # parquet here — the exact source the pipeline consumes — instead.
    run .venv/bin/python -c 'import os, pyarrow.parquet as pq; from pathlib import Path; r = Path(os.environ["ME2_GOLD_DIR"]); fs = sorted(p for p in r.rglob("*.parquet") if ".cache" not in p.parts); print(f"parquet files: {len(fs)}"); print(f"total rows: {sum(pq.ParquetFile(f).metadata.num_rows for f in fs)}")'
    if [[ "$WITH_V1_SLICES" == "1" ]]; then
        note "data(+v1)" "also fetch Speech Commands v2 for the clean Single-Words FAR slice"
        url "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
        run mkdir -p data/external
        run curl -L -o data/external/speech_commands_v0.02.tar.gz \
            https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
    fi
}

stage_weights() {
    note "weights" "download release assets (checkpoints, ONNX, personal recordings) and verify"
    local assets=(
        "B2_s0:checkpoints/B2_s0/best.pt"
        "B2_s1:checkpoints/B2_s1/best.pt"
        "B2_s2:checkpoints/B2_s2/best.pt"
        "E1_s0:checkpoints/E1_s0/best.pt"
        "E1_s1:checkpoints/E1_s1/best.pt"
        "E1_s2:checkpoints/E1_s2/best.pt"
        "G2_s0:checkpoints/G2_s0/best.pt"
        "G2_s1:checkpoints/G2_s1/best.pt"
        "G2_s2:checkpoints/G2_s2/best.pt"
        "B2f_s0:checkpoints/B2f_s0/best.pt"
        "B2f_s1:checkpoints/B2f_s1/best.pt"
        "B2f_s2:checkpoints/B2f_s2/best.pt"
        "E1f_s0:checkpoints/E1f_s0/best.pt"
        "E1f_s1:checkpoints/E1f_s1/best.pt"
        "E1f_s2:checkpoints/E1f_s2/best.pt"
        "G2f_s0:checkpoints/G2f_s0/best.pt"
        "G2f_s1:checkpoints/G2f_s1/best.pt"
        "G2f_s2:checkpoints/G2f_s2/best.pt"
        "E1f_s1.onnx:exports/me2_gold/E1f_s1/model.onnx"
        "B2f_s0.onnx:exports/me2_gold/B2f_s0/model.onnx"
        "E1f_s1.deploy.json:exports/me2_gold/E1f_s1/deploy.json"
        "E1f_s1.labels.json:exports/me2_gold/E1f_s1/labels.json"
        "B2f_s0.deploy.json:exports/me2_gold/B2f_s0/deploy.json"
        "B2f_s0.labels.json:exports/me2_gold/B2f_s0/labels.json"
    )
    run mkdir -p checkpoints exports/me2_gold release
    for a in "${assets[@]}"; do
        local src="${a%%:*}"; local dst="${a##*:}"
        if [[ -n "$WEIGHTS_LOCAL" && -f "$WEIGHTS_LOCAL/$src" ]]; then
            run cp "$WEIGHTS_LOCAL/$src" "$dst"
        else
            url "https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${src}"
            run curl -fL -o "${dst}" \
                "https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${src}"
        fi
    done
    # model_config.json sidecars (architecture config, next to each best.pt)
    local sidecars=("E1_s0" "E1_s1" "E1_s2" "G2_s0" "G2_s1" "G2_s2"
                    "B2f_s0" "B2f_s1" "B2f_s2" "E1f_s0" "E1f_s1" "E1f_s2"
                    "G2f_s0" "G2f_s1" "G2f_s2")
    for id in "${sidecars[@]}"; do
        local dst="checkpoints/${id}/model_config.json"
        if [[ -n "$WEIGHTS_LOCAL" && -f "$WEIGHTS_LOCAL/${id}.model_config.json" ]]; then
            run cp "$WEIGHTS_LOCAL/${id}.model_config.json" "$dst"
        else
            run curl -fL -o "$dst" \
                "https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${id}.model_config.json" || true
        fi
    done
    if [[ "$NO_PERSONAL" == "0" ]]; then
        note "weights(+personal)" "Awi's personal recordings (published; needed for the exact manifest)"
        if [[ -n "$WEIGHTS_LOCAL" && -f "$WEIGHTS_LOCAL/personal_awi.zip" ]]; then
            run cp "$WEIGHTS_LOCAL/personal_awi.zip" release/personal_awi.zip
        else
            run curl -fL -o release/personal_awi.zip \
                "https://github.com/${REPO}/releases/download/${RELEASE_TAG}/personal_awi.zip"
        fi
        run mkdir -p data/personal/raw/202453069
        run unzip -o release/personal_awi.zip -d data/personal/raw/202453069
    fi
    run sha256sum -c results/me2_gold/checksums.sha256
}

stage_manifest() {
    note "manifest" "build me2_gold_v1.csv from the dataset + Awi recordings"
    run .venv/bin/python training/scripts/adapt_me2_gold.py --extract
    run .venv/bin/python training/scripts/extract_near_miss.py
    run sha256sum data/manifests/me2_gold_v1.csv
    printf '  ↳ expect manifest sha256 = %s\n' "$MANIFEST_SHA256"
}

stage_train() {
    note "train" "the exact launch_parallel.py invocation from configs/runs_me2_gold.txt"
    if [[ "$TRAIN_TARGET" == "jetson" ]]; then
        run .venv/bin/python training/scripts/launch_parallel.py \
            --runs configs/runs_me2_gold.txt --gpus -1 --per-gpu 1
    else
        run .venv/bin/python training/scripts/launch_parallel.py \
            --runs configs/runs_me2_gold.txt --gpus "$GPUS" --per-gpu 1
    fi
}

stage_eval() {
    note "eval" "evaluate all 18 checkpoints at τ=0.95, then apply the release gate"
    local models="B2_s0,B2_s1,B2_s2,E1_s0,E1_s1,E1_s2,G2_s0,G2_s1,G2_s2,B2f_s0,B2f_s1,B2f_s2,E1f_s0,E1f_s1,E1f_s2,G2f_s0,G2f_s1,G2f_s2"
    run .venv/bin/python training/scripts/me2_gold_eval.py --infer --models "$models" --gpu 0
    run .venv/bin/python training/scripts/me2_gold_eval.py --decide
}

stage_export() {
    note "export" "single-file ONNX (opset 17) + GPU/CPU augmentation parity check"
    run .venv/bin/python training/scripts/export_me2_gold.py
    run .venv/bin/python training/scripts/me2_parity_check.py --manifest data/manifests/me2_gold_v1.csv
}

stage_app() {
    note "app" "build the React UI, then start the laptop backend in mock-edge mode"
    run bash -c 'cd app/laptop/frontend && npm ci && npm run build'
    run bash -c 'cd app && ../.venv/bin/python -m laptop.app.main --config config/demo.yaml'
    url "http://localhost:8080"
    url "health: http://localhost:8080/api/state"
}

stage_bench() {
    note "bench" "deploy the edge service to the Pi and run the class benchmark"
    [[ -n "$PI_HOST" ]] || { echo "bench requires --pi <user@host>" >&2; exit 1; }
    run ./scripts/bench_pi.sh "$PI_HOST" --model exports/me2_gold/E1f_s1/model.onnx \
        --run-id 20261003-005002 --seed 15206
}

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
declare -a STAGES
if [[ -n "$ONLY_STAGE" ]]; then
    STAGES=("$ONLY_STAGE")
elif [[ "$MODE" == "train" ]]; then
    STAGES=(setup data manifest weights train eval export)
elif [[ "$MODE" == "pi" ]]; then
    STAGES=(setup data manifest weights eval export app bench)
else
    STAGES=(setup data manifest weights eval export app)
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "== reproduce.sh DRY RUN =="
    echo "   repo:   https://github.com/${REPO}  (release ${RELEASE_TAG})"
    echo "   dataset: ${DATASET} @ ${DATASET_REVISION}"
    echo "   stages:  $(IFS=,; echo "${STAGES[*]}")"
    echo
fi

for s in "${STAGES[@]}"; do
    case "$s" in
        setup)    stage_setup ;;
        data)     stage_data ;;
        weights)  stage_weights ;;
        manifest) stage_manifest ;;
        train)    stage_train ;;
        eval)     stage_eval ;;
        export)   stage_export ;;
        app)      stage_app ;;
        bench)    stage_bench ;;
        *)        echo "unknown stage: $s" >&2; exit 1 ;;
    esac
done

if [[ "$DRY_RUN" == "1" ]]; then
    echo
    echo "== dry run complete (no commands executed) =="
fi
