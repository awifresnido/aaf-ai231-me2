#!/usr/bin/env bash
# bench_pi.sh — the exact class-benchmark invocation for live run 20261003-005002.
#
# Wraps bench/vcm-benchmark (the class SOP, pinned submodule) with the arguments
# recorded in docs/deployment.md. Usage:
#
#   ./scripts/bench_pi.sh <user@host> [--model path] [--seed N] [--run-id id]
#
# The run is written to bench/vcm-benchmark/runs/<run-id>; its metrics.json is
# then copied to results/me2_gold/live/<run-id>/metrics.json.
set -euo pipefail

HOST="${1:?usage: bench_pi.sh <user@host> [--model path] [--seed N] [--run-id id]}"
shift

MODEL="exports/me2_gold/E1f_s1/model.onnx"
SEED=15206
RUN_ID="20261003-005002"
STUDENT="202453069"
WAKE_WORD="Hey Rhasspy"
WAKE_GAP=0.8
SIZE="full"
PROC="tinyvcm"          # regex matching the edge-service process command line

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="$2"; shift 2 ;;
        --seed)    SEED="$2";  shift 2 ;;
        --run-id)  RUN_ID="$2"; shift 2 ;;
        --student) STUDENT="$2"; shift 2 ;;
        --size)    SIZE="$2";  shift 2 ;;
        --proc)    PROC="$2";  shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

python bench/vcm-benchmark/benchmark.py \
    --mode ssh --host "$HOST" \
    --student "$STUDENT" \
    --wake-word "$WAKE_WORD" --wake-gap "$WAKE_GAP" \
    --size "$SIZE" --seed "$SEED" \
    --model "$MODEL" \
    --proc "$PROC" \
    --yes

mkdir -p "results/me2_gold/live/${RUN_ID}"
cp "bench/vcm-benchmark/runs/${RUN_ID}/metrics.json" \
   "results/me2_gold/live/${RUN_ID}/metrics.json" 2>/dev/null || true
echo "benchmark complete; see results/me2_gold/live/${RUN_ID}/"
