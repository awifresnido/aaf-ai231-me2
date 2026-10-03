#!/usr/bin/env python
"""Pack many small training runs onto all GPUs of the node.

A ~70k-parameter model cannot fill an A100 by itself, and must not be made
bigger (RPi 4 constraint) or be trained with a much bigger batch (that would
change the optimisation and break comparability with B2). So the GPUs are
kept busy by running MANY runs at once: K concurrent runs per GPU, each with
its own share of the CPU cores for data loading, pulling from a queue until
it is empty. GPU utilisation and memory are sampled to a CSV so the report
can show the actual numbers.

Runs file: one run per line, the arguments for train_vcm.py, and it must
include --run-name. Blank lines and lines starting with # are ignored.

    --phase 2 --arch crnn_attn --seed 0 --run-name E1_s0 ...

Usage:
    python scripts/launch_parallel.py --runs configs/runs_crnn_attn.txt \
        --gpus 0,1,2,3,4,5,6,7 --per-gpu 2 --waveform-cache data/cache/composite_v1_wave16k
"""
from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


def read_runs(path: str) -> list[list[str]]:
    runs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        args = shlex.split(line)
        if "--run-name" not in args:
            raise SystemExit(f"every run needs --run-name: {line}")
        runs.append(args)
    return runs


def run_name(args: list[str]) -> str:
    return args[args.index("--run-name") + 1]


def gpu_sampler(out_csv: Path, stop: threading.Event, period_s: float) -> None:
    """Sample nvidia-smi until stopped: timestamp, gpu, util %, memory used MiB."""
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "gpu", "util_pct", "mem_used_mib", "mem_total_mib"])
        while not stop.is_set():
            try:
                q = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout
                now = datetime.now().isoformat(timespec="seconds")
                for row in q.strip().splitlines():
                    w.writerow([now] + [c.strip() for c in row.split(",")])
                fh.flush()
            except Exception:
                pass
            stop.wait(period_s)


def summarise_util(csv_path: Path) -> str:
    per: dict[str, list[tuple[float, float]]] = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            per.setdefault(r["gpu"], []).append((float(r["util_pct"]), float(r["mem_used_mib"])))
    lines = []
    for g, vals in sorted(per.items(), key=lambda kv: int(kv[0])):
        u = [v[0] for v in vals]; m = [v[1] for v in vals]
        lines.append(f"  GPU {g}: util mean {sum(u) / len(u):5.1f} %  max {max(u):5.1f} % | "
                     f"mem mean {sum(m) / len(m):7.0f} MiB  max {max(m):7.0f} MiB")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--per-gpu", type=int, default=2, help="concurrent runs per GPU")
    ap.add_argument("--cpu-reserve", type=int, default=4, help="cores left for the OS / main processes")
    ap.add_argument("--waveform-cache", default=None, help="added to every run unless it sets its own")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--sample-period", type=float, default=15.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = read_runs(args.runs)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    slots = [(g, k) for g in gpus for k in range(args.per_gpu)]
    concurrent = min(len(slots), len(runs))
    workers = max(2, (os.cpu_count() or 8) - args.cpu_reserve) // max(1, concurrent)
    log_dir = Path("logs/launch"); log_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(runs)} runs | {len(gpus)} GPUs x {args.per_gpu} = {len(slots)} slots | "
          f"{workers} DataLoader workers per run ({os.cpu_count()} cores)")

    queue = list(runs)
    active: dict[tuple[str, int], tuple[subprocess.Popen, str, float]] = {}
    results: list[tuple[str, int, float, str]] = []

    stop = threading.Event()
    util_csv = log_dir / f"gpu_util_{datetime.now():%Y%m%d_%H%M%S}.csv"
    sampler = threading.Thread(target=gpu_sampler, args=(util_csv, stop, args.sample_period), daemon=True)
    if not args.dry_run:
        sampler.start()

    try:
        while queue or active:
            for slot in slots:
                if slot in active or not queue:
                    continue
                run_args = queue.pop(0)
                if args.waveform_cache and "--waveform-cache" not in run_args:
                    run_args = run_args + ["--waveform-cache", args.waveform_cache]
                if "--num-workers" not in run_args:
                    run_args = run_args + ["--num-workers", str(workers)]
                name = run_name(run_args)
                cmd = [args.python, "training/train_vcm.py", *run_args]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot[0], OMP_NUM_THREADS="2", PYTHONPATH="training")
                print(f"[start] {name} on GPU {slot[0]}: {' '.join(cmd)}")
                if args.dry_run:
                    continue
                log = open(log_dir / f"{name}.log", "w")
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                active[slot] = (proc, name, time.time())
            if args.dry_run:
                queue.clear()
                break
            time.sleep(5)
            for slot, (proc, name, t0) in list(active.items()):
                rc = proc.poll()
                if rc is not None:
                    mins = (time.time() - t0) / 60
                    results.append((name, rc, mins, slot[0]))
                    print(f"[done ] {name} rc={rc} {mins:.1f} min (GPU {slot[0]})")
                    del active[slot]
    finally:
        stop.set()

    if args.dry_run:
        return
    failed = [r for r in results if r[1] != 0]
    print(f"\n{len(results) - len(failed)}/{len(results)} runs succeeded")
    for name, rc, mins, g in failed:
        print(f"  FAILED {name} rc={rc} (see logs/launch/{name}.log)")
    if util_csv.exists():
        print(f"GPU utilisation ({util_csv}):\n{summarise_util(util_csv)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
