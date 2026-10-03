"""Headless beta test: replay labelled WAVs through the RUNNING app and report
the per-model scoreboard. Used by agents (no browser, no microphone needed).

    python scripts/replay_eval.py --manifest ../tiny-vcm/data/manifests/composite_v1.csv \
        --split test --per-class 5 --out runtime/eval_B2_vs_A7.json

Reads a tiny-vcm manifest CSV (columns: relative_path, training_label,
slot_value, source_split[, corpus_id]); audio paths are resolved against
--audio-root (default: the tiny-vcm checkout). Each clip goes through the same
edge path as a live command (active model + challengers), tagged with its true
class, so the app's scoreboard and CSV exports are the result.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from vcm_common.ontology import get_ontology  # noqa: E402

SLOTTED = set(get_ontology().slotted)
BATCH_TIMEOUT_S = 120.0


def class_key(row: dict) -> str:
    label, slot = row["training_label"], (row.get("slot_value") or "").strip()
    return f"{label}|{slot}" if label in SLOTTED and slot else label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--audio-root", type=Path, default=REPO.parent / "tiny-vcm")
    ap.add_argument("--split", default="test")
    ap.add_argument("--corpus", action="append", default=None, help="filter corpus_id (repeatable)")
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    leaves = set(get_ontology().leaf_labels())
    by_class: dict[str, list[Path]] = defaultdict(list)
    with args.manifest.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if args.split and row.get("source_split") != args.split:
                continue
            if args.corpus and row.get("corpus_id") not in args.corpus:
                continue
            key = class_key(row)
            path = args.audio_root / row["relative_path"]
            if key in leaves and path.suffix.lower() == ".wav" and path.exists():
                by_class[key].append(path)
    if not by_class:
        sys.exit("no usable rows (check --split / --corpus / --audio-root)")

    rng = random.Random(args.seed)
    client = httpx.Client(base_url=args.url, timeout=30)
    sent = 0
    start = {row["model_id"]: row["n_clips"] for row in client.get("/api/state").json()["scoreboard"]}
    for key in sorted(by_class):
        clips = rng.sample(by_class[key], min(args.per_class, len(by_class[key])))
        files = [("files", (p.name, p.read_bytes(), "audio/wav")) for p in clips]
        r = client.post("/api/replay", files=files, data={"expected_class": key})
        r.raise_for_status()
        sent += len(clips)
        t0 = time.time()  # wait until this batch has been classified
        while time.time() - t0 < BATCH_TIMEOUT_S:
            s = client.get("/api/state").json()
            active = s["edge"]["active_vcm"]
            n_scored = sum(row["n_clips"] for row in s["scoreboard"] if row["model_id"] == active)
            if s["phase"].get("phase") == "idle" and n_scored - start.get(active, 0) >= sent:
                break
            time.sleep(0.5)
        else:
            sys.exit(f"timed out waiting for batch {key}")
        print(f"{key:32s} {len(clips)} clips")

    s = client.get("/api/state").json()
    report = {"manifest": str(args.manifest), "split": args.split, "per_class": args.per_class,
              "clips_sent": sent, "active_vcm": s["edge"]["active_vcm"],
              "compare_ids": s["edge"]["compare_ids"], "scoreboard": s["scoreboard"],
              "note": "scoreboard is cumulative over the app's runtime DB, not only this run"}
    print(json.dumps(report["scoreboard"], indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
