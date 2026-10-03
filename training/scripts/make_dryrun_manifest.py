#!/usr/bin/env python3
"""Create a compact deterministic manifest for Option B end-to-end dry runs.

A ``--force-include-corpus`` (default ``personal_awi``) is always kept in full,
and the remaining per-class budget is topped up from other corpora. This
guarantees the owner's recordings are actually exercised by the dry run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from vcm_data.common import write_manifest

# Budgets per split and group. Positives are per Option B label; UNKNOWN and
# SILENCE get their own caps so negatives cannot swamp the dry run.
DEFAULT_LIMITS = {
    "train": {"positive": 40, "unknown": 400, "silence": 90},
    "validation": {"positive": 10, "unknown": 100, "silence": 30},
    "test": {"positive": 10, "unknown": 100, "silence": 30},
}


def deterministic_rank(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{sample_id}".encode()).hexdigest()


def group_kind(label: str) -> str:
    if label == "UNKNOWN":
        return "unknown"
    if label == "SILENCE":
        return "silence"
    return "positive"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/manifests/public_base_v0.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/dryrun_public_v0.csv"))
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument(
        "--force-include-corpus", type=str, default="personal_awi",
        help="corpus_id to keep in full; remaining budget filled from other corpora",
    )
    args = parser.parse_args()

    positives: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    negatives: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with args.input.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            kind = group_kind(row["training_label"])
            if kind == "positive":
                positives[(row["source_split"], row["training_label"])].append(row)
            else:
                negatives[(row["source_split"], kind)].append(row)

    forced = args.force_include_corpus

    def select(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
        mine = [r for r in rows if r["corpus_id"] == forced]
        rest = [r for r in rows if r["corpus_id"] != forced]
        rest.sort(key=lambda row: deterministic_rank(row["sample_id"], args.seed))
        return mine + rest[: max(0, limit - len(mine))]

    selected: list[dict[str, str]] = []
    for (split, _label), rows in sorted(positives.items()):
        limit = DEFAULT_LIMITS.get(split, {}).get("positive", 0)
        selected.extend(select(rows, limit))
    for (split, kind), rows in sorted(negatives.items()):
        limit = DEFAULT_LIMITS.get(split, {}).get(kind, 0)
        selected.extend(select(rows, limit))

    selected.sort(key=lambda row: (row["source_split"], row["training_label"], row["sample_id"]))
    count = write_manifest(args.output, selected)
    print(f"Wrote {count} compact dry-run rows to {args.output}")


if __name__ == "__main__":
    main()
