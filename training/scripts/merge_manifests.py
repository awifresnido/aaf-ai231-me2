#!/usr/bin/env python3
"""Merge canonical manifests while preserving provenance and rejecting collisions."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from vcm_data.common import MANIFEST_COLUMNS, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    sample_ids: set[str] = set()
    hashes: dict[str, str] = {}
    for source in args.inputs:
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{source}: missing columns {sorted(missing)}")
            for row in reader:
                sample_id = row["sample_id"]
                if sample_id in sample_ids:
                    raise ValueError(f"Duplicate sample_id: {sample_id}")
                digest = row["sha256"]
                if digest and digest in hashes:
                    raise ValueError(
                        f"Duplicate audio hash across {hashes[digest]} and {sample_id}; "
                        "do not leak duplicates across corpora"
                    )
                sample_ids.add(sample_id)
                if digest:
                    hashes[digest] = sample_id
                rows.append(row)
    rows.sort(key=lambda row: (row["corpus_id"], row["sample_id"]))
    count = write_manifest(args.output, rows)
    print(f"Merged {count} rows from {len(args.inputs)} manifests into {args.output}")
    print("By corpus:", dict(sorted(Counter(row["corpus_id"] for row in rows).items())))
    print("By label:", dict(sorted(Counter(row["training_label"] for row in rows).items())))


if __name__ == "__main__":
    main()
