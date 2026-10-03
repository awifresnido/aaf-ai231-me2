#!/usr/bin/env python3
"""Adapt locally recorded sessions into the canonical manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from vcm_data.common import empty_record, normalize_text, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/personal/raw"))
    parser.add_argument(
        "--augmented-root", type=Path, default=Path("data/processed/personal_awi_augmented")
    )
    parser.add_argument("--output", type=Path, default=Path("data/manifests/personal_awi.csv"))
    parser.add_argument("--session-splits", type=Path, default=None)
    args = parser.parse_args()

    split_map: dict[str, str] = {}
    if args.session_splits:
        with args.session_splits.open(newline="", encoding="utf-8") as handle:
            split_map = {row["session_id"]: row["split"] for row in csv.DictReader(handle)}

    rows: list[dict[str, object]] = []
    session_files = sorted(args.root.glob("*/*/recordings.csv"))
    if args.augmented_root.exists():
        session_files += sorted(args.augmented_root.glob("*/*/recordings.csv"))
    for session_file in session_files:
        with session_file.open(newline="", encoding="utf-8") as handle:
            for source in csv.DictReader(handle):
                is_augmented = source.get("is_augmented", "false").lower() == "true"
                source_split = (source.get("source_split") or "").strip() or split_map.get(
                    source["session_id"], "unassigned"
                )
                if is_augmented and source_split != "train":
                    raise ValueError(
                        f"Synthetic sample {source['sample_id']} belongs to {source_split}; "
                        "augmentation is train-only"
                    )
                record = empty_record()
                record.update(source)
                record.update({
                    "schema_version": "2.0.0",
                    "corpus_id": "personal_awi",
                    "normalized_transcript": normalize_text(source["raw_transcript"]),
                    "language": "en",
                    "locale": "en-PH",
                    "source_split": source_split,
                    "device_id": source["microphone"],
                    "parent_sample_id": source.get("parent_sample_id", ""),
                    "is_augmented": "true" if is_augmented else "false",
                    "annotation_status": "synthetic_derived" if is_augmented else "prompt_verified",
                })
                rows.append(record)
    count = write_manifest(args.output, rows)
    print(f"Wrote {count} personal rows to {args.output}")


if __name__ == "__main__":
    main()
