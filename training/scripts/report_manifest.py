#!/usr/bin/env python3
"""Generate a compact Markdown dataset card from a canonical manifest."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def table(title: str, counter: Counter[str]) -> list[str]:
    lines = [f"## {title}", "", "| Value | Samples |", "|---|---:|"]
    lines.extend(f"| `{key or '(empty)'}` | {value:,} |" for key, value in sorted(counter.items()))
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    by_corpus: Counter[str] = Counter()
    by_label: Counter[str] = Counter()
    by_leaf: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    speakers: set[str] = set()
    total_duration_ms = 0
    rows = 0
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            by_corpus[row["corpus_id"]] += 1
            by_label[row["training_label"]] += 1
            by_leaf[row["leaf_label"]] += 1
            by_split[row["source_split"]] += 1
            if row["speaker_id"]:
                speakers.add(row["speaker_id"])
            if row["duration_ms"]:
                total_duration_ms += int(row["duration_ms"])

    duration_hours = total_duration_ms / 3_600_000
    lines = [
        f"# Dataset card: {args.manifest.stem}",
        "",
        f"- Manifest: `{args.manifest}`",
        f"- Samples: **{rows:,}**",
        f"- Unique speaker/source IDs: **{len(speakers):,}**",
        f"- Total audio duration from headers: **{duration_hours:,.2f} h**",
        "- Counts describe the source pool before class-balanced training sampling.",
        "",
    ]
    lines += table("By corpus", by_corpus)
    lines += table("By assignment training label", by_label)
    lines += table("By executable leaf label", by_leaf)
    lines += table("By source split", by_split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote dataset card to {args.output}")


if __name__ == "__main__":
    main()
