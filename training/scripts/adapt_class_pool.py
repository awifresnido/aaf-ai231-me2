#!/usr/bin/env python3
"""Ingest the class voice pool into the canonical manifest, with no conversion step.

Reads every ``<raw-dir>/<speaker_id>/manifest.csv`` written by the class
recorder (our recorder writes the same format), and emits one canonical
manifest row per approved take.

The class manifest has no session/environment/split metadata, so those fields
are left empty or ``unassigned`` on purpose. Assign speakers to splits in a
separate, explicit step -- never by clip.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from vcm_data.common import (
    empty_record,
    load_ontology,
    normalize_text,
    sha256_file,
    write_manifest,
)

POOL_FIELDS = [
    "speaker_id", "prompt_id", "label", "type", "text", "slot_value",
    "take", "filename", "recorded_at", "whisper_transcript", "wer", "status",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/manifests/class_pool.csv"))
    parser.add_argument("--ontology", type=Path, default=Path("configs/ontology.json"))
    parser.add_argument("--license", default="class-pool-research")
    args = parser.parse_args()

    valid_labels, leaf_map = load_ontology(args.ontology)
    rows: list[dict[str, object]] = []
    skipped_status: Counter[str] = Counter()
    unknown_labels: Counter[str] = Counter()
    speakers: set[str] = set()

    for manifest_path in sorted(args.raw_dir.glob("*/manifest.csv")):
        folder_speaker = manifest_path.parent.name
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(POOL_FIELDS) - set(reader.fieldnames or [])
            if missing:
                raise SystemExit(f"{manifest_path}: missing class manifest columns {sorted(missing)}")
            for row in reader:
                status = (row.get("status") or "").strip().lower()
                if status != "approved":
                    skipped_status[status or "(blank)"] += 1
                    continue
                label = (row.get("label") or "").strip().upper()
                if label not in valid_labels:
                    unknown_labels[label or "(blank)"] += 1
                    continue
                speaker_id = (row.get("speaker_id") or folder_speaker).strip()
                if speaker_id != folder_speaker:
                    print(
                        f"WARNING: {manifest_path}: speaker_id={speaker_id!r} != folder "
                        f"{folder_speaker!r}; using folder name for speaker-disjoint splits"
                    )
                speakers.add(folder_speaker)
                wav_path = manifest_path.parent / row["filename"]
                if not wav_path.exists():
                    raise SystemExit(f"{manifest_path}: missing audio file {wav_path}")
                slot_value = (row.get("slot_value") or "").strip()
                prompt_type = (row.get("type") or "").strip().lower()
                take = (row.get("take") or "").strip()
                record = empty_record()
                record.update({
                    "sample_id": f"classpool_{folder_speaker}_{row['prompt_id']}_t{take}",
                    "schema_version": "2.0.0",
                    "corpus_id": "class_pool",
                    "relative_path": wav_path.as_posix(),
                    "sha256": sha256_file(wav_path),
                    "speaker_id": folder_speaker,
                    "prompt_id": row["prompt_id"],
                    "phrase_family_id": row["prompt_id"].rsplit("_V", 1)[0],
                    "take_number": take,
                    "raw_transcript": row.get("text", ""),
                    "normalized_transcript": normalize_text(row.get("text", "")),
                    "language": "en",
                    "locale": "unknown",
                    "training_label": label,
                    "leaf_label": leaf_map[label],
                    "action": label,
                    "slot_type": "slotted" if prompt_type == "slotted" else "none",
                    "slot_value": slot_value,
                    "slots_json": "{}",
                    "sample_type": "command",
                    "source_split": "unassigned",
                    "recorded_at_utc": row.get("recorded_at", ""),
                    "annotation_status": "whisper_validated",
                    "license": args.license,
                })
                rows.append(record)

    if not rows:
        raise SystemExit(f"No approved pool rows ingested from {args.raw_dir}")
    count = write_manifest(args.output, rows)
    print(f"Wrote {count} canonical pool rows from {len(speakers)} speaker(s) to {args.output}")
    print("By label:", dict(sorted(Counter(str(r['training_label']) for r in rows).items())))
    if skipped_status:
        print("Skipped non-approved statuses:", dict(skipped_status))
    if unknown_labels:
        print("Skipped labels outside the Option B ontology:", dict(unknown_labels))


if __name__ == "__main__":
    main()
