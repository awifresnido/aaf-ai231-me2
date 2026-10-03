#!/usr/bin/env python3
"""Validate canonical VCM manifests and report leakage/quality failures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from vcm_data.common import (
    MANIFEST_COLUMNS,
    load_ontology,
    load_slot_values,
    sha256_file,
    wav_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--ontology", type=Path, default=Path("configs/ontology.json"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--check-hash", action="store_true")
    parser.add_argument("--require-audio-standard", action="store_true")
    parser.add_argument("--check-speaker-split", action="store_true")
    args = parser.parse_args()

    training_labels, leaf_to_training = load_ontology(args.ontology)
    slot_values = load_slot_values(args.ontology)
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    speaker_splits: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                errors.append(f"line {line_number}: malformed or truncated CSV row")
                continue
            sample_id = row["sample_id"]
            counts[row["training_label"]] += 1
            if not sample_id:
                errors.append(f"line {line_number}: empty sample_id")
            elif sample_id in seen_ids:
                errors.append(f"line {line_number}: duplicate sample_id {sample_id}")
            seen_ids.add(sample_id)

            label = row["training_label"]
            leaf = row["leaf_label"]
            if label not in training_labels:
                errors.append(f"{sample_id}: unknown training_label {label}")
            if leaf not in leaf_to_training:
                errors.append(f"{sample_id}: unknown leaf_label {leaf}")
            elif leaf_to_training[leaf] != label:
                errors.append(f"{sample_id}: leaf {leaf} maps to {leaf_to_training[leaf]}, not {label}")
            try:
                json.loads(row["slots_json"] or "{}")
            except json.JSONDecodeError:
                errors.append(f"{sample_id}: invalid slots_json")

            slot_type = row.get("slot_type", "")
            slot_value = row.get("slot_value", "")
            if slot_type == "slotted":
                allowed = slot_values.get(label)
                if not allowed:
                    errors.append(f"{sample_id}: label {label} is not slotted in the ontology")
                elif slot_value not in allowed:
                    errors.append(
                        f"{sample_id}: slot_value {slot_value!r} not in the closed set for {label}"
                    )
            elif label in slot_values:
                errors.append(f"{sample_id}: slotted label {label} has slot_type={slot_type!r}")

            digest = row["sha256"]
            if digest and digest in seen_hashes:
                errors.append(f"{sample_id}: duplicate audio hash also used by {seen_hashes[digest]}")
            if not digest:
                warnings.append(f"{sample_id}: no sha256 recorded; dedupe and hash check skipped")
            if digest:
                seen_hashes[digest] = sample_id

            audio_path = Path(row["relative_path"])
            if not audio_path.is_absolute():
                audio_path = args.project_root / audio_path
            if not audio_path.exists():
                errors.append(f"{sample_id}: missing file {audio_path}")
                continue
            if args.check_hash and digest and sha256_file(audio_path) != digest:
                errors.append(f"{sample_id}: SHA-256 mismatch")
            if audio_path.suffix.lower() == ".wav":
                metadata = wav_metadata(audio_path)
                for key, observed in metadata.items():
                    if row[key] and int(row[key]) != observed:
                        errors.append(f"{sample_id}: {key} manifest={row[key]} file={observed}")
                if args.require_audio_standard:
                    expected = {"sample_rate_hz": 16000, "channels": 1, "bit_depth": 16}
                    for key, value in expected.items():
                        if metadata[key] != value:
                            errors.append(f"{sample_id}: expected {key}={value}, got {metadata[key]}")
            else:
                warnings.append(f"{sample_id}: non-WAV audio not PCM-validated ({audio_path.suffix})")

            if row["speaker_id"] and row["source_split"] not in {"", "unassigned"}:
                speaker_splits[row["speaker_id"]].add(row["source_split"])

    if args.check_speaker_split:
        for speaker, splits in speaker_splits.items():
            if len(splits) > 1:
                errors.append(f"speaker leakage: {speaker} appears in {sorted(splits)}")

    print(f"Rows: {sum(counts.values())}")
    print("By training_label:", dict(sorted(counts.items())))
    print(f"Warnings: {len(warnings)}; Errors: {len(errors)}")
    for warning in warnings[:20]:
        print("WARNING:", warning)
    for error in errors[:50]:
        print("ERROR:", error)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
