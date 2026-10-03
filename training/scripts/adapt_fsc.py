#!/usr/bin/env python3
"""Adapt Fluent Speech Commands into the Option B (19-label) canonical manifest.

FSC only overlaps six Option B labels. Everything else becomes UNKNOWN so it
serves as realistic near-domain negative speech instead of being discarded.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vcm_data.common import (
    empty_record,
    normalize_text,
    sha256_file,
    stable_id,
    wav_metadata,
    write_manifest,
)

FSC_MAPPING = {
    ("activate", "lights"): "LIGHT_ON",
    ("activate", "lamp"): "LIGHT_ON",
    ("deactivate", "lights"): "LIGHT_OFF",
    ("deactivate", "lamp"): "LIGHT_OFF",
    ("activate", "music"): "PLAY_MUSIC",
    ("deactivate", "music"): "PAUSE",
    ("increase", "volume"): "VOLUME_UP",
    ("decrease", "volume"): "VOLUME_DOWN",
}

# Deliberately NOT mapped to TEMPERATURE: Option B expects a degrees slot, and
# "increase/decrease the heat" carries no value. Keeping it as UNKNOWN preserves
# it as a hard negative near the thermostat domain.
EXCLUDED_WITH_REASON = {("increase", "heat"), ("decrease", "heat")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/external/fsc/raw/fluent_speech_commands_dataset"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/manifests/fsc.csv"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    duplicate_count = 0
    data_dir = args.root / "data"
    for split, filename in [
        ("train", "train_data.csv"),
        ("validation", "valid_data.csv"),
        ("test", "test_data.csv"),
    ]:
        with (data_dir / filename).open(newline="", encoding="utf-8") as handle:
            for source in csv.DictReader(handle):
                wav_path = args.root / source["path"]
                if not wav_path.exists():
                    raise FileNotFoundError(wav_path)
                metadata = wav_metadata(wav_path)
                digest = sha256_file(wav_path)
                if digest in seen_hashes:
                    duplicate_count += 1
                    continue
                seen_hashes.add(digest)

                mapped_label = FSC_MAPPING.get((source["action"], source["object"]))
                mapped = mapped_label is not None
                slots: dict[str, object] = {
                    "fsc_action": source["action"],
                    "fsc_object": source["object"],
                }
                if source["location"] != "none":
                    slots["location"] = source["location"]
                if not mapped and (source["action"], source["object"]) in EXCLUDED_WITH_REASON:
                    slots["unmapped_reason"] = "optionb_requires_degrees_slot"

                record = empty_record()
                record.update({
                    "sample_id": stable_id("fsc", source["path"]),
                    "schema_version": "2.0.0",
                    "corpus_id": "fsc",
                    "relative_path": wav_path.as_posix(),
                    "sha256": digest,
                    "speaker_id": f"fsc_{source['speakerId']}",
                    "prompt_id": "",
                    "phrase_family_id": normalize_text(source["transcription"]),
                    "raw_transcript": source["transcription"],
                    "normalized_transcript": normalize_text(source["transcription"]),
                    "language": "en",
                    "locale": "unknown",
                    "training_label": mapped_label or "UNKNOWN",
                    "leaf_label": mapped_label or "UNKNOWN",
                    "action": mapped_label or "UNKNOWN",
                    "slot_type": "none",
                    "slot_value": "",
                    "slots_json": json.dumps(slots, sort_keys=True, separators=(",", ":")),
                    "sample_type": "command" if mapped else "unknown_speech",
                    "source_split": split,
                    **metadata,
                    "is_augmented": "false",
                    "license": "FSC-Public-License-academic-noncommercial-no-sharing",
                    "annotation_status": "mapped_from_source_semantics" if mapped else "unknown_by_design",
                })
                rows.append(record)
    count = write_manifest(args.output, rows)
    print(
        f"Wrote {count} unique FSC rows to {args.output}; dropped {duplicate_count} duplicate file(s)"
    )


if __name__ == "__main__":
    main()
