#!/usr/bin/env python3
"""Adapt Google Speech Commands v0.02 into the Option B canonical manifest.

GSC is used only for UNKNOWN speech and digit-word slot vocabulary. Its isolated
words are not valid positives for multi-word Option B commands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vcm_data.common import empty_record, sha256_file, stable_id, wav_metadata, write_manifest

DIGITS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/external/gsc/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/gsc_v2.csv"))
    args = parser.parse_args()

    validation = set((args.root / "validation_list.txt").read_text().splitlines())
    testing = set((args.root / "testing_list.txt").read_text().splitlines())
    rows: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    duplicate_count = 0
    for wav_path in sorted(args.root.glob("*/*.wav")):
        word = wav_path.parent.name
        if word == "_background_noise_":
            continue
        source_relative = wav_path.relative_to(args.root).as_posix()
        source_split = (
            "test" if source_relative in testing
            else "validation" if source_relative in validation
            else "train"
        )
        speaker = wav_path.stem.split("_nohash_")[0]
        is_digit = word in DIGITS
        metadata = wav_metadata(wav_path)
        digest = sha256_file(wav_path)
        if digest in seen_hashes:
            duplicate_count += 1
            continue
        seen_hashes.add(digest)

        record = empty_record()
        record.update({
            "sample_id": stable_id("gsc_v2", source_relative),
            "schema_version": "2.0.0",
            "corpus_id": "gsc_v2",
            "relative_path": wav_path.as_posix(),
            "sha256": digest,
            "speaker_id": f"gsc_v2_{speaker}",
            "prompt_id": f"gsc_word_{word}",
            "phrase_family_id": f"gsc_word_{word}",
            "take_number": wav_path.stem.rsplit("_", 1)[-1],
            "raw_transcript": word,
            "normalized_transcript": word,
            "language": "en",
            "locale": "unknown",
            "training_label": "UNKNOWN",
            "leaf_label": "UNKNOWN",
            "action": "UNKNOWN",
            "slot_type": "numeric_word" if is_digit else "none",
            "slot_value": word if is_digit else "",
            "slots_json": json.dumps({"digit_word": word}) if is_digit else "{}",
            "sample_type": "slot_value" if is_digit else "unknown_speech",
            "source_split": source_split,
            **metadata,
            "is_augmented": "false",
            "license": "CC-BY-4.0",
            "annotation_status": "source_label",
        })
        rows.append(record)
    count = write_manifest(args.output, rows)
    print(
        f"Wrote {count} unique GSC rows to {args.output}; dropped {duplicate_count} duplicate file(s)"
    )


if __name__ == "__main__":
    main()
