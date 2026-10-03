#!/usr/bin/env python3
"""Slice GSC background recordings into command-window silence/noise clips."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

from vcm_data.common import empty_record, sha256_file, stable_id, wav_metadata, write_manifest

NOISE_SOURCE_SPLITS = {
    "doing_the_dishes": "train",
    "dude_miaowing": "train",
    "exercise_bike": "train",
    "pink_noise": "train",
    "running_tap": "validation",
    "white_noise": "test",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/external/gsc/raw/_background_noise_"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/gsc_noise"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/gsc_noise.csv"))
    parser.add_argument("--window-ms", type=int, default=2500)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for source in sorted(args.root.glob("*.wav")):
        if source.stem not in NOISE_SOURCE_SPLITS:
            raise ValueError(f"No source-disjoint split assigned for {source.name}")
        source_split = NOISE_SOURCE_SPLITS[source.stem]
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate_hz = handle.getframerate()
            if channels != 1 or sample_width != 2:
                raise ValueError(f"Expected mono 16-bit PCM: {source}")
            frames_per_window = round(sample_rate_hz * args.window_ms / 1000)
            index = 0
            while True:
                frames = handle.readframes(frames_per_window)
                if len(frames) < frames_per_window * sample_width:
                    break
                sample_id = stable_id("gsc_noise", source.name, index, args.window_ms)
                target = args.output_dir / f"{sample_id}.wav"
                with wave.open(str(target), "wb") as out:
                    out.setnchannels(channels)
                    out.setsampwidth(sample_width)
                    out.setframerate(sample_rate_hz)
                    out.writeframes(frames)
                metadata = wav_metadata(target)
                record = empty_record()
                record.update({
                    "sample_id": sample_id,
                    "schema_version": "2.0.0",
                    "corpus_id": "gsc_noise",
                    "relative_path": target.as_posix(),
                    "sha256": sha256_file(target),
                    "speaker_id": f"gsc_noise_{source.stem}",
                    "session_id": source.stem,
                    "prompt_id": "gsc_background_noise",
                    "phrase_family_id": f"gsc_background_{source.stem}",
                    "take_number": str(index + 1),
                    "raw_transcript": "",
                    "normalized_transcript": "",
                    "language": "none",
                    "training_label": "SILENCE",
                    "leaf_label": "SILENCE",
                    "action": "SILENCE",
                    "slot_type": "none",
                    "slot_value": "",
                    "slots_json": "{}",
                    "sample_type": "silence_noise",
                    "source_split": source_split,
                    **metadata,
                    "parent_sample_id": f"gsc_background_{source.stem}",
                    "is_augmented": "false",
                    "license": "CC-BY-4.0",
                    "annotation_status": "derived_window",
                })
                rows.append(record)
                index += 1
    count = write_manifest(args.manifest, rows)
    print(f"Wrote {count} noise clips and manifest rows")


if __name__ == "__main__":
    main()
