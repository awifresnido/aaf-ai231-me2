#!/usr/bin/env python3
"""Normalize every row of a manifest to 16 kHz mono 16-bit PCM.

Source audio is never modified. Clips that already satisfy the contract keep
their original path. Anything else is resampled into ``--output-root`` and the
manifest row is repointed at the derived file, with ``parent_sample_id`` linking
back to the source clip.

Resampling is a technical format conversion, not data augmentation, so
``is_augmented`` stays false; the change is recorded in ``annotation_status``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import wave
from collections import Counter
from pathlib import Path

import numpy as np

TARGET_RATE = 16_000


def read_pcm16_mono(path: Path) -> tuple[np.ndarray, int, int, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM is supported (got {width * 8}-bit)")
    samples = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype("<i2")
    return samples, rate, channels, width


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    new_length = max(1, round(len(samples) * target_rate / source_rate))
    old_positions = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    new_positions = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
    resampled = np.interp(new_positions, old_positions, samples.astype(np.float64))
    return np.round(resampled).astype("<i2")


def write_pcm16(path: Path, samples: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/normalized_16k"))
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    converted: Counter[str] = Counter()
    already_ok = 0
    for row in rows:
        source = Path(row["relative_path"])
        samples, rate, channels, _ = read_pcm16_mono(source)
        if rate == TARGET_RATE and channels == 1:
            already_ok += 1
            continue
        normalized = resample(samples, rate, TARGET_RATE)
        target = args.output_root / row["corpus_id"] / f"{row['sample_id']}.wav"
        write_pcm16(target, normalized, TARGET_RATE)
        converted[f"{rate}Hz/{channels}ch -> 16kHz/1ch"] += 1
        row.update({
            "relative_path": target.as_posix(),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "sample_rate_hz": str(TARGET_RATE),
            "channels": "1",
            "bit_depth": "16",
            "duration_ms": str(round(len(normalized) * 1000 / TARGET_RATE)),
            "parent_sample_id": row["sample_id"],
            "annotation_status": f"{row['annotation_status']}+resampled_from_{rate}hz",
        })

    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.manifest)

    print(f"Manifest rewritten in place: {args.manifest}")
    print(f"Already compliant: {already_ok} | converted: {sum(converted.values())}")
    print("Conversions:", dict(converted) or "none")


if __name__ == "__main__":
    main()
