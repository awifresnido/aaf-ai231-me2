#!/usr/bin/env python3
"""Create deterministic, train-only synthetic variants of personal WAV recordings.

The script does not clone a voice or create new speakers. It produces mild
same-speaker augmentation using resampling, gain adjustment, and optional real
GSC background-noise mixing. Validation/test sessions must never be passed in.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import wave
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

RECORDING_FIELDS = [
    "sample_id", "relative_path", "speaker_id", "session_id", "prompt_id",
    "phrase_family_id", "raw_transcript", "training_label", "leaf_label", "action",
    "slot_type", "slot_value",
    "slots_json", "sample_type", "take_number", "recorded_at_utc", "microphone",
    "environment", "distance_cm", "speaking_style", "sample_rate_hz", "channels",
    "bit_depth", "duration_ms", "sha256", "consent_id", "license",
    "parent_sample_id", "is_augmented",
]


def read_pcm16_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit PCM WAV into float32 samples in [-1, 1]."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"Expected mono 16-bit PCM WAV: {path}")
        sample_rate_hz = handle.getframerate()
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return samples.astype(np.float32) / 32768.0, sample_rate_hz


def fit_length(samples: np.ndarray, target_length: int) -> np.ndarray:
    """Center-crop or zero-pad samples to a fixed length."""
    if len(samples) > target_length:
        start = (len(samples) - target_length) // 2
        return samples[start : start + target_length]
    if len(samples) < target_length:
        total_padding = target_length - len(samples)
        left = total_padding // 2
        return np.pad(samples, (left, total_padding - left))
    return samples


def resample_speed(samples: np.ndarray, speed_factor: float) -> np.ndarray:
    """Apply mild speed/pitch variation using dependency-light interpolation."""
    output_length = max(1, round(len(samples) / speed_factor))
    source_positions = np.linspace(0, len(samples) - 1, num=len(samples), dtype=np.float64)
    target_positions = np.linspace(0, len(samples) - 1, num=output_length, dtype=np.float64)
    changed = np.interp(target_positions, source_positions, samples).astype(np.float32)
    return fit_length(changed, len(samples))


def mix_noise_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mix noise at the requested SNR while retaining fixed clip length."""
    if len(noise) < len(speech):
        noise = np.tile(noise, math.ceil(len(speech) / len(noise)))
    max_start = len(noise) - len(speech)
    start = int(rng.integers(0, max_start + 1)) if max_start else 0
    noise = noise[start : start + len(speech)]
    noise_rms = float(np.sqrt(np.mean(np.square(noise))) + 1e-8)
    speech_rms = float(np.sqrt(np.mean(np.square(speech))) + 1e-8)
    target_noise_rms = speech_rms / (10 ** (snr_db / 20))
    return speech + noise * (target_noise_rms / noise_rms)


def write_pcm16_mono(path: Path, samples: np.ndarray, sample_rate_hz: int) -> None:
    """Write clipped float audio as mono 16-bit PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/processed/personal_awi_augmented")
    )
    parser.add_argument("--noise-root", type=Path, default=Path("data/processed/gsc_noise"))
    parser.add_argument("--variants-per-clip", type=int, default=1, choices=range(1, 4))
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument(
        "--confirm-train-only",
        action="store_true",
        help="Required acknowledgment that the input session belongs to the train split",
    )
    args = parser.parse_args()
    if not args.confirm_train_only:
        raise SystemExit("Refusing augmentation without --confirm-train-only")

    with args.recordings.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise SystemExit(f"No rows found in {args.recordings}")

    noise_paths = sorted(args.noise_root.glob("*.wav"))
    output_rows: list[dict[str, str]] = []
    for source in source_rows:
        if source["sample_type"] == "silence_noise":
            continue
        source_path = Path(source["relative_path"])
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        speech, sample_rate_hz = read_pcm16_mono(source_path)
        for variant_index in range(1, args.variants_per_clip + 1):
            digest_seed = f"{source['sample_id']}|{variant_index}|{args.seed}"
            numeric_seed = int.from_bytes(
                hashlib.sha256(digest_seed.encode("utf-8")).digest()[:8], "big"
            )
            rng = np.random.default_rng(numeric_seed)
            speed_factor = float(rng.uniform(0.94, 1.06))
            gain_db = float(rng.uniform(-3.0, 3.0))
            snr_db = float(rng.uniform(12.0, 20.0))

            augmented = resample_speed(speech, speed_factor)
            augmented *= 10 ** (gain_db / 20)
            noise_name = "none"
            if noise_paths:
                noise_path = noise_paths[int(rng.integers(0, len(noise_paths)))]
                noise, noise_rate_hz = read_pcm16_mono(noise_path)
                if noise_rate_hz != sample_rate_hz:
                    raise ValueError(
                        f"Noise sample rate {noise_rate_hz} != speech {sample_rate_hz}: {noise_path}"
                    )
                augmented = mix_noise_at_snr(augmented, noise, snr_db, rng)
                noise_name = noise_path.stem

            sample_id = "personal_awi_aug_" + hashlib.sha1(
                digest_seed.encode("utf-8"), usedforsecurity=False
            ).hexdigest()[:16]
            output_dir = (
                args.output_root / source["speaker_id"] / f"{source['session_id']}_synth"
            )
            target = output_dir / f"{sample_id}.wav"
            write_pcm16_mono(target, augmented, sample_rate_hz)
            file_digest = hashlib.sha256(target.read_bytes()).hexdigest()
            recipe = (
                f"synth_speed={speed_factor:.3f};gain_db={gain_db:.2f};"
                f"snr_db={snr_db:.1f};noise={noise_name}"
            )
            record = {field: source.get(field, "") for field in RECORDING_FIELDS}
            record.update({
                "sample_id": sample_id,
                "relative_path": target.as_posix(),
                "take_number": f"synth_{variant_index}",
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "environment": f"synthetic_from_{source['environment']}",
                "speaking_style": recipe,
                "duration_ms": str(round(len(augmented) * 1000 / sample_rate_hz)),
                "sha256": file_digest,
                "parent_sample_id": source["sample_id"],
                "is_augmented": "true",
            })
            output_rows.append(record)

    if not output_rows:
        raise SystemExit("No speech rows were eligible for augmentation")
    manifest = (
        args.output_root
        / source_rows[0]["speaker_id"]
        / f"{source_rows[0]['session_id']}_synth"
        / "recordings.csv"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORDING_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(manifest)
    print(f"Wrote {len(output_rows)} train-only synthetic variants to {manifest}")


if __name__ == "__main__":
    main()
