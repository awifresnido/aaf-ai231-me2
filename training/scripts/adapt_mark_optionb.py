#!/usr/bin/env python3
"""Adapt Option B synthetic set into the canonical Option B manifest.

Source: markandrian30/AI231 -> MEX2/OptionB (Chatterbox TTS, 3 phrasings x 3 slot
values x many speakers, each clip in clean and ~30 dB-SNR noisy form).

Decisions baked in:
- ``FLAGGED/`` is own quarantine and is excluded entirely.
- ``_clean`` clips are the canonical source (``is_augmented=false``).
- ``_noisy`` clips are optional (``--include-noisy``) and are recorded as
  derived samples with ``parent_sample_id`` pointing at their clean twin.
- Splits are assigned per speaker by a deterministic hash, so no speaker ever
  appears in two partitions. Re-running gives identical splits.

The upstream repository ships no LICENSE file. Treat the material as
class-internal research data until Mark confirms written terms.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

from vcm_data.common import empty_record, sha256_file, stable_id, wav_metadata, write_manifest

CORPUS_ID = "mark_optionb"
FILENAME_RE = re.compile(
    r"^(?P<prefix>.+)_s(?P<speaker>\d+)_v(?P<variant>\d+)_(?P<kind>clean|noisy)\.wav$"
)

# directory -> (Option B label, closed slot value or "")
DIR_MAP: dict[str, tuple[str, str]] = {
    "PLAY_MUSIC": ("PLAY_MUSIC", ""),
    "WEATHER": ("WEATHER", ""),
    "TIME": ("TIME", ""),
    "LIGHT_ON": ("LIGHT_ON", ""),
    "LIGHT_OFF": ("LIGHT_OFF", ""),
    "PAUSE": ("PAUSE", ""),
    "STOP": ("STOP", ""),
    "NEXT": ("NEXT", ""),
    "VOLUME_UP": ("VOLUME_UP", ""),
    "VOLUME_DOWN": ("VOLUME_DOWN", ""),
    "CALL": ("CALL", ""),
    "MESSAGE": ("MESSAGE", ""),
    "LIST_REMINDERS": ("LIST_REMINDERS", ""),
    "TIMER_10s": ("TIMER", "10 seconds"),
    "TIMER_30s": ("TIMER", "30 seconds"),
    "TIMER_1m": ("TIMER", "1 minute"),
    "ALARM_6_00AM": ("ALARM", "6:00 AM"),
    "ALARM_8_00AM": ("ALARM", "8:00 AM"),
    "ALARM_9_00PM": ("ALARM", "9:00 PM"),
    "TEMPERATURE_18": ("TEMPERATURE", "18 degrees"),
    "TEMPERATURE_22": ("TEMPERATURE", "22 degrees"),
    "TEMPERATURE_26": ("TEMPERATURE", "26 degrees"),
    "BRIGHTNESS_20": ("BRIGHTNESS", "20 percent"),
    "BRIGHTNESS_60": ("BRIGHTNESS", "60 percent"),
    "BRIGHTNESS_100": ("BRIGHTNESS", "100 percent"),
    "COLOR_RED": ("COLOR", "Red"),
    "COLOR_BLUE": ("COLOR", "Blue"),
    "COLOR_GREEN": ("COLOR", "Green"),
    "CREATE_REMINDER_DRINK_WATER": ("CREATE_REMINDER", "Drink water"),
    "CREATE_REMINDER_STUDY": ("CREATE_REMINDER", "Study"),
    "CREATE_REMINDER_EXERCISE": ("CREATE_REMINDER", "Exercise"),
}

SLOTTED_LABELS = {
    "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER",
}


def speaker_split(speaker: str) -> str:
    """Deterministic ~80/10/10 speaker-disjoint split."""
    digest = hashlib.sha256(f"{CORPUS_ID}|{speaker}".encode()).digest()
    fraction = int.from_bytes(digest[:4], "big") / 2**32
    if fraction < 0.80:
        return "train"
    if fraction < 0.90:
        return "validation"
    return "test"


def load_prompt_index(path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    """Map (label, slot_value, variation number) -> class schema prompt row."""
    index: dict[tuple[str, str, int], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = row["label"].upper()
            slot = (row.get("slot_value") or "").strip()
            match = re.search(r"_V(\d+)(?:_|$)", row["prompt_id"])
            if not match:
                continue
            variation = int(match.group(1))
            index[(label, slot, variation)] = {
                "prompt_id": row["prompt_id"],
                "text": row["text"],
            }
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("data/external/mark_optionb/MEX2/OptionB")
    )
    parser.add_argument("--prompts", type=Path, default=Path("configs/recording_prompts.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/mark_optionb.csv"))
    parser.add_argument("--include-noisy", action="store_true")
    args = parser.parse_args()

    prompt_index = load_prompt_index(args.prompts)
    rows: list[dict[str, object]] = []
    seen_hashes: dict[str, str] = {}
    duplicate_count = 0
    unresolved: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    speakers: set[str] = set()

    for directory in sorted(p for p in args.root.iterdir() if p.is_dir()):
        if directory.name == "FLAGGED":
            continue
        if directory.name not in DIR_MAP:
            unresolved[f"unmapped_directory:{directory.name}"] += 1
            continue
        label, slot_value = DIR_MAP[directory.name]
        for wav_path in sorted(directory.glob("*.wav")):
            match = FILENAME_RE.match(wav_path.name)
            if not match:
                unresolved[f"unparsed_filename:{wav_path.name}"] += 1
                continue
            if match.group("prefix") != directory.name:
                unresolved[f"prefix_mismatch:{wav_path.name}"] += 1
                continue
            kind = match.group("kind")
            if kind == "noisy" and not args.include_noisy:
                continue

            speaker_number = match.group("speaker")
            variation = int(match.group("variant"))
            speaker_id = f"{CORPUS_ID}_s{speaker_number}"
            speakers.add(speaker_id)

            prompt = prompt_index.get((label, slot_value, variation))
            if prompt is None:
                unresolved[f"no_prompt:{label}|{slot_value}|v{variation}"] += 1

            metadata = wav_metadata(wav_path)
            formats[f"{metadata['sample_rate_hz']}Hz/{metadata['channels']}ch/"
                    f"{metadata['bit_depth']}bit"] += 1
            digest = sha256_file(wav_path)
            if digest in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes[digest] = wav_path.name

            split = speaker_split(speaker_number)
            by_split[split] += 1
            relative = wav_path.relative_to(args.root.parents[2]).as_posix()
            clean_relative = relative.replace("_noisy.wav", "_clean.wav")
            record = empty_record()
            record.update({
                "sample_id": stable_id(CORPUS_ID, relative),
                "schema_version": "2.0.0",
                "corpus_id": CORPUS_ID,
                "relative_path": wav_path.as_posix(),
                "sha256": digest,
                "speaker_id": speaker_id,
                "prompt_id": prompt["prompt_id"] if prompt else "",
                "phrase_family_id": prompt["prompt_id"] if prompt else f"{label}_{slot_value}",
                "take_number": str(variation),
                "raw_transcript": prompt["text"] if prompt else "",
                "normalized_transcript": prompt["text"].lower() if prompt else "",
                "language": "en",
                "locale": "unknown",
                "training_label": label,
                "leaf_label": label,
                "action": label,
                "slot_type": "slotted" if label in SLOTTED_LABELS else "none",
                "slot_value": slot_value,
                "slots_json": "{}",
                "sample_type": "command_slotted" if label in SLOTTED_LABELS else "command",
                "source_split": split,
                "device_id": "chatterbox_tts",
                "microphone": "chatterbox_tts",
                "environment": "synthetic_clean" if kind == "clean" else "synthetic_noisy_30db",
                "speaking_style": "tts_reference_clone",
                **metadata,
                "parent_sample_id": "" if kind == "clean" else stable_id(CORPUS_ID, clean_relative),
                "is_augmented": "false" if kind == "clean" else "true",
                "license": "AI231-classmate-share-no-license-file",
                "annotation_status": "classmate_synthetic_mapped_to_class_prompt",
            })
            rows.append(record)

    if not rows:
        raise SystemExit(f"No clips found under {args.root}")
    count = write_manifest(args.output, rows)
    print(f"Wrote {count} rows from Option B set to {args.output}")
    print(f"Speakers: {len(speakers)} | dropped {duplicate_count} duplicate file(s)")
    print("Audio formats:", dict(formats))
    print("By split:", dict(sorted(by_split.items())))
    print("By label:", dict(sorted(Counter(str(r['training_label']) for r in rows).items())))
    if unresolved:
        print("Unresolved:", dict(unresolved))


if __name__ == "__main__":
    main()
