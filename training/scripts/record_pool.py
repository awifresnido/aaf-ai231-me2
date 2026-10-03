#!/usr/bin/env python3
"""Dual-output command recorder for AI231 ME2.

Records once and writes two artefacts from the same take:

1. **Class pool format** -- ``<pool-root>/<speaker_id>/<prompt_id>_t<take>.wav``
   plus a ``manifest.csv`` whose columns match the class recorder exactly, so
   the folder uploads to the shared Drive pool with no conversion.
2. **Local canonical format** -- ``data/personal/raw/<speaker_id>/<session_id>/``
   with the rich manifest the DGX training pipeline needs (session, device,
   environment, distance, split, hashes).

UNKNOWN and SILENCE prompts are local-only: the shared pool schema has no such
labels, so they are never written to the pool folder.

Run this on the machine with the microphone, not over SSH.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import wave
from datetime import UTC, datetime
from pathlib import Path

from vcm_data.common import MANIFEST_COLUMNS, normalize_text, stable_id

POOL_FIELDS = [
    "speaker_id", "prompt_id", "label", "type", "text", "slot_value",
    "take", "filename", "recorded_at", "whisper_transcript", "wer", "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("configs/recording_prompts.csv"))
    parser.add_argument("--speaker-id", required=True, help="lowercase, no spaces; must match Drive folder")
    parser.add_argument("--session-id", required=True, help="e.g. awi01_s01")
    parser.add_argument("--session-split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--microphone", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--distance-cm", type=int, required=True)
    parser.add_argument("--speaking-style", default="normal")
    parser.add_argument("--device", default=None)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--approved-takes", type=int, default=2)
    parser.add_argument("--take-offset", type=int, default=0, help="keep pool filenames unique across sessions")
    parser.add_argument("--labels", nargs="*", default=None, help="only record these labels")
    parser.add_argument("--pool-root", type=Path, default=Path("recordings"))
    parser.add_argument("--local-root", type=Path, default=Path("data/personal/raw"))
    parser.add_argument("--no-pool", action="store_true", help="write local canonical files only")
    parser.add_argument(
        "--resume", action="store_true",
        help="skip prompts that already have the requested number of local takes",
    )
    parser.add_argument("--whisper-model", default=None, help="enable whisper.cpp validation, e.g. base.en")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_words(text: str) -> list[str]:
    text = text.lower().replace("%", " percent")
    text = re.sub(r"(\d):(\d\d)", r"\1 \2", text)
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return text.split()


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


def write_pcm16(path: Path, frames_bytes: bytes, sample_rate_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(frames_bytes)


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_rows(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    sample_rate_hz = 16_000
    prompts = load_rows(args.prompts)
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompts}")
    if args.labels:
        wanted = {label.upper() for label in args.labels}
        prompts = [row for row in prompts if row["label"].upper() in wanted]
        if not prompts:
            raise SystemExit("No prompts matched --labels")
    if args.dry_run:
        pool_rows = [row for row in prompts if row["pool_eligible"].lower() == "true"]
        print(f"DRY RUN: {len(prompts)} prompts, {len(pool_rows)} pool-eligible, "
              f"takes={args.approved_takes}, pool_root={args.pool_root / args.speaker_id}")
        return

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit("Install recording dependencies: python -m pip install numpy sounddevice") from exc
    if args.list_devices:
        print(sd.query_devices())
        return

    device = args.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    model = None
    if args.whisper_model:
        try:
            from pywhispercpp.model import Model
        except ImportError as exc:
            raise SystemExit(
                "Whisper validation requested. Install it with: python -m pip install pywhispercpp"
            ) from exc
        print(f"Loading whisper model '{args.whisper_model}'...")
        model = Model(args.whisper_model, redirect_whispercpp_logs_to=False)

    local_dir = args.local_root / args.speaker_id / args.session_id
    local_manifest = local_dir / "recordings.csv"
    pool_dir = args.pool_root / args.speaker_id
    pool_manifest = pool_dir / "manifest.csv"

    pool_existing = load_rows(pool_manifest)
    done_counts: dict[str, int] = {}
    for row in load_rows(local_manifest):
        done_counts[row["prompt_id"]] = done_counts.get(row["prompt_id"], 0) + 1
    if args.resume:
        before = len(prompts)
        prompts = [p for p in prompts if done_counts.get(p["prompt_id"], 0) < args.approved_takes]
        print(f"--resume: skipping {before - len(prompts)} completed prompt(s)")
    if not prompts:
        print("Nothing left to record for this session.")
        return

    print(f"Speaker {args.speaker_id} | session {args.session_id} | split {args.session_split}")
    print(f"{len(prompts)} prompts x {args.approved_takes} take(s); "
          f"{len(pool_existing)} pool rows already present; pool write: "
          f"{'off' if args.no_pool else 'on'}\n")

    new_pool_rows: list[dict[str, object]] = []
    new_local_rows: list[dict[str, object]] = []
    for index, prompt in enumerate(prompts, start=1):
        for take_index in range(1, args.approved_takes + 1):
            take = args.take_offset + take_index
            expected = normalize_words(prompt["text"])
            print("=" * 72)
            print(f"[{index}/{len(prompts)}] ({prompt['label']}) take {take}")
            print(f'    say: "{prompt["text"]}"')
            while True:
                command = input("  [Enter=record, s=skip, q=quit] > ").strip().lower()
                if command == "q":
                    print("Stopping. Rerun with the same arguments to continue.")
                    break
                if command == "s":
                    break
                print("  RECORDING")
                audio = sd.rec(
                    int(args.duration * sample_rate_hz), samplerate=sample_rate_hz,
                    channels=1, dtype="int16", device=device,
                )
                sd.wait()
                frames = audio.tobytes()
                transcript, wer = "", 1.0
                if model is not None:
                    print("  Transcribing...")
                    segments = model.transcribe(audio.flatten(), print_progress=False)
                    transcript = " ".join(seg.text for seg in segments).strip()
                    if transcript in ("[BLANK_AUDIO]", "[SILENCE]", "[NO SPEECH]"):
                        transcript = ""
                    wer = word_error_rate(expected, normalize_words(transcript)) if transcript else 1.0
                    print(f'    whisper: "{transcript or "(nothing heard)"}" (wer={wer:.2f})')
                else:
                    print("    (no whisper model passed: transcript blank, wer recorded as 1.000)")
                decision = input("  [Enter=keep, r=redo, s=skip, q=quit] > ").strip().lower()
                if decision == "q":
                    break
                if decision == "s":
                    break
                if decision == "r":
                    continue

                now = datetime.now(UTC)
                is_pool = prompt["pool_eligible"].lower() == "true" and not args.no_pool
                pool_filename = f"{prompt['prompt_id']}_t{take}.wav"
                if is_pool and (pool_dir / pool_filename).exists():
                    print(f"  WARNING: pool file {pool_filename} already exists; "
                          "keeping the local take only. Use a different --take-offset "
                          "for this session to avoid overwriting pooled audio.")
                    is_pool = False
                if is_pool:
                    write_pcm16(pool_dir / pool_filename, frames, sample_rate_hz)

                sample_id = stable_id(
                    "personal_awi", args.speaker_id, args.session_id, prompt["prompt_id"], take
                )
                local_path = local_dir / f"{sample_id}.wav"
                write_pcm16(local_path, frames, sample_rate_hz)

                slot_value = prompt.get("slot_value", "") or ""
                prompt_type = prompt.get("type", "fixed")
                local_row = {column: "" for column in MANIFEST_COLUMNS}
                local_row.update({
                    "sample_id": sample_id,
                    "schema_version": "2.0.0",
                    "corpus_id": "personal_awi",
                    "relative_path": local_path.as_posix(),
                    "sha256": hashlib.sha256(local_path.read_bytes()).hexdigest(),
                    "speaker_id": args.speaker_id,
                    "session_id": args.session_id,
                    "prompt_id": prompt["prompt_id"],
                    "phrase_family_id": prompt["prompt_id"].rsplit("_V", 1)[0],
                    "take_number": str(take),
                    "raw_transcript": prompt["text"],
                    "normalized_transcript": normalize_text(prompt["text"]),
                    "language": "en",
                    "locale": "en-PH",
                    "training_label": prompt["label"].upper(),
                    "leaf_label": prompt["label"].upper(),
                    "action": prompt["label"].upper(),
                    "slot_type": "slotted" if prompt_type == "slotted" else "none",
                    "slot_value": slot_value,
                    "slots_json": "{}",
                    "sample_type": {
                        "slotted": "command_slotted",
                        "local": "unknown_speech" if prompt["label"].upper() == "UNKNOWN" else "silence_noise",
                    }.get(prompt_type, "command"),
                    "source_split": args.session_split,
                    "recorded_at_utc": now.isoformat(),
                    "device_id": args.microphone,
                    "microphone": args.microphone,
                    "environment": args.environment,
                    "distance_cm": str(args.distance_cm),
                    "speaking_style": args.speaking_style,
                    "sample_rate_hz": str(sample_rate_hz),
                    "channels": "1",
                    "bit_depth": "16",
                    "duration_ms": str(round(args.duration * 1000)),
                    "parent_sample_id": "",
                    "is_augmented": "false",
                    "consent_id": f"consent_{args.speaker_id}_v1",
                    "license": "private-research",
                    "annotation_status": "whisper_validated" if model else "prompt_verified",
                })
                new_local_rows.append(local_row)

                if is_pool:
                    new_pool_rows.append({
                        "speaker_id": args.speaker_id,
                        "prompt_id": prompt["prompt_id"],
                        "label": prompt["label"].upper(),
                        "type": prompt_type,
                        "text": prompt["text"],
                        "slot_value": slot_value,
                        "take": take,
                        "filename": pool_filename,
                        "recorded_at": now.isoformat(timespec="seconds"),
                        "whisper_transcript": transcript,
                        "wer": f"{wer:.3f}",
                        "status": "approved",
                    })
                print(f"  saved {local_path.name}")
                break

    if new_local_rows:
        append_rows(local_manifest, MANIFEST_COLUMNS, new_local_rows)
        print(f"\nLocal: {len(new_local_rows)} rows -> {local_manifest}")
    if new_pool_rows:
        append_rows(pool_manifest, POOL_FIELDS, new_pool_rows)
        print(f"Pool:  {len(new_pool_rows)} rows -> {pool_manifest}")
        print(f"Upload the whole folder to the shared Drive raw/: {pool_dir}")
    if not new_local_rows:
        print("\nNo new recordings.")


if __name__ == "__main__":
    main()
