"""Canonical manifest helpers for the tiny VCM dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import wave
from collections.abc import Iterable
from pathlib import Path

MANIFEST_COLUMNS = [
    "sample_id", "schema_version", "corpus_id", "relative_path", "sha256",
    "speaker_id", "session_id", "prompt_id", "phrase_family_id", "take_number",
    "raw_transcript", "normalized_transcript", "language", "locale",
    "training_label", "leaf_label", "action", "slot_type", "slot_value", "slots_json", "sample_type",
    "source_split", "recorded_at_utc", "device_id", "microphone", "environment",
    "distance_cm", "speaking_style", "sample_rate_hz", "channels", "bit_depth",
    "duration_ms", "parent_sample_id", "is_augmented", "consent_id", "license",
    "annotation_status",
]


def normalize_text(text: str) -> str:
    """Return a conservative lowercase transcript normalization."""
    return " ".join(text.lower().strip().split())


def stable_id(corpus_id: str, *parts: object) -> str:
    """Build a deterministic corpus-prefixed sample ID."""
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return f"{corpus_id}_{digest}"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Calculate a file SHA-256 without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def wav_metadata(path: Path) -> dict[str, int]:
    """Read lossless PCM WAV metadata using the Python standard library."""
    with wave.open(str(path), "rb") as audio:
        frames = audio.getnframes()
        sample_rate_hz = audio.getframerate()
        return {
            "sample_rate_hz": sample_rate_hz,
            "channels": audio.getnchannels(),
            "bit_depth": audio.getsampwidth() * 8,
            "duration_ms": round(frames * 1000 / sample_rate_hz),
        }


def empty_record() -> dict[str, str]:
    """Return an empty record with every canonical field present."""
    return {column: "" for column in MANIFEST_COLUMNS}


def write_manifest(path: Path, rows: Iterable[dict[str, object]]) -> int:
    """Write canonical rows with deterministic column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in MANIFEST_COLUMNS})
            count += 1
    temporary_path.replace(path)
    return count


def load_ontology(path: Path) -> tuple[set[str], dict[str, str]]:
    """Return valid training labels and leaf-to-training mapping."""
    data = json.loads(path.read_text(encoding="utf-8"))
    leaf_to_training: dict[str, str] = {}
    for training_label, leaves in data["training_labels"].items():
        for leaf in leaves:
            if leaf in leaf_to_training:
                raise ValueError(f"Duplicate leaf label in ontology: {leaf}")
            leaf_to_training[leaf] = training_label
    return set(data["training_labels"]), leaf_to_training


def load_slot_values(path: Path) -> dict[str, list[str]]:
    """Return the closed slot value set for each slotted intent."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {label: list(values) for label, values in data.get("slot_values", {}).items()}



def check_intent_contract(intent_mapping: dict[str, int], ontology_path: Path) -> None:
    """Fail fast if an intent mapping has drifted from the canonical ontology.

    Guards the exact bug behind MODEL_DATASET_LABEL_DISCREPANCY.md: a mapping
    built from a hardcoded n_intents literal that no longer matches
    configs/ontology.json. Raises ValueError if the class count differs or the
    label set differs.
    """
    valid_labels, _ = load_ontology(ontology_path)
    if len(intent_mapping) != len(valid_labels):
        raise ValueError(
            f"n_intents drift: mapping has {len(intent_mapping)} classes, "
            f"ontology.json has {len(valid_labels)}"
        )
    if set(intent_mapping) != valid_labels:
        missing = sorted(valid_labels - set(intent_mapping))
        extra = sorted(set(intent_mapping) - valid_labels)
        raise ValueError(
            f"intent mapping does not match ontology.json "
            f"(missing={missing}, extra={extra})"
        )


def normalize_phrase(text: str) -> str:
    """Normalize a spoken phrase for command matching (Step 1 audit).

    lowercase, collapse "6:00 am" -> "6 am", strip punctuation, collapse ws.
    """
    import re
    t = (text or "").lower()
    t = re.sub(r"(\d{1,2}):(\d{2})\s*(am|pm)\b", r"\1 \3", t)
    t = re.sub(r"(\d{1,2}):(\d{2})\b", r"\1 \2", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())
