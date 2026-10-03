#!/usr/bin/env python3
"""PHASE2 Step 0b inventory. Manifest-only by default; --with-audio adds the
trimmed-duration (4-s window) analysis using the pipeline's own AudioProcessor.

Usage:
    python scripts/phase2_inventory.py [--with-audio] [--audio-root DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ONTOLOGY = Path("configs/ontology.json")
SLOTTED = {"TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER"}
MANIFESTS = {
    "composite_v0": "data/manifests/composite_v0.csv",
    "personal_awi": "data/manifests/personal_awi.csv",
    "mark_optionb": "data/manifests/mark_optionb.csv",
    "fsc": "data/manifests/fsc.csv",
    "gsc_v2": "data/manifests/gsc_v2.csv",
    "gsc_noise": "data/manifests/gsc_noise.csv",
}


def leaf_key(row: dict) -> str:
    label = row.get("training_label", "")
    slot = (row.get("slot_value") or "").strip()
    return f"{label}|{slot}" if slot else label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-audio", action="store_true")
    ap.add_argument("--audio-root", default=".")
    ap.add_argument("--out", default="logs/phase2_inventory.json")
    args = ap.parse_args()

    ontology = json.loads(ONTOLOGY.read_text())
    slot_values = ontology["slot_values"]
    report: dict = {"corpora": {}}

    for name, path in MANIFESTS.items():
        p = Path(path)
        if not p.exists():
            report["corpora"][name] = {"error": f"missing {path}"}
            continue
        with p.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        by_split = Counter(r.get("source_split", "") for r in rows)
        by_label_split = Counter(
            (leaf_key(r), r.get("source_split", "")) for r in rows
        )
        rates = Counter(r.get("sample_rate_hz", "") for r in rows)
        unknown = sum(1 for r in rows if r.get("training_label") == "UNKNOWN")
        # slot_value validity
        bad_slot = []
        for r in rows:
            lab = r.get("training_label", "")
            slot = (r.get("slot_value") or "").strip()
            if lab in SLOTTED:
                if not slot or slot not in slot_values.get(lab, []):
                    bad_slot.append((r.get("sample_id", ""), lab, slot))
        # FSC transcripts per label
        fsc_transcripts = None
        if name == "fsc":
            fsc_transcripts = defaultdict(set)
            for r in rows:
                fsc_transcripts[r.get("training_label", "")].add(
                    (r.get("normalized_transcript") or r.get("raw_transcript") or "").strip()
                )
        # personal session map + takes per prompt
        pers_sessions = None
        pers_takes = None
        if name == "personal_awi":
            pers_sessions = Counter(
                (r.get("session_id", ""), r.get("source_split", "")) for r in rows
            )
            pers_takes = Counter(r.get("prompt_id", "") for r in rows)
        report["corpora"][name] = {
            "rows": len(rows),
            "by_split": dict(sorted(by_split.items())),
            "by_leaf_split": {
                f"{k[0]} :: {k[1]}": v for k, v in sorted(by_label_split.items())
            },
            "sample_rate_hz": dict(sorted(rates.items())),
            "unknown_rows": unknown,
            "unknown_share": round(unknown / len(rows), 4) if rows else None,
            "bad_slot_rows": len(bad_slot),
            "bad_slot_examples": bad_slot[:20],
        }
        if fsc_transcripts is not None:
            report["corpora"][name]["fsc_transcripts_per_label"] = {
                k: sorted(v) for k, v in sorted(fsc_transcripts.items())
            }
        if pers_sessions is not None:
            report["corpora"][name]["session_split_map"] = {
                f"{k[0]} -> {k[1]}": v for k, v in sorted(pers_sessions.items())
            }
            report["corpora"][name]["takes_per_prompt"] = dict(
                sorted(pers_takes.items())
            )

    # is mark_optionb inside composite_v0?
    if Path(MANIFESTS["composite_v0"]).exists() and Path(MANIFESTS["mark_optionb"]).exists():
        with open(MANIFESTS["composite_v0"], newline="", encoding="utf-8") as fh:
            comp_ids = {r["sample_id"] for r in csv.DictReader(fh)}
        with open(MANIFESTS["mark_optionb"], newline="", encoding="utf-8") as fh:
            mark_ids = {r["sample_id"] for r in csv.DictReader(fh)}
        report["mark_optionb_in_composite_v0"] = {
            "mark_rows": len(mark_ids),
            "mark_rows_in_composite": len(mark_ids & comp_ids),
            "in_share": round(len(mark_ids & comp_ids) / len(mark_ids), 4) if mark_ids else None,
        }

    # ---- audio: trimmed duration on command rows (4-s window check) ----
    if args.with_audio:
        import torch
        from src.vcm_data_loader import AudioProcessor

        proc = AudioProcessor(sample_rate=16000, n_mels=40, n_fft=512,
                              hop_length=160, f_min=50, f_max=7600,
                              top_db=80.0, trim_db=40.0)
        dur: dict = defaultdict(list)
        for name in ("personal_awi", "fsc", "mark_optionb", "gsc_v2"):
            p = Path(MANIFESTS[name])
            if not p.exists():
                continue
            with p.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            # command rows only (not UNKNOWN/SILENCE)
            for r in rows:
                if r.get("training_label") in ("UNKNOWN", "SILENCE"):
                    continue
                try:
                    wav, sr = torchaudio_load(Path(args.audio_root) / r["relative_path"])
                    wav = proc.to_mono_16k(wav, sr)
                    wav = proc.trim_silence(wav)
                    d = wav.shape[-1] / 16000.0
                    dur[name].append(d)
                except Exception as e:  # noqa: BLE001
                    dur[f"{name}__error"].append(str(e))
        out_dur = {}
        for k, v in dur.items():
            if not v:
                out_dur[k] = {"n": 0}
                continue
            if k.endswith("__error"):
                out_dur[k] = {"n": len(v), "errors": v[:5]}
                continue
            over = sum(1 for d in v if d > 2.5)
            import statistics as st
            out_dur[k] = {
                "n": len(v),
                "mean_s": round(st.mean(v), 3),
                "median_s": round(st.median(v), 3),
                "p95_s": round(sorted(v)[min(len(v) - 1, int(0.95 * len(v)))], 3),
                "over_2.5s": over,
                "over_2.5s_pct": round(100.0 * over / len(v), 2),
            }
        report["trimmed_duration_command_rows"] = out_dur

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


def torchaudio_load(path: Path):
    import torchaudio
    return torchaudio.load(str(path))


if __name__ == "__main__":
    main()
