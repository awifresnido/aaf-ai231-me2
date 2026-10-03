#!/usr/bin/env python3
"""Step 3 (FALLBACK): build composite_v1.csv from audited manifests + Mark 16 kHz.

Merge: fsc + gsc_v2 + gsc_noise + personal_awi (all audited, from
composite_v0_audited.csv) + mark_optionb_16k.csv (clean + noisy, resampled).
Mark is replayed (replaces the old clean-only mark rows inside composite_v0).

Train-only transforms:
  * UNKNOWN balancing: cap train UNKNOWN at --unknown-ratio k x m, where m is the
    median train count over the 31 command leaf classes. Stratified for diversity
    (equal FSC/GSC quota; FSC even across phrase_family_id; GSC even across word
    and <= 2 clips/speaker), deterministic via sha256. Personal UNKNOWN and
    FSC-heat rows are always kept (hardest negatives).
  * personal oversampling x --personal-repeat (the demo speaker).

Val/test are NOT balanced (the false-accept metric needs the natural distribution).

Usage:
  python scripts/build_composite_v1.py [--unknown-ratio 3] [--personal-repeat 12]
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LEAF_SEP = "|"
SLOTTED = {"TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER"}
FIXED_COMMAND = {
    "PLAY_MUSIC", "WEATHER", "TIME", "LIGHT_ON", "LIGHT_OFF", "PAUSE", "STOP",
    "NEXT", "VOLUME_UP", "VOLUME_DOWN", "CALL", "MESSAGE", "LIST_REMINDERS",
}


def leaf_key(row: dict) -> str:
    label = row.get("training_label", "")
    slot = (row.get("slot_value") or "").strip()
    if slot and label in SLOTTED:
        return f"{label}{LEAF_SEP}{slot}"
    return label


def pick_stratified(rows, budget, keyfn, max_per_speaker=None):
    """Deterministic stratified selection up to `budget` rows (round-robin)."""
    if budget <= 0 or not rows:
        return []
    groups = defaultdict(list)
    for r in rows:
        groups[keyfn(r)].append(r)
    for g in groups:
        groups[g].sort(key=lambda r: r.get("sha256", ""))
        if max_per_speaker:
            ps = Counter()
            tmp = []
            for r in groups[g]:
                sp = r.get("speaker_id", "")
                if ps[sp] < max_per_speaker:
                    tmp.append(r)
                    ps[sp] += 1
            groups[g] = tmp
    keys = sorted(groups)
    idx = {k: 0 for k in keys}
    selected = []
    while len(selected) < budget:
        progress = False
        for k in keys:
            if idx[k] < len(groups[k]) and len(selected) < budget:
                selected.append(groups[k][idx[k]])
                idx[k] += 1
                progress = True
        if not progress:
            break
    return selected


def is_heat(row: dict) -> bool:
    t = (row.get("normalized_transcript") or row.get("raw_transcript") or "").lower()
    return any(w in t for w in ("heat", "temperature", "heating"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audited", default="data/manifests/composite_v0_audited.csv")
    ap.add_argument("--mark-16k", default="data/manifests/mark_optionb_16k.csv")
    ap.add_argument("--output", default="data/manifests/composite_v1.csv")
    ap.add_argument("--unknown-ratio", type=float, default=3.0)
    ap.add_argument("--personal-repeat", type=int, default=12)
    args = ap.parse_args()

    with open(args.audited, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        audited = list(reader)
    with open(args.mark_16k, newline="", encoding="utf-8") as f:
        mark = list(csv.DictReader(f))

    # drop old clean-only mark from audited; replay with 16k clean+noisy
    audited_no_mark = [r for r in audited if r["corpus_id"] != "mark_optionb"]
    merged = audited_no_mark + mark
    print(f"audited (excl mark): {len(audited_no_mark)} | mark_16k: {len(mark)} | merged: {len(merged)}")

    # ---- split ----
    train = [r for r in merged if r["source_split"] == "train"]
    val = [r for r in merged if r["source_split"] == "validation"]
    test = [r for r in merged if r["source_split"] == "test"]

    # ---- UNKNOWN balancing (train only) ----
    command_train = [r for r in train if r["training_label"] not in ("UNKNOWN", "SILENCE")]
    counts = Counter(leaf_key(r) for r in command_train)
    m = st.median(sorted(counts.values()))
    budget = int(round(args.unknown_ratio * m))
    print(f"command-leaf median train count m = {m:.1f} | UNKNOWN budget = k*m = {budget}")

    unk = [r for r in train if r["training_label"] == "UNKNOWN"]
    personal_unk = [r for r in unk if r["corpus_id"] == "personal_awi"]
    fsc_heat = [r for r in unk if r["corpus_id"] == "fsc" and is_heat(r)]
    fsc_rest = [r for r in unk if r["corpus_id"] == "fsc" and not is_heat(r)]
    gsc_unk = [r for r in unk if r["corpus_id"] == "gsc_v2"]
    print(f"UNKNOWN train: personal={len(personal_unk)} fsc_heat={len(fsc_heat)} "
          f"fsc_rest={len(fsc_rest)} gsc={len(gsc_unk)}")

    keep_unk = personal_unk + fsc_heat  # always kept (hardest negatives)
    # NOTE: keep_always (personal + FSC thermostat) exceeds k*m on its own, so the
    # k*m cap applies to the *sampled bulk* (GSC words + non-thermostat FSC),
    # split evenly between the two remaining sources.
    remaining = budget
    half = remaining // 2
    sel_fsc = pick_stratified(fsc_rest, half, keyfn=lambda r: r.get("phrase_family_id", r["normalized_transcript"]))
    sel_gsc = pick_stratified(gsc_unk, remaining - half,
                              keyfn=lambda r: r.get("normalized_transcript", ""), max_per_speaker=2)
    kept_unk = keep_unk + sel_fsc + sel_gsc
    kept_unk_ids = {id(r) for r in kept_unk}
    dropped_unk = [r for r in unk if id(r) not in kept_unk_ids]
    for r in sel_fsc + sel_gsc:
        r["annotation_status"] = f"{r.get('annotation_status','')}+unknown_balanced_keep"
    print(f"kept UNKNOWN: {len(kept_unk)} (personal {len(personal_unk)} + fsc_heat {len(fsc_heat)} "
          f"+ fsc {len(sel_fsc)} + gsc {len(sel_gsc)}) | dropped UNKNOWN: {len(dropped_unk)}")

    # ---- personal oversampling (train only, repeat-1 extra copies) ----
    personal_train = [r for r in train if r["corpus_id"] == "personal_awi"]
    extra_personal = []
    for r in personal_train:
        for i in range(1, args.personal_repeat):
            rr = dict(r)
            rr["sample_id"] = f"{r['sample_id']}_rep{i}"
            rr["annotation_status"] = f"{r.get('annotation_status','')}+oversampled_personal_x{args.personal_repeat}"
            extra_personal.append(rr)
    print(f"personal train: {len(personal_train)} x{args.personal_repeat} = {len(personal_train) * args.personal_repeat} rows")

    # ---- assemble train (non-UNKNOWN + kept UNKNOWN + extra personal copies) ----
    non_unk_train = [r for r in train if r["training_label"] != "UNKNOWN"]
    final_train = non_unk_train + kept_unk + extra_personal

    final = final_train + val + test
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(final)
    print(f"\nwrote {args.output}: train={len(final_train)} val={len(val)} test={len(test)} total={len(final)}")

    # ---- gate 3 report ----
    print("\n=== Gate 3 checks ===")
    leaf_train = Counter(leaf_key(r) for r in final_train)
    missing = [k for k in sorted(leaf_train) if leaf_train[k] < 1]
    # full 33 classes
    all_leaves = set(leaf_train)
    print(f"distinct leaf classes in train: {len(all_leaves)}")
    unknown_train = sum(1 for r in final_train if r["training_label"] == "UNKNOWN")
    print(f"UNKNOWN train share: {unknown_train}/{len(final_train)} = {unknown_train/len(final_train)*100:.2f}%")
    # real-speech (non-TTS) coverage per intent: FSC/GSC/personal command rows
    real = Counter()
    tts_only = set(FIXED_COMMAND) | set(SLOTTED)
    for r in final_train:
        if r["training_label"] in ("UNKNOWN", "SILENCE"):
            continue
        if r["corpus_id"] in ("fsc", "gsc_v2"):
            real[r["training_label"]] += 1
    covered_real = set(real)
    print(f"intents with non-TTS real-speech train coverage: {len(covered_real)} -> {sorted(covered_real)}")
    print(f"TTS-only intents (no FSC/GSC/personal command train rows): {sorted(tts_only - covered_real)}")
    # speaker split check (no speaker in >1 split)
    sp_split = defaultdict(set)
    for r in final:
        sp_split[r["speaker_id"]].add(r["source_split"])
    leak = {sp: ss for sp, ss in sp_split.items() if len(ss) > 1}
    print(f"speaker-split: {len(sp_split)} speakers | leaks={len(leak)}")
    if leak:
        for sp, ss in list(leak.items())[:10]:
            print(f"  LEAK {sp} -> {ss}")
    # per-corpus x split composition
    comp = Counter((r["corpus_id"], r["source_split"]) for r in final)
    print("\n=== composite_v1 composition (corpus x split) ===")
    for k, v in sorted(comp.items()):
        print(f"  {k[0]:14s} {k[1]:10s} {v}")


if __name__ == "__main__":
    main()
