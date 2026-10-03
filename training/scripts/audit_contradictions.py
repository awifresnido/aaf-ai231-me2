#!/usr/bin/env python3
"""Step 1b contradiction audit (PHASE2 rev2 + FALLBACK addendum).

Audits composite_v0.csv against configs/command_phrases.csv + the approved
configs/label_overrides.csv, applying the relabel/drop resolution rules, and
writes composite_v0_audited.csv. Dropped rows go to logs/audit_dropped.csv.
Gate 1: re-running with --input on the audited output reports zero contradictions.

Usage:
  python scripts/audit_contradictions.py [--input data/manifests/composite_v0.csv]
      [--output data/manifests/composite_v0_audited.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vcm_data.common import normalize_phrase  # noqa: E402

AUDITABLE_CORPORA = {"fsc", "gsc_v2", "personal_awi"}

# rev2 §1b "keep as UNKNOWN" hard negatives: true negatives with no actionable
# value, NOT contradictions under the rules (normalized).
KEEP_UNKNOWN = {
    "increase the heat", "decrease the heat",
    "set a timer", "remind me in ten minutes",
    "red", "twenty degrees",
    "turn the lights off in ten minutes", "do not set an alarm",
    "the weather was nice yesterday",
    "go", "up", "down", "on", "off",  # GSC single words
}


def whole_word_contains(text: str, phrase: str) -> bool:
    """True if `phrase`'s words appear in order (gap allowed) in `text`.

    In-order subsequence, per rev2's example: "stop the music" contains "stop
    music" ("the" is a gap). Whole-word only: each phrase word must equal a text
    token, never a substring of one.
    """
    words = text.split()
    p = phrase.split()
    i = 0
    for w in words:
        if i < len(p) and w == p[i]:
            i += 1
    return i == len(p)


def load_phrases(path: Path):
    exact = {}
    multi = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ph = r["phrase"].strip()
            ck = r["class_key"].strip()
            exact[ph] = ck
            if len(ph.split()) >= 2:
                multi.append((ph, ck))
    multi.sort(key=lambda x: -len(x[0].split()))  # most specific first
    return exact, multi


def load_overrides(path: Path):
    ov = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ov[r["prompt_id"].strip()] = r["new_label"].strip()
    return ov


def apply_class(row: dict, class_key: str) -> dict:
    """Set label fields from a leaf class key ('INTENT' or 'INTENT|slot')."""
    if "|" in class_key:
        intent, slot = class_key.split("|", 1)
    else:
        intent, slot = class_key, ""
    row["training_label"] = intent
    row["leaf_label"] = intent
    row["action"] = intent
    row["slot_type"] = "slotted" if slot else "none"
    row["slot_value"] = slot
    row["sample_type"] = "command_slotted" if slot else "command"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/manifests/composite_v0.csv")
    ap.add_argument("--output", default="data/manifests/composite_v0_audited.csv")
    ap.add_argument("--phrases", default="configs/command_phrases.csv")
    ap.add_argument("--overrides", default="configs/label_overrides.csv")
    args = ap.parse_args()

    exact, multi = load_phrases(Path(args.phrases))
    overrides = load_overrides(Path(args.overrides))

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    kept, dropped = [], []
    relabeled = Counter()
    dropped_by_rule = Counter()
    relabeled_transcripts = Counter()
    dropped_transcripts = Counter()

    for row in rows:
        corpus = row["corpus_id"]
        if corpus not in AUDITABLE_CORPORA:
            kept.append(row)
            continue

        text = normalize_phrase(row.get("normalized_transcript") or row.get("raw_transcript") or "")
        if not text or text in KEEP_UNKNOWN:
            kept.append(row)
            continue

        current = row["training_label"]
        status = row.get("annotation_status", "")

        # 1) explicit override (personal UNKNOWN_xx) -- wins over rules
        if corpus == "personal_awi" and row.get("prompt_id") in overrides:
            new = overrides[row["prompt_id"]]
            if new == "DROP":
                row["annotation_status"] = f"{status}+dropped:explicit_override"
                dropped.append(row)
                dropped_transcripts[text] += 1
                dropped_by_rule[("explicit_override", corpus)] += 1
            else:
                apply_class(row, new)
                row["annotation_status"] = f"{status}+relabeled:explicit_override"
                kept.append(row)
                relabeled[("explicit_override", corpus)] += 1
                relabeled_transcripts[text] += 1
            continue

        # 2) exact equality with a phrase of a different intent -> relabel
        exact_ck = exact.get(text)
        if exact_ck is not None and exact_ck.split("|")[0] != current:
            apply_class(row, exact_ck)
            row["annotation_status"] = f"{status}+relabeled:exact_equality"
            kept.append(row)
            relabeled[("exact_equality", corpus)] += 1
            relabeled_transcripts[text] += 1
            continue

        # 3) containment of a multi-word phrase of a different intent -> drop
        hit = None
        for ph, ck in multi:
            if ck.split("|")[0] == current:
                continue
            if whole_word_contains(text, ph):
                hit = (ph, ck)
                break
        if hit:
            row["annotation_status"] = f"{status}+dropped:containment"
            dropped.append(row)
            dropped_transcripts[text] += 1
            dropped_by_rule[("containment", corpus)] += 1
            continue

        kept.append(row)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)

    Path("logs").mkdir(exist_ok=True)
    if dropped:
        with open("logs/audit_dropped.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(dropped)

    print(f"input rows: {len(rows)}")
    print(f"kept: {len(kept)} | dropped: {len(dropped)}")
    print("\n== relabeled counts (rule x corpus) ==")
    for k, v in sorted(relabeled.items()):
        print(f"  {k[0]:16s} {k[1]:14s} {v}")
    print("\n== dropped counts (rule x corpus) ==")
    for k, v in sorted(dropped_by_rule.items()):
        print(f"  {k[0]:16s} {k[1]:14s} {v}")
    print("\n== top relabeled transcripts ==")
    for t, n in relabeled_transcripts.most_common(20):
        print(f"  {n:5d}  {t!r}")
    print("\n== top dropped transcripts ==")
    for t, n in dropped_transcripts.most_common(20):
        print(f"  {n:5d}  {t!r}")
    print(f"\nwrote {args.output}")

    # Gate 1 check: no transcript maps to >1 leaf class key
    t2c = defaultdict(set)
    for r in kept:
        key = r["training_label"] + (("|" + (r.get("slot_value") or "").strip()) if (r.get("slot_value") or "").strip() else "")
        t2c[normalize_phrase(r.get("normalized_transcript") or r.get("raw_transcript") or "")].add(key)
    multi_map = {t: c for t, c in t2c.items() if len(c) > 1}
    if multi_map:
        print(f"\nGATE1 WARNING: {len(multi_map)} transcripts map to >1 class key")
        for t, c in sorted(multi_map.items())[:30]:
            print(f"  {t!r} -> {sorted(c)}")
    else:
        print("\nGATE1 PASS: every transcript maps to a single class key")


if __name__ == "__main__":
    main()
