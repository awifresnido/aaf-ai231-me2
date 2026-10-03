#!/usr/bin/env python3
"""Part 3: build the me2_master_v1 dataset from ME2 Master parquet + Awi + v1 negatives.

Writes:
  data/manifests/me2_master_v1.csv      (composite_v1 schema)
  reports/me2_master_v1_drops.csv       (every dropped row + reason)
  reports/me2_master_v1.json            (per-split/source stats + G1-G4 gate results)
  data/external/me2_master/<split>/...wav  (audio extracted from parquet)

Run: tools/.venv/bin/python scripts/adapt_me2_master.py   (needs pandas + pyarrow)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

TVCM = Path(__file__).resolve().parents[2]
ME2 = Path("~/projects/ME2_MASTER")
AUDIT = ME2 / "_audit"

ONT = json.loads((TVCM / "configs/ontology.json").read_text())
FIXED = set(ONT["labels"]["fixed"])
SLOTTED = set(ONT["labels"]["slotted"])
SLOT_VALUES = ONT["slot_values"]

# master split -> target split (test -> validation, holdout -> test; numerals is a pool)
SPLIT_MAP = {"train": "train", "test": "validation", "holdout": "test"}

COLS = ["sample_id", "schema_version", "corpus_id", "relative_path", "sha256",
        "speaker_id", "session_id", "prompt_id", "phrase_family_id", "take_number",
        "raw_transcript", "normalized_transcript", "language", "locale",
        "training_label", "leaf_label", "action", "slot_type", "slot_value",
        "slots_json", "sample_type", "source_split", "recorded_at_utc", "device_id",
        "microphone", "environment", "distance_cm", "speaking_style",
        "sample_rate_hz", "channels", "bit_depth", "duration_ms", "parent_sample_id",
        "is_augmented", "consent_id", "license", "annotation_status"]

SEED = 0
RNG = random.Random(SEED)


def empty_row() -> dict:
    return {c: "" for c in COLS}


def master_leaf(command: str, slot: str) -> str:
    """Return (training_label, slot_value, type) for an in-schema master row."""
    if command in FIXED:
        return command, "", "fixed"
    if command in SLOTTED:
        return command, slot, "slotted"
    raise ValueError(f"in-schema master command {command!r} not in ontology")


def sha256_of(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--extract", action="store_true", help="write WAVs from parquet")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--out-manifest", default=str(TVCM / "data/manifests/me2_master_v1.csv"))
    args = ap.parse_args()
    RNG.seed(args.seed)

    # ---------------- master metadata (audio_stats.csv has sha256 + stats) -----
    st = pd.read_csv(AUDIT / "audio_stats.csv", dtype=str, keep_default_na=False)
    st["out_of_scope"] = st["out_of_scope"].astype(int)
    print(f"master clips: {len(st)}  splits={dict(Counter(st['split']))}")

    # ---------------- drop lists (abs_path -> reason) ---------------------------
    drops: dict[str, str] = {}
    for fname, reason in [("listen_not_understood.csv", "not_understood"),
                          ("listen_other_command.csv", "other_command")]:
        df = pd.read_csv(AUDIT / fname, dtype=str, keep_default_na=False)
        for p in df["abs_path"]:
            drops[p] = reason
    # off-schema = bucket contains "other slot value"
    off_schema = st["bucket"].str.contains("other slot value")

    # ---------------- v1 negatives + sha256->split map for gsc digits ----------
    v1 = pd.read_csv(TVCM / "data/manifests/composite_v1.csv", dtype=str, keep_default_na=False)
    # full gsc_v2 (per-corpus) carries the digits; composite_v1.csv's gsc_v2 is a curated subset
    gsc_full = pd.read_csv(TVCM / "data/manifests/gsc_v2.csv", dtype=str, keep_default_na=False)
    gsc_v1_split = dict(zip(gsc_full["sha256"], gsc_full["source_split"]))
    gsc_v1_speaker = dict(zip(gsc_full["sha256"], gsc_full["speaker_id"]))
    gsc_v1_valtest_speakers = set(gsc_full[gsc_full["source_split"].isin(["validation", "test"])]["speaker_id"])

    # FSC speakers present in master test/holdout (exclude from fsc negatives)
    master_fsc_speakers = set(st[(st["split"].isin(["test", "holdout"])) &
                                 (st["source"] == "FluentSpeechCommands")]["speaker_id"])

    # ---------------- Awi sessions --------------------------------------------
    awi_root = TVCM / "data/personal/raw/202453069"
    awi = pd.read_csv(awi_root / "manifest.csv", dtype=str, keep_default_na=False)
    awi_map = pd.read_csv(awi_root / "awi01_mapping.csv", dtype=str, keep_default_na=False)
    awi_split = dict(zip(awi_map["new_filename"], awi_map["source_split"]))
    awi_sha = dict(zip(awi_map["new_filename"], awi_map["sha256"]))

    rows: list[dict] = []
    near_miss: list[dict] = []
    dropped: list[dict] = []

    # ---------------- master rows: label + drop --------------------------------
    for _, r in st.iterrows():
        if r["split"] == "numerals":
            continue  # numerals are only a negative pool, not a split
        abs_path = r["abs_path"]
        reason = drops.get(abs_path) or ("off_schema" if r["bucket"] and "other slot value" in r["bucket"] else "")
        if reason:
            rec = {"abs_path": abs_path, "split": r["split"], "reason": reason,
                   "command": r["command"], "transcript": r["transcript"]}
            if reason == "off_schema":
                near_miss.append(rec)
            else:
                dropped.append(rec)
            continue
        if int(r["out_of_scope"]) == 1:
            lbl, slot, typ = ("SILENCE", "", "silence") if r["transcript"].strip() == "" else ("UNKNOWN", "", "unknown")
        else:
            lbl, slot, typ = master_leaf(r["command"], r["slot_value"])
            if typ == "slotted" and slot not in SLOT_VALUES.get(r["command"], []):
                raise SystemExit(f"off-schema slot {slot!r} for {r['command']} at {abs_path}")
        rows.append(_master_row(r, lbl, slot, typ))

    # ---------------- negatives ------------------------------------------------
    unk_rows = _build_negatives(st, v1, gsc_v1_split, gsc_v1_speaker, gsc_v1_valtest_speakers,
                                master_fsc_speakers, RNG)
    rows.extend(unk_rows)

    # ---------------- Awi rows -------------------------------------------------
    for _, r in awi.iterrows():
        fn = r["filename"]
        split = {"train": "train", "validation": "validation", "test": "test"}[awi_split.get(fn, "")]
        rows.append(_awi_row(r, split, awi_sha.get(fn, "")))

    # ---------------- repeat Awi s01 x4 (train) --------------------------------
    awi_train = [r for r in rows if r["corpus_id"] == "personal_awi" and r["source_split"] == "train"]
    for r in awi_train:
        for k in range(1, 4):
            c = dict(r)
            c["sample_id"] = f"{r['sample_id']}_rep{k}"
            rows.append(c)

    # ---------------- write manifest -------------------------------------------
    rows.sort(key=lambda x: (x["source_split"], x["corpus_id"], x["relative_path"]))
    out = Path(args.out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"manifest: {len(rows)} rows -> {out}")

    # ---------------- drops report ---------------------------------------------
    drop_report = TVCM / "reports/me2_master_v1_drops.csv"
    drop_report.parent.mkdir(parents=True, exist_ok=True)
    all_drops = [{"abs_path": d["abs_path"], "split": d["split"], "reason": d["reason"],
                  "command": d["command"], "transcript": d["transcript"]}
                 for d in (dropped + near_miss)]
    with open(drop_report, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["abs_path", "split", "reason", "command", "transcript"])
        w.writeheader()
        w.writerows(all_drops)
    print(f"drops: {len(dropped)} dropped + {len(near_miss)} near_miss -> {drop_report}")

    # ---------------- audio extraction ------------------------------------------
    if not args.skip_extract:
        _extract_audio(rows, st, args.extract)

    # ---------------- gate checks + report -------------------------------------
    report = _gate_report(rows)
    rep_path = TVCM / "reports/me2_master_v1.json"
    rep_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"report -> {rep_path}")
    print(json.dumps({k: report[k] for k in ["gates", "splits"]}, indent=2, default=str))
    return 0


def _master_row(r: pd.Series, lbl: str, slot: str, typ: str) -> dict:
    d = empty_row()
    new_split = SPLIT_MAP[r["split"]]
    d["sample_id"] = "me2_" + r["sha256"][:16]
    d["schema_version"] = "2.0.0"
    d["corpus_id"] = "me2_real" if str(r["is_synthetic"]) == "0" else "me2_syn"
    d["relative_path"] = f"data/external/me2_master/{new_split}/{r['abs_path'].split('#',1)[1]}"
    d["sha256"] = r["sha256"]
    d["speaker_id"] = r["speaker_id"]
    d["session_id"] = r["split"]
    d["raw_transcript"] = r["transcript"]
    d["normalized_transcript"] = r["transcript"].lower()
    d["language"] = "en"
    d["locale"] = "en"
    d["training_label"] = lbl
    d["leaf_label"] = lbl
    d["action"] = lbl if typ not in ("unknown", "silence") else ""
    d["slot_type"] = {"fixed": "none", "slotted": "slotted", "unknown": "", "silence": ""}[typ]
    d["slot_value"] = slot
    d["slots_json"] = "{}"
    d["sample_type"] = {"fixed": "command", "slotted": "command", "unknown": "negative", "silence": "silence"}[typ]
    d["source_split"] = new_split
    d["sample_rate_hz"] = "16000"
    d["channels"] = "1"
    d["bit_depth"] = "16"
    d["duration_ms"] = str(int(float(r["duration_s"]) * 1000)) if r["duration_s"] else ""
    d["is_augmented"] = "false"
    d["annotation_status"] = "mapped_from_master"
    return d


def _build_negatives(st, v1, gsc_v1_split, gsc_v1_speaker, gsc_v1_valtest_speakers,
                     master_fsc_speakers, rng) -> list[dict]:
    out: list[dict] = []

    def neg_row(corpus, rel, sha, speaker, split="train") -> dict:
        d = empty_row()
        d["sample_id"] = f"{corpus}_{sha[:16]}"
        d["schema_version"] = "2.0.0"
        d["corpus_id"] = corpus
        d["relative_path"] = rel
        d["sha256"] = sha
        d["speaker_id"] = speaker
        d["language"] = "en"
        d["training_label"] = "UNKNOWN"
        d["leaf_label"] = "UNKNOWN"
        d["sample_type"] = "negative"
        d["source_split"] = split
        d["sample_rate_hz"] = "16000"
        d["channels"] = "1"
        d["bit_depth"] = "16"
        d["is_augmented"] = "false"
        return d

    # 1. MLEnd numerals (~900)
    mle = st[(st["split"] == "numerals") & (st["source"] == "MLEnd_numerals")]
    mle = mle[~mle["near_silent"].isin(["True", "1"])]
    mle = mle[mle["clip_frac"].astype(float) < 0.001]
    for _, r in mle.sample(min(900, len(mle)), random_state=rng.randrange(10**6)).iterrows():
        out.append(neg_row("me2_numerals",
                           f"data/external/me2_master/numerals/{r['abs_path'].split('#',1)[1]}",
                           r["sha256"], r["speaker_id"]))

    # 2. GSC digits (master numerals SpeechCommands_v2, v1-train only)
    gscd = st[(st["split"] == "numerals") & (st["source"] == "SpeechCommands_v2")]
    gscd = gscd[gscd["sha256"].map(gsc_v1_split) == "train"]
    gscd = gscd[~gscd["sha256"].map(gsc_v1_speaker).isin(gsc_v1_valtest_speakers)]
    for _, r in gscd.sample(min(600, len(gscd)), random_state=rng.randrange(10**6)).iterrows():
        out.append(neg_row("me2_numerals",
                           f"data/external/me2_master/numerals/{r['abs_path'].split('#',1)[1]}",
                           r["sha256"], r["speaker_id"]))

    # 3. v1 gsc_v2 non-command words (train)
    v1g = v1[(v1["corpus_id"] == "gsc_v2") & (v1["source_split"] == "train") &
             (v1["training_label"] == "UNKNOWN")]
    for _, r in v1g.iterrows():
        out.append(neg_row("gsc_v2", r["relative_path"], r["sha256"], r["speaker_id"]))

    # 4. v1 fsc + personal_awi UNKNOWN (train), exclude FSC speakers in master test/holdout
    v1u = v1[(v1["corpus_id"].isin(["fsc", "personal_awi"])) & (v1["source_split"] == "train") &
             (v1["training_label"] == "UNKNOWN")]
    v1_fsc_speaker = v1u["speaker_id"].str.replace("^fsc_", "", regex=True)
    v1u = v1u[~((v1u["corpus_id"] == "fsc") & (v1_fsc_speaker.isin(master_fsc_speakers)))]
    for _, r in v1u.sample(min(500, len(v1u)), random_state=rng.randrange(10**6)).iterrows():
        out.append(neg_row(r["corpus_id"], r["relative_path"], r["sha256"], r["speaker_id"]))

    # 5. SILENCE: v1 gsc_noise train + master silence (11) + Awi SILENCE s01
    def sil_row(corpus, rel, sha, speaker) -> dict:
        d = neg_row(corpus, rel, sha, speaker)
        d["training_label"] = "SILENCE"
        d["leaf_label"] = "SILENCE"
        d["sample_type"] = "silence"
        return d
    v1n = v1[(v1["corpus_id"] == "gsc_noise") & (v1["source_split"] == "train")]
    for _, r in v1n.iterrows():
        out.append(sil_row("gsc_noise", r["relative_path"], r["sha256"], r["speaker_id"]))
    # (master silence clips are already added by the master row loop -> SILENCE)

    return out


def _awi_row(r: pd.Series, split: str, sha: str) -> dict:
    d = empty_row()
    d["sample_id"] = "awi_" + r["filename"].replace(".wav", "")
    d["schema_version"] = "2.0.0"
    d["corpus_id"] = "personal_awi"
    d["relative_path"] = f"data/personal/raw/202453069/{r['filename']}"
    d["sha256"] = sha
    d["speaker_id"] = "202453069"
    d["session_id"] = r["filename"].rsplit("_t", 1)[0]
    d["prompt_id"] = r["prompt_id"]
    d["take_number"] = str(r["take"])
    d["raw_transcript"] = r["text"]
    d["normalized_transcript"] = r["text"].lower()
    d["language"] = "en"
    d["locale"] = "en"
    d["training_label"] = r["label"]
    d["leaf_label"] = r["label"]
    d["action"] = r["label"] if r["label"] not in ("UNKNOWN", "SILENCE") else ""
    d["slot_type"] = "slotted" if r["type"] == "slotted" else ("none" if r["type"] == "fixed" else "")
    d["slot_value"] = r["slot_value"]
    d["slots_json"] = "{}"
    d["sample_type"] = {"slotted": "command", "fixed": "command", "unknown": "negative", "silence": "silence"}[r["type"]]
    d["source_split"] = split
    d["sample_rate_hz"] = "16000"
    d["channels"] = "1"
    d["bit_depth"] = "16"
    d["is_augmented"] = "false"
    d["consent_id"] = "consent_awi01_v1"
    d["license"] = "private-research"
    d["annotation_status"] = "approved"
    return d


def _extract_audio(rows, st, force: bool) -> None:
    """Write master/numerals audio from parquet to data/external/me2_master/<subdir>/."""
    need = {}  # filename -> target subdir (train/validation/test/numerals)
    for r in rows:
        if r["relative_path"].startswith("data/external/me2_master/"):
            parts = r["relative_path"].split("/")
            need[parts[4]] = parts[3]
    written = 0
    for f in sorted(ME2.rglob("*.parquet")):
        if ".cache" in f.parts:
            continue
        for rb in pq.ParquetFile(f).iter_batches(batch_size=1000):
            t = rb.to_pydict()
            audio = t.pop("audio")
            for a, file_ in zip(audio, t["file"]):
                fn = a.get("path") or Path(file_).name
                if fn in need:
                    out_dir = TVCM / "data/external/me2_master" / need[fn]
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / fn).write_bytes(a["bytes"])
                    written += 1
    print(f"extracted {written} master audio files")


def _gate_report(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    by = df.groupby(["source_split", "corpus_id", "training_label"]).size().reset_index(name="n")

    # G1: contradictions
    dup_file = df.groupby("sha256")["training_label"].nunique()
    g1_contradictions = int((dup_file > 1).sum())
    g1_stop_unknown = int(((df["training_label"] == "UNKNOWN") &
                           (df["raw_transcript"].str.lower().str.strip() == "stop")).sum())

    # G2: coverage (all 33 leaves in train and validation)
    all_leaves = set()
    for label in sorted(set(FIXED) | set(SLOTTED)):
        if label in SLOTTED:
            all_leaves.update(f"{label}|{v}" for v in SLOT_VALUES[label])
        else:
            all_leaves.add(label)
    all_leaves |= {"UNKNOWN", "SILENCE"}
    def leaves_of(sub):
        s = set()
        for _, r in sub.iterrows():
            if r["training_label"] in SLOTTED and r["slot_value"]:
                s.add(f"{r['training_label']}|{r['slot_value']}")
            else:
                s.add(r["training_label"])
        return s
    train_leaves = leaves_of(df[df["source_split"] == "train"])
    val_leaves = leaves_of(df[df["source_split"] == "validation"])
    g2_train_missing = sorted(all_leaves - train_leaves)
    g2_val_missing = sorted(all_leaves - val_leaves)

    # G3: balance
    train = df[df["source_split"] == "train"]
    unk_share = float((train["training_label"] == "UNKNOWN").mean())
    silence_n = int((train["training_label"] == "SILENCE").sum())
    g3_balance = (0.15 <= unk_share <= 0.40) and silence_n >= 100

    # G4: disjointness (speakers + sha256)
    spk_splits = df[df["speaker_id"] != "202453069"].groupby("speaker_id")["source_split"].agg(lambda s: set(s))
    g4_speaker_leak = {k: sorted(v) for k, v in spk_splits.items() if len(v) > 1}
    eval_sha = set(df[df["source_split"].isin(["validation", "test"])]["sha256"])
    train_sha = set(df[df["source_split"] == "train"]["sha256"])
    g4_sha_shared = sorted(train_sha & eval_sha)[:20]

    gates = {
        "G1_contradictions": {"no_file_two_labels": g1_contradictions == 0,
                              "no_stop_in_unknown": g1_stop_unknown == 0},
        "G2_coverage": {"train_missing": g2_train_missing, "val_missing": g2_val_missing,
                        "pass": not g2_train_missing and not g2_val_missing},
        "G3_balance": {"unk_share": round(unk_share, 4), "silence_n": silence_n, "pass": g3_balance},
        "G4_disjointness": {"speaker_leak": g4_speaker_leak, "sha_shared": g4_sha_shared,
                            "pass": not g4_speaker_leak and not g4_sha_shared},
    }
    report = {"gates": gates, "splits": {}, "totals": {"rows": len(df)}}
    for split, sub in df.groupby("source_split"):
        report["splits"][split] = {
            "clips": int(len(sub)),
            "hours": round(float(pd.to_numeric(sub["duration_ms"], errors="coerce").fillna(0).sum()) / 3.6e6, 3),
            "unique_speakers": int(sub["speaker_id"].nunique()),
            "per_source": sub.groupby("corpus_id").size().to_dict(),
            "per_label": sub.groupby("training_label").size().to_dict(),
        }
    return report


if __name__ == "__main__":
    sys.exit(main())
