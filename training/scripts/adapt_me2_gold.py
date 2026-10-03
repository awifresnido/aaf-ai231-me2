#!/usr/bin/env python3
"""Part B: build the me2_gold_v1 dataset from ME2 Gold parquet + Awi recordings.

Writes:
  data/manifests/me2_gold_v1.csv          (composite schema)
  reports/me2_gold_v1.json                (per-split/source stats + G1-G4 gates)
  reports/me2_gold_v1_drops.csv           (every dropped row + reason)
  data/external/me2_gold/<split>/...wav   (audio extracted from parquet)

Run (CPU): tools/.venv/bin/python scripts/adapt_me2_gold.py [--extract]

Label rules (Section 5):
  in-schema  -> leaf COMMAND|slot_value / COMMAND
  OUT_OF_SCOPE speech + group_synthetic_oos -> UNKNOWN
  empty-transcript noise -> SILENCE
  synthetic_negatives: noise_only/near_silence -> SILENCE; babble/reversed/truncated -> UNKNOWN

Excluded from training/validation:
  90 relabelled group recordings (real_voice + out_of_scope, note="out-of-scope sample")
  off-schema slot values (bucket "other slot value") -> near_miss slice
  listen_other_command.csv rows
  all of supplemental_synth

Splits: train = gold train (after exclusions) + Awi s01 x4 + negatives;
  validation = gold test + synthetic_negatives:test + Awi s02;
  final test = gold holdout + Awi s03.
"""
from __future__ import annotations

import argparse
import csv
import os
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

TVCM = Path(__file__).resolve().parents[2]
ME2G = Path(os.environ.get("ME2_GOLD_DIR", str(Path(__file__).resolve().parents[2] / "data" / "external" / "me2_gold")))
AUDIT = ME2G / "_audit"

ONT = json.loads((TVCM / "configs/ontology.json").read_text())
FIXED = set(ONT["labels"]["fixed"])
SLOTTED = set(ONT["labels"]["slotted"])
SLOT_VALUES = ONT["slot_values"]

# gold split -> target split (test -> validation, holdout -> final test)
SPLIT_MAP = {"train": "train", "test": "validation", "holdout": "test"}

# gold source -> corpus_id (preserves enough granularity for the eval slices)
PARAPHRASE_SOURCES = {
    "SLURP", "SNIPS", "FluentSpeechCommands", "TimersAndSuch",
    "xela_Multi-Sensor", "xela_SET_TEMPERATURE_REAL",
}
SLOT_NUMBERS = {"6", "8", "9", "10", "18", "20", "22", "26", "30", "60", "100"}

COLS = ["sample_id", "schema_version", "corpus_id", "relative_path", "sha256",
        "speaker_id", "session_id", "prompt_id", "phrase_family_id", "take_number",
        "raw_transcript", "normalized_transcript", "language", "locale",
        "training_label", "leaf_label", "action", "slot_type", "slot_value",
        "slots_json", "sample_type", "source_split", "recorded_at_utc", "device_id",
        "microphone", "environment", "distance_cm", "speaking_style",
        "sample_rate_hz", "channels", "bit_depth", "duration_ms", "parent_sample_id",
        "is_augmented", "consent_id", "license", "annotation_status"]

SEED = 0
UNK_TARGET = 0.18          # midpoint of the 15-22 % band
UNK_LO, UNK_HI = 0.15, 0.22
SLOT_WEIGHT = 3.0          # weight multiplier for slot numbers in MLEnd sampling


def empty_row() -> dict:
    return {c: "" for c in COLS}


def gold_leaf(command: str, slot: str) -> tuple[str, str, str]:
    """(training_label, slot_value, type) for an in-schema gold row."""
    if command in FIXED:
        return command, "", "fixed"
    if command in SLOTTED:
        if slot not in SLOT_VALUES.get(command, []):
            raise SystemExit(f"off-schema slot {slot!r} for {command}")
        return command, slot, "slotted"
    raise SystemExit(f"in-schema gold command {command!r} not in ontology")


def corpus_of(source: str, oos: int) -> str:
    if oos:
        return "gold_oos"
    if source == "real_voice":
        return "gold_real"
    if source == "group_synthetic":
        return "gold_synthetic"
    if source in PARAPHRASE_SOURCES:
        return "gold_paraphrase"
    return "gold_other"


def meta_json(source: str, orig_split: str, is_synthetic: str, oos: str, neg_kind: str) -> str:
    return json.dumps({"gold_source": source, "gold_split": orig_split,
                       "is_synthetic": is_synthetic, "out_of_scope": oos,
                       "neg_kind": neg_kind})


def _gold_row(r: pd.Series, lbl: str, slot: str, typ: str, corpus: str) -> dict:
    d = empty_row()
    new_split = SPLIT_MAP[r["split"]]
    fn = r["abs_path"].split("#", 1)[1]
    d["sample_id"] = "gold_" + r["sha256"][:16]
    d["schema_version"] = "2.0.0"
    d["corpus_id"] = corpus
    d["relative_path"] = f"data/external/me2_gold/{new_split}/{fn}"
    d["sha256"] = r["sha256"]
    d["speaker_id"] = r["speaker_id"]
    d["session_id"] = r["split"]
    d["raw_transcript"] = r["transcript"]
    d["normalized_transcript"] = r["transcript"].lower()
    d["language"] = "en"
    d["locale"] = "en"
    d["training_label"] = lbl
    d["leaf_label"] = lbl
    d["action"] = lbl if typ in ("fixed", "slotted") else ""
    d["slot_type"] = {"fixed": "none", "slotted": "slotted", "unknown": "", "silence": ""}[typ]
    d["slot_value"] = slot
    d["slots_json"] = meta_json(r["source"], r["split"], r["is_synthetic"],
                                r["out_of_scope"], r["neg_kind"])
    d["sample_type"] = {"fixed": "command", "slotted": "command",
                        "unknown": "negative", "silence": "silence"}[typ]
    d["source_split"] = new_split
    d["sample_rate_hz"] = "16000"
    d["channels"] = "1"
    d["bit_depth"] = "16"
    d["duration_ms"] = str(int(float(r["duration_s"]) * 1000)) if r["duration_s"] else ""
    d["is_augmented"] = "false"
    d["annotation_status"] = "mapped_from_gold"
    return d


def neg_row(corpus: str, rel: str, sha: str, speaker: str, split: str,
            lbl: str = "UNKNOWN", meta: str = "{}") -> dict:
    d = empty_row()
    d["sample_id"] = f"{corpus}_{sha[:16]}"
    d["schema_version"] = "2.0.0"
    d["corpus_id"] = corpus
    d["relative_path"] = rel
    d["sha256"] = sha
    d["speaker_id"] = speaker
    d["language"] = "en"
    d["training_label"] = lbl
    d["leaf_label"] = lbl
    d["slots_json"] = meta
    d["sample_type"] = "negative" if lbl == "UNKNOWN" else "silence"
    d["source_split"] = split
    d["sample_rate_hz"] = "16000"
    d["channels"] = "1"
    d["bit_depth"] = "16"
    d["is_augmented"] = "false"
    return d


def _awi_row(r: pd.Series, split: str, sha: str) -> dict:
    d = empty_row()
    d["sample_id"] = "awi_" + r["filename"].replace(".wav", "")
    d["schema_version"] = "2.0.0"
    d["corpus_id"] = "personal_awi"
    d["relative_path"] = f"data/external/personal_awi/{r['filename']}"
    d["sha256"] = sha
    d["speaker_id"] = "202453069"
    d["session_id"] = r["session_id"]
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
    d["sample_type"] = {"slotted": "command", "fixed": "command",
                        "unknown": "negative", "silence": "silence"}[r["type"]]
    d["source_split"] = split
    d["sample_rate_hz"] = "16000"
    d["channels"] = "1"
    d["bit_depth"] = "16"
    d["is_augmented"] = "false"
    d["consent_id"] = "consent_awi01_v1"
    d["license"] = "private-research"
    d["annotation_status"] = "approved"
    return d


def _extract_audio(rows: list[dict]) -> None:
    """Write gold audio from parquet to data/external/me2_gold/<split>/.

    Keyed by (orig_split, basename) because the synthetic_negatives train/test
    parquets reuse basenames for different content.
    """
    need = {}  # (orig_split, basename) -> target subdir
    for r in rows:
        if r["relative_path"].startswith("data/external/me2_gold/"):
            p = r["relative_path"].split("/")
            subdir, basename = p[3], p[4]
            try:
                orig = json.loads(r["slots_json"]).get("gold_split", "")
            except Exception:
                orig = ""
            need[(orig, basename)] = subdir
    written = 0
    for f in sorted(ME2G.rglob("*.parquet")):
        if ".cache" in f.parts:
            continue
        if f.parent.name == "data":
            orig_split = f.name.split("-")[0]
        else:
            orig_split = f"{f.parent.name}:{f.name.split('-')[0]}"
        for rb in pq.ParquetFile(f).iter_batches(batch_size=1000):
            t = rb.to_pydict()
            audio = t.pop("audio")
            for a in audio:
                fn = a.get("path") or ""
                base = Path(fn).name
                if (orig_split, base) in need:
                    out_dir = TVCM / "data/external/me2_gold" / need[(orig_split, base)]
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / base).write_bytes(a["bytes"])
                    written += 1
    print(f"extracted {written} gold audio files")


def _gate_report(rows: list[dict], drops: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    # G1: no file with two labels; no "stop" labelled UNKNOWN
    dup = df.groupby("sha256")["training_label"].nunique()
    g1_contradictions = int((dup > 1).sum())
    g1_stop_unknown = int(((df["training_label"] == "UNKNOWN") &
                           (df["raw_transcript"].str.lower().str.strip() == "stop")).sum())
    # G1 exclusions applied: no relabelled/other_command/supplemental in manifest
    excl_reasons = Counter(d["reason"] for d in drops)

    # G2: all 33 leaves in train and validation
    all_leaves = set()
    for label in sorted(FIXED | SLOTTED):
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

    # G3: UNKNOWN 15-22 % of train; SILENCE >= 100
    train = df[df["source_split"] == "train"]
    unk_share = float((train["training_label"] == "UNKNOWN").mean())
    silence_n = int((train["training_label"] == "SILENCE").sum())
    g3_balance = (UNK_LO <= unk_share <= UNK_HI) and silence_n >= 100

    # G4: no speaker in two of train/validation/test; no sha256 shared train<->eval
    spk = df[(df["speaker_id"] != "202453069") & (df["speaker_id"] != "")]
    spk_splits = spk.groupby("speaker_id")["source_split"].agg(lambda s: set(s))
    g4_speaker_leak = {k: sorted(v) for k, v in spk_splits.items() if len(v) > 1}
    eval_sha = set(df[df["source_split"].isin(["validation", "test"])]["sha256"])
    train_sha = set(df[df["source_split"] == "train"]["sha256"])
    g4_sha_shared = sorted(train_sha & eval_sha)[:20]

    gates = {
        "G1_contradictions": {"no_file_two_labels": g1_contradictions == 0,
                              "no_stop_in_unknown": g1_stop_unknown == 0,
                              "exclusions_applied": dict(excl_reasons)},
        "G2_coverage": {"train_missing": g2_train_missing, "val_missing": g2_val_missing,
                        "pass": not g2_train_missing and not g2_val_missing},
        "G3_balance": {"unk_share": round(unk_share, 4), "silence_n": silence_n, "pass": g3_balance},
        "G4_disjointness": {"speaker_leak": g4_speaker_leak, "sha_shared": g4_sha_shared,
                            "pass": not g4_speaker_leak and not g4_sha_shared},
    }

    def real_vs_synthetic(sub):
        real = synthetic = 0
        for _, r in sub.iterrows():
            try:
                m = json.loads(r["slots_json"])
            except Exception:
                m = {}
            if m.get("is_synthetic") == "1" or r["corpus_id"] in ("gold_synthetic", "synthetic_negative", "gold_numerals"):
                synthetic += 1
            else:
                real += 1
        return real, synthetic

    def neg_kind_of(r):
        try:
            return json.loads(r["slots_json"]).get("neg_kind", "") or ""
        except Exception:
            return ""

    report = {"gates": gates, "splits": {}, "totals": {"rows": len(df)},
              "manifest_sha256": None}
    for split, sub in df.groupby("source_split"):
        real, synthetic = real_vs_synthetic(sub)
        # per-source real/synthetic
        per_source_rs = {}
        for cid, g in sub.groupby("corpus_id"):
            rs = real_vs_synthetic(g)
            per_source_rs[cid] = {"clips": int(len(g)), "real": rs[0], "synthetic": rs[1]}
        # UNKNOWN / SILENCE by source (corpus) and neg_kind
        unk = sub[sub["training_label"] == "UNKNOWN"]
        sil = sub[sub["training_label"] == "SILENCE"]
        unk_by_src = unk.groupby("corpus_id").size().to_dict()
        sil_by_src = sil.groupby("corpus_id").size().to_dict()
        unk_by_kind = {}
        sil_by_kind = {}
        for _, r in unk.iterrows():
            nk = neg_kind_of(r)
            unk_by_kind[nk] = unk_by_kind.get(nk, 0) + 1
        for _, r in sil.iterrows():
            nk = neg_kind_of(r)
            sil_by_kind[nk] = sil_by_kind.get(nk, 0) + 1
        # per-leaf and real-voice per-leaf
        rv = sub[sub["corpus_id"].isin(["gold_real", "personal_awi"])]
        def leaf_of(r):
            if r["training_label"] in SLOTTED and r["slot_value"]:
                return f"{r['training_label']}|{r['slot_value']}"
            return r["training_label"]
        per_leaf = Counter(leaf_of(r) for _, r in sub.iterrows())
        rv_per_leaf = Counter(leaf_of(r) for _, r in rv.iterrows())
        report["splits"][split] = {
            "clips": int(len(sub)),
            "hours": round(float(pd.to_numeric(sub["duration_ms"], errors="coerce").fillna(0).sum()) / 3.6e6, 3),
            "unique_speakers": int(sub["speaker_id"].nunique()),
            "real": real, "synthetic": synthetic,
            "per_source": per_source_rs,
            "per_label": sub.groupby("training_label").size().to_dict(),
            "unknown_by_source": {k: int(v) for k, v in unk_by_src.items()},
            "silence_by_source": {k: int(v) for k, v in sil_by_src.items()},
            "unknown_by_neg_kind": unk_by_kind,
            "silence_by_neg_kind": sil_by_kind,
            "per_leaf": dict(per_leaf),
            "real_voice_per_leaf": dict(rv_per_leaf),
        }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--extract", action="store_true", help="write WAVs from parquet")
    ap.add_argument("--unk-target", type=float, default=UNK_TARGET)
    ap.add_argument("--out-manifest", default=str(TVCM / "data/manifests/me2_gold_v1.csv"))
    args = ap.parse_args()
    RNG = random.Random(args.seed)

    st = pd.read_csv(AUDIT / "audio_stats.csv", dtype=str, keep_default_na=False)
    st["out_of_scope"] = st["out_of_scope"].astype(int)

    # ---- drop sets (gold policy: relabelled, other_command, supplemental) ------
    drops: dict[str, str] = {}
    # only listen_other_command is excluded (no "not_understood" exclusion in gold)
    df = pd.read_csv(AUDIT / "listen_other_command.csv", dtype=str, keep_default_na=False)
    for p in df["abs_path"]:
        drops[p] = "other_command"
    # relabelled group recordings: real_voice + out_of_scope (note "out-of-scope sample")
    relabelled = set(st[(st["source"] == "real_voice") &
                        (st["out_of_scope"] == 1)]["abs_path"])
    for p in relabelled:
        drops[p] = "relabelled_group_recording"
    # supplemental_synth -> excluded entirely
    supplemental = set(st[st["split"] == "supplemental_synth:train"]["abs_path"])
    for p in supplemental:
        drops[p] = "supplemental_synth"

    rows: list[dict] = []
    near_miss: list[dict] = []
    dropped: list[dict] = []

    # ---- gold rows: label + split + drop -------------------------------------
    for _, r in st.iterrows():
        if r["split"] in ("numerals", "supplemental_synth:train",
                         "synthetic_negatives:train", "synthetic_negatives:test"):
            continue  # numerals = negative pool; supplemental excluded; synthetic_negatives handled separately
        abs_path = r["abs_path"]
        if r["bucket"] and "other slot value" in r["bucket"]:
            near_miss.append({"abs_path": abs_path, "split": r["split"],
                              "command": r["command"], "slot_value": r["slot_value"],
                              "transcript": r["transcript"], "speaker_id": r["speaker_id"]})
            continue
        if abs_path in drops:
            dropped.append({"abs_path": abs_path, "split": r["split"],
                            "reason": drops[abs_path], "command": r["command"],
                            "transcript": r["transcript"]})
            continue
        if int(r["out_of_scope"]) == 1:
            if r["transcript"].strip() == "":
                lbl, slot, typ = "SILENCE", "", "silence"
            else:
                lbl, slot, typ = "UNKNOWN", "", "unknown"
        else:
            lbl, slot, typ = gold_leaf(r["command"], r["slot_value"])
        rows.append(_gold_row(r, lbl, slot, typ, corpus_of(r["source"], int(r["out_of_scope"]))))

    # ---- synthetic_negatives (train -> train, test -> validation) -------------
    for _, r in st[st["split"].isin(["synthetic_negatives:train", "synthetic_negatives:test"])].iterrows():
        nk = r["neg_kind"]
        if nk in ("noise_only", "near_silence"):
            lbl, typ = "SILENCE", "silence"
            speaker = ""  # noise/silence has no real speaker
        else:  # babble, reversed, truncated
            lbl, typ = "UNKNOWN", "unknown"
            speaker = r["speaker_id"]
        target = "train" if r["split"] == "synthetic_negatives:train" else "validation"
        fn = r["abs_path"].split("#", 1)[1]
        meta = meta_json("synthetic_negative", r["split"], "1", "1", nk)
        d = neg_row("synthetic_negative", f"data/external/me2_gold/{target}/{fn}",
                    r["sha256"], speaker, target, lbl, meta)
        d["duration_ms"] = str(int(float(r["duration_s"]) * 1000)) if r["duration_s"] else ""
        rows.append(d)

    # ---- Awi rows (s01 train, s02 validation, s03 test) -----------------------
    awi_root = TVCM / "data/external/personal_awi"
    awi = pd.read_csv(awi_root / "manifest.csv", dtype=str, keep_default_na=False)
    awi_map = pd.read_csv(awi_root / "awi01_mapping.csv", dtype=str, keep_default_na=False)
    awi_split = dict(zip(awi_map["new_filename"], awi_map["source_split"]))
    awi_sha = dict(zip(awi_map["new_filename"], awi_map["sha256"]))
    awi_sess = dict(zip(awi_map["new_filename"], awi_map["session_id"]))
    for _, r in awi.iterrows():
        fn = r["filename"]
        split = awi_split.get(fn, "")
        if split not in ("train", "validation", "test"):
            continue
        r = r.copy()
        r["session_id"] = awi_sess.get(fn, "")
        rows.append(_awi_row(r, split, awi_sha.get(fn, "")))

    # ---- repeat Awi s01 (train) x4 -------------------------------------------
    awi_train = [r for r in rows if r["corpus_id"] == "personal_awi" and r["source_split"] == "train"]
    for r in awi_train:
        for k in range(1, 4):
            c = dict(r)
            c["sample_id"] = f"{r['sample_id']}_rep{k}"
            rows.append(c)

    # ---- MLEnd numerals (UNKNOWN negatives) to hit the target share -----------
    mle = st[(st["split"] == "numerals") & (st["source"] == "MLEnd_numerals")]
    mle = mle[~mle["near_silent"].isin(["True", "1"])]
    mle = mle[mle["clip_frac"].astype(float) < 0.001]
    mle = mle.copy()
    mle["_w"] = mle["numerals"].apply(lambda n: SLOT_WEIGHT if n in SLOT_NUMBERS else 1.0)

    # solve N: (unk + N) / (total + N) = target
    cur = pd.DataFrame(rows)
    pos = int((cur["source_split"] == "train").sum()) - int(((cur["source_split"] == "train") &
                                                             (cur["training_label"].isin(["UNKNOWN", "SILENCE"]))).sum())
    unk = int(((cur["source_split"] == "train") & (cur["training_label"] == "UNKNOWN")).sum())
    sil = int(((cur["source_split"] == "train") & (cur["training_label"] == "SILENCE")).sum())
    N = max(0, int(round((args.unk_target * (pos + sil) - (1 - args.unk_target) * unk) / (1 - args.unk_target))))
    N = min(N, len(mle))
    mle_sel = mle.sample(N, weights="_w", random_state=args.seed)
    for _, r in mle_sel.iterrows():
        fn = r["abs_path"].split("#", 1)[1]
        meta = meta_json("MLEnd_numerals", "numerals", "0", "0", "")
        d = neg_row("gold_numerals", f"data/external/me2_gold/train/{fn}",
                    r["sha256"], r["speaker_id"], "train", "UNKNOWN", meta)
        d["duration_ms"] = str(int(float(r["duration_s"]) * 1000)) if r["duration_s"] else ""
        rows.append(d)

    # ---- write manifest -------------------------------------------------------
    rows.sort(key=lambda x: (x["source_split"], x["corpus_id"], x["relative_path"]))
    out = Path(args.out_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"manifest: {len(rows)} rows -> {out}")

    # ---- drops + near_miss report ---------------------------------------------
    rep_dir = TVCM / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    with open(rep_dir / "me2_gold_v1_drops.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["abs_path", "split", "reason", "command", "transcript"])
        w.writeheader()
        w.writerows(dropped)
    with open(rep_dir / "me2_gold_v1_near_miss.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["abs_path", "split", "command", "slot_value",
                                          "transcript", "speaker_id"])
        w.writeheader()
        w.writerows(near_miss)
    print(f"drops: {len(dropped)} dropped + {len(near_miss)} near_miss")

    # ---- audio extraction -----------------------------------------------------
    if args.extract:
        _extract_audio(rows)

    # ---- gates + report -------------------------------------------------------
    report = _gate_report(rows, dropped)
    report["manifest_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    (rep_dir / "me2_gold_v1.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ["gates", "totals"]}, indent=2, default=str))
    print(f"report -> reports/me2_gold_v1.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
