#!/usr/bin/env python3
"""Extract near_miss (off-schema) audio + write a manifest-ready near_miss.csv.

The 158 off-schema clips live in the gold train parquet but are excluded from
me2_gold_v1.csv. This extracts their audio to data/external/me2_gold/near_miss/
and writes a CSV with relative_path + label (UNKNOWN) for the eval.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pyarrow.parquet as pq

TVCM = Path(__file__).resolve().parents[2]
ME2G = Path("ME2_GOLD")
SRC = TVCM / "reports/me2_gold_v1_near_miss.csv"
OUT = TVCM / "data/external/me2_gold/near_miss"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    # near_miss abs_path -> (parquet_name, basename)
    need: dict[str, tuple[str, str]] = {}
    for r in csv.DictReader(open(SRC)):
        parquet, basename = r["abs_path"].split("#", 1)
        need[basename] = (parquet, basename)

    written = 0
    for f in sorted(ME2G.rglob("*.parquet")):
        if ".cache" in f.parts:
            continue
        for rb in pq.ParquetFile(f).iter_batches(batch_size=1000):
            t = rb.to_pydict()
            audio = t.pop("audio")
            for a in audio:
                base = Path(a.get("path") or "").name
                if base in need:
                    (OUT / base).write_bytes(a["bytes"])
                    written += 1
    print(f"extracted {written} near_miss audio files to {OUT}")

    # write manifest-ready near_miss.csv
    rows = []
    for r in csv.DictReader(open(SRC)):
        basename = r["abs_path"].split("#", 1)[1]
        rows.append({
            "relative_path": f"data/external/me2_gold/near_miss/{basename}",
            "training_label": "UNKNOWN",
            "slot_value": "",
            "sha256": "",
            "speaker_id": r["speaker_id"],
        })
    with open(OUT / "near_miss.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["relative_path", "training_label",
                                          "slot_value", "sha256", "speaker_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT / 'near_miss.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
