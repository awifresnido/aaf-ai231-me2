#!/usr/bin/env python
"""Build the shared, memory-mapped waveform cache used by VCMDataset.

Decodes every clip referenced by the given manifests ONCE, through the exact
same ``AudioProcessor.to_mono_16k`` the loader uses, and writes:

    <out>/waves.f32    contiguous float32 samples (all clips back to back)
    <out>/index.json   {"sample_rate", "total_samples", "manifests",
                        "index": {relative_path: [offset, n_samples]}}

Every concurrent training run memory-maps the same file, so the page cache
holds one copy and no run decodes audio per epoch. That removes the CPU
decode/resample cost that kept the GPUs mostly idle in phase 1/2.

Usage (from repo root):
    python scripts/build_waveform_cache.py \
        --manifest data/manifests/composite_v1.csv \
        --out data/cache/composite_v1_wave16k
    python scripts/build_waveform_cache.py --out data/cache/composite_v1_wave16k \
        --manifest data/manifests/composite_v1.csv --verify 300

--verify N re-decodes N random cached clips from disk and fails if any
differs by more than 1e-6 (gate C0 in CRNN_ATTN_INSTRUCTIONS.md).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vcm_data_loader import AudioProcessor  # noqa: E402


def unique_paths(manifests: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for m in manifests:
        with open(m, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.setdefault(row["relative_path"], None)
    return list(seen)


def build(manifests: list[str], out: Path, audio_root: Path) -> None:
    proc = AudioProcessor()
    paths = unique_paths(manifests)
    out.mkdir(parents=True, exist_ok=True)
    index: dict[str, list[int]] = {}
    offset = 0
    failed: list[str] = []
    with open(out / "waves.f32", "wb") as fh:
        for i, rel in enumerate(paths):
            try:
                wav, sr = torchaudio.load(str(audio_root / rel))
                wav = proc.to_mono_16k(wav, sr)[0].numpy().astype(np.float32, copy=False)
            except Exception as exc:  # report, never silently drop
                failed.append(f"{rel}\t{exc}")
                continue
            fh.write(wav.tobytes())
            index[rel] = [offset, int(wav.shape[0])]
            offset += int(wav.shape[0])
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(paths)} clips, {offset / 16000 / 3600:.2f} h audio")
    meta = {"sample_rate": proc.sample_rate, "total_samples": offset,
            "manifests": manifests, "n_clips": len(index), "index": index}
    (out / "index.json").write_text(json.dumps(meta))
    gb = offset * 4 / 1e9
    print(f"cached {len(index)} clips, {offset / 16000 / 3600:.2f} h, {gb:.2f} GB -> {out}")
    if failed:
        (out / "failed.txt").write_text("\n".join(failed))
        print(f"WARNING: {len(failed)} clips failed to decode, listed in {out / 'failed.txt'}")


def verify(out: Path, audio_root: Path, n: int, seed: int = 0) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.vcm_data_loader import WaveformCache

    cache = WaveformCache(str(out))
    proc = AudioProcessor()
    keys = list(cache.index)
    random.Random(seed).shuffle(keys)
    worst = 0.0
    for rel in keys[:n]:
        wav, sr = torchaudio.load(str(audio_root / rel))
        ref = proc.to_mono_16k(wav, sr)
        got = cache.get(rel)
        if ref.shape != got.shape:
            raise SystemExit(f"FAIL shape {rel}: disk {tuple(ref.shape)} vs cache {tuple(got.shape)}")
        worst = max(worst, float((ref - got).abs().max()))
    status = "PASS" if worst <= 1e-6 else "FAIL"
    print(f"verify {status}: {min(n, len(keys))} clips, max |disk - cache| = {worst:.2e}")
    if status == "FAIL":
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", action="append", required=True, help="repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio-root", default=".")
    ap.add_argument("--verify", type=int, default=0, help="only verify N random clips of an existing cache")
    args = ap.parse_args()
    out, root = Path(args.out), Path(args.audio_root)
    if args.verify:
        verify(out, root, args.verify)
    else:
        build(args.manifest, out, root)


if __name__ == "__main__":
    main()
