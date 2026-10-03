#!/usr/bin/env python3
"""make_figures.py — generate the results tables and figures from results/ only.

Reads results/me2_gold/eval_arch_*.json and decision.json and (re)writes:

  - the README "Results" table (between AUTOGEN markers)
  - docs/figures/baseline_table.md
  - docs/figures/holdout.svg

No hand-typed numbers. `--check` regenerates in-memory and fails on any diff.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "me2_gold"
README = ROOT / "README.md"
FIGS = ROOT / "docs" / "figures"

MODELS = [
    # (arch key, label)
    ("E1f", "E1f (CRNN-Attn)"),
    ("B2f", "B2f (TC-ResNet)"),
    ("G2f", "G2f (DS-CNN)"),
    ("E1",  "v1 E1 (CRNN-Attn)"),
    ("B2",  "v1 B2 (TC-ResNet)"),
    ("G2",  "v1 G2 (DS-CNN)"),
]

BEGIN = "<!-- AUTOGEN:results -->"
END = "<!-- /AUTOGEN:results -->"


def pct(x) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.2f} %"


def load_arch(key: str) -> dict:
    p = RESULTS / f"eval_arch_{key}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def render_table() -> str:
    rows = []
    header = ("| Model (arch) | holdout_only | holdout_real | holdout_syn | holdout FAR |\n"
              "|---|---|---|---|---|")
    for key, label in MODELS:
        d = load_arch(key)
        ps = d.get("per_slice", {})
        holdout_only = ps.get("holdout_only", {}).get("correct")
        holdout_real = ps.get("holdout_real", {}).get("correct")
        holdout_syn = ps.get("holdout_syn", {}).get("correct")
        far = ps.get("holdout_oos", {}).get("FAR")
        rows.append(f"| **{label}** | {pct(holdout_only)} | {pct(holdout_real)} | "
                    f"{pct(holdout_syn)} | {pct(far)} |")
    return header + "\n" + "\n".join(rows) + "\n"


def render_svg() -> str:
    """Horizontal bar chart of holdout_only correct per model (pure SVG)."""
    bars = []
    for key, label in MODELS:
        d = load_arch(key)
        v = d.get("per_slice", {}).get("holdout_only", {}).get("correct") or 0.0
        bars.append((label, v))
    w, h = 720, 40 + 34 * len(bars)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'font-family="sans-serif" font-size="13">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    top = 24
    maxv = 1.0
    for i, (label, v) in enumerate(bars):
        y = top + i * 34
        bw = int((v / maxv) * 480)
        lines.append(f'<text x="8" y="{y}" fill="#111">{label}</text>')
        lines.append(f'<rect x="220" y="{y - 11}" width="{bw}" height="18" '
                     f'rx="3" fill="#0b6bcb"/>')
        lines.append(f'<text x="{230 + bw}" y="{y}" fill="#111">{pct(v)}</text>')
    lines.append('</svg>')
    return "\n".join(lines) + "\n"


def update_readme(table: str) -> None:
    text = README.read_text()
    if BEGIN not in text or END not in text:
        print("README missing AUTOGEN markers; skipping README update", file=sys.stderr)
        return
    pre = text.split(BEGIN)[0]
    post = text.split(END)[1]
    README.write_text(pre + BEGIN + "\n" + table + "\n" + END + post)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="regenerate in-memory and fail on any diff")
    args = ap.parse_args()

    table = render_table()
    svg = render_svg()

    FIGS.mkdir(parents=True, exist_ok=True)
    table_path = FIGS / "baseline_table.md"
    svg_path = FIGS / "holdout.svg"

    if args.check:
        failures = 0
        if table_path.exists() and table_path.read_text() != table:
            print(f"DIFF: {table_path}", file=sys.stderr)
            failures += 1
        if svg_path.exists() and svg_path.read_text() != svg:
            print(f"DIFF: {svg_path}", file=sys.stderr)
            failures += 1
        if README.exists():
            t = README.read_text()
            if BEGIN in t and table not in t:
                print("DIFF: README results table", file=sys.stderr)
                failures += 1
        if failures:
            print(f"make_figures --check: {failures} diff(s)", file=sys.stderr)
            return 1
        print("make_figures --check: clean")
        return 0

    update_readme(table)
    table_path.write_text(table)
    svg_path.write_text(svg)
    print(f"wrote {table_path} and {svg_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
