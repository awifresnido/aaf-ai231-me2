#!/usr/bin/env python3
"""check_public.py — public-repo hygiene gate (run in CI, and by hand).

Fails (exit 1) on any tracked file that still contains a sensitive pattern or
on any tracked binary that must never be committed. The rules mirror Section 8
of the build instructions.
"""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Ordered rules: (name, regex, hint). A match is a failure.
RULES = [
    ("internal-ip",      re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "internal IP (10.x)"),
    ("lan-ip",           re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b"), "LAN IP"),
    ("tailnet-ip",       re.compile(r"\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "tailnet IP"),
    ("hpc-hostname",     re.compile(r"hpc\.coe\.upd\.edu\.ph"), "internal hostname"),
    ("dgx-username",     re.compile(r"aldwin\.fresnido"), "username"),
    ("ssh-key",          re.compile(r"id_ed25519"), "SSH key name"),
    ("abs-home",         re.compile(r"/home/[\w.-]+/"), "absolute /home path"),
    ("abs-opt",          re.compile(r"/opt/"), "absolute /opt path"),
    ("phone",            re.compile(r"\+63[\d\s-]{6,}"), "phone number"),
    ("email",            re.compile(r"[\w.+-]+@[\w.-]+\.\w{2,}"), "email address"),
    ("api-key",          re.compile(r"(?i)(sk-[A-Za-z0-9]{16,}|AIza[A-Za-z0-9_-]{30,}|xox[baprs]-[A-Za-z0-9-]{10,}|ghp_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{20,})"), "API key/token"),
]

# File extensions that must never be tracked.
BANNED_EXT = {".pt", ".pth", ".ckpt", ".onnx", ".tflite", ".safetensors",
              ".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".parquet",
              ".arrow", ".bin", ".zip", ".7z"}

TEXT_EXT = {".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".json", ".csv",
            ".cfg", ".toml", ".cff", ".ts", ".tsx", ".js", ".jsx", ".html",
            ".css", ".example"}


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return [f for f in out if f]


def main() -> int:
    failures = 0
    # Detectors contain the sensitive literals by design (as regex sources).
    SKIP = {"scripts/check_public.py", "scripts/secret_scan.py"}
    for rel in tracked_files():
        if rel in SKIP:
            continue
        p = ROOT / rel
        if p.suffix.lower() in BANNED_EXT:
            print(f"BANNED  {rel}")
            failures += 1
            continue
        if p.suffix.lower() not in TEXT_EXT:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, pat, hint in RULES:
                if pat.search(line):
                    print(f"{rel}:{lineno}: {name} ({hint})")
                    failures += 1
    if failures:
        print(f"\ncheck_public: {failures} violation(s) found", file=sys.stderr)
        return 1
    print("check_public: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
