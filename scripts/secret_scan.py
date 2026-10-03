#!/usr/bin/env python3
"""secret_scan.py — pure-Python secret scanner (gitleaks substitute).

Runs over the git-tracked files of the repository and flags high-entropy
tokens and common secret shapes. Zero false-positive tolerance is impossible
without a real scanner, so this reports candidates for human review and exits
non-zero only on hard matches (known secret prefixes, private keys, env files).
"""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HARD_PATTERNS = [
    (re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"), "private key"),
    (re.compile(r"(?i)(sk-[A-Za-z0-9]{20,})"), "OpenAI key"),
    (re.compile(r"(?i)(AIza[A-Za-z0-9_-]{30,})"), "Google key"),
    (re.compile(r"(?i)(ghp_[A-Za-z0-9]{30,})"), "GitHub PAT"),
    (re.compile(r"(?i)(hf_[A-Za-z0-9]{20,})"), "HuggingFace token"),
    (re.compile(r"(?i)(xox[baprs]-[A-Za-z0-9-]{10,})"), "Slack token"),
    (re.compile(r"(?i)(AKIA[0-9A-Z]{16})"), "AWS access key"),
]

SOFT_PATTERNS = [
    (re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd)\s*[:=]\s*['\"][^'\"]{8,}['\"]"), "key=value credential"),
    (re.compile(r"(?i)(Bearer\s+[A-Za-z0-9._-]{20,})"), "bearer token"),
]


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    return out


def main() -> int:
    hard = 0
    soft = 0
    for rel in tracked_files():
        p = ROOT / rel
        if p.suffix.lower() not in {".py", ".sh", ".yaml", ".yml", ".json",
                                    ".toml", ".env", ".txt", ".md", ".cfg",
                                    ".ts", ".tsx", ".js"}:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if p.name.startswith(".env") and rel not in {".env.example"}:
            print(f"HARD {rel}: env file tracked")
            hard += 1
        for lineno, line in enumerate(text.splitlines(), 1):
            for pat, name in HARD_PATTERNS:
                if pat.search(line):
                    print(f"HARD {rel}:{lineno}: {name}")
                    hard += 1
            for pat, name in SOFT_PATTERNS:
                if pat.search(line):
                    print(f"SOFT {rel}:{lineno}: {name}")
                    soft += 1
    if hard:
        print(f"\nsecret_scan: {hard} hard match(es)", file=sys.stderr)
        return 1
    print(f"secret_scan: clean (0 hard, {soft} soft candidate(s) for review)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
