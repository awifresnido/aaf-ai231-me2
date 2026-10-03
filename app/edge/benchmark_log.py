"""vcm-benchmark decision/wake log lines (one JSON line per event, flushed).

Written to ~/vcm_benchmark/<name>.log on the edge; the benchmark reads the newest
.log there. Enabled only when the config sets ``edge_service.benchmark_mode: true``.
"""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

DEFAULT_LOG_DIR = Path(os.path.expanduser("~/vcm_benchmark"))


class BenchmarkLog:
    """Appends the benchmark's expected lines to a log file on the edge.

    Wake:      {"event": "wake"}
    Decision:  {"intent": "<INTENT|OUT_OF_SCOPE>", "slot": "...", "infer_ms": f, "audio_ms": i}
    """

    def __init__(self, enabled: bool = True, log_dir: Path | None = None) -> None:
        self._fh = None
        if not enabled:
            return
        log_dir = log_dir or DEFAULT_LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        name = f"vcm_{datetime.datetime.now():%Y%m%d-%H%M%S}.log"
        self._fh = open(log_dir / name, "a", buffering=1, encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    def wake(self) -> None:
        if self._fh is not None:
            print(json.dumps({"event": "wake"}), file=self._fh, flush=True)

    def decision(self, intent: str, slot: str, infer_ms: float, audio_ms: float) -> None:
        if self._fh is not None:
            print(json.dumps({"intent": intent, "slot": slot,
                              "infer_ms": round(float(infer_ms), 1),
                              "audio_ms": round(float(audio_ms))}),
                  file=self._fh, flush=True)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
