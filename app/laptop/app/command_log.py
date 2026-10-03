"""Per-command log (SQLite) and per-model scoreboard for beta testing.

Every clip's predictions -- the active model's and each challenger's -- are
stored, so labelling one command ('it should have been X') scores every model
that heard that clip.
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
from pathlib import Path
from typing import Optional

from vcm_common.protocol import InferenceResult

from .clock_sync import timing_trusted
from .dispatcher import Outcome

COLUMNS = ("command_id", "ts_ms", "source", "model_id", "class_key", "confidence", "accepted",
           "reject_reason", "threshold", "feature_ms", "inference_ms", "clip_ms",
           "clip_rms_dbfs", "capture_to_result_ms", "result_to_action_ms", "action",
           "executed", "expected_class", "t_received_ms", "network_ms",
           "edge_clock_offset_ms", "timing_valid", "clock_quality")


class CommandLog:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS commands (
              command_id TEXT PRIMARY KEY, ts_ms REAL, source TEXT, model_id TEXT,
              class_key TEXT, confidence REAL, accepted INTEGER, reject_reason TEXT,
              threshold REAL, feature_ms REAL, inference_ms REAL, clip_ms REAL,
              clip_rms_dbfs REAL, capture_to_result_ms REAL, result_to_action_ms REAL,
              action TEXT, executed INTEGER, expected_class TEXT,
              t_received_ms REAL, network_ms REAL, edge_clock_offset_ms REAL, timing_valid INTEGER,
              clock_quality TEXT);
            CREATE TABLE IF NOT EXISTS predictions (
              command_id TEXT, model_id TEXT, role TEXT, class_key TEXT,
              confidence REAL, inference_ms REAL, top_k TEXT);
            """
        )
        self.db.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Add clock-sync columns to an existing DB; flag pre-fix rows.

        Rows written before ``clock_quality`` existed get 'unknown', so they are
        excluded from cross-device latency (their offset may have been the
        coarse ingress estimate) but kept for accuracy."""
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(commands)")}
        for col, decl in (("t_received_ms", "REAL"), ("network_ms", "REAL"),
                          ("edge_clock_offset_ms", "REAL"), ("timing_valid", "INTEGER"),
                          ("clock_quality", "TEXT")):
            if col not in cols:
                self.db.execute(f"ALTER TABLE commands ADD COLUMN {col} {decl}")
        self.db.commit()
        self.db.execute("UPDATE commands SET timing_valid=0 WHERE timing_valid IS NULL")
        self.db.execute("UPDATE commands SET clock_quality='unknown' WHERE clock_quality IS NULL")
        self.db.commit()

    def record(self, r: InferenceResult, o: Outcome, action_ms: float) -> dict:
        p = r.primary
        cap = (r.edge_times["t_result_ms"] - r.edge_times["t_capture_end_ms"]
               if r.edge_times else r.t_result_ms - r.t_capture_end_ms)
        rta = action_ms - r.t_received_ms if r.t_received_ms is not None else None
        # cross-device hop: only with a settled, accurate (NTP-style or local) clock
        network = (r.t_received_ms - r.t_result_ms
                   if r.t_received_ms is not None
                   and timing_trusted(r.timing_valid, r.clock_quality) else None)
        row = (r.command_id, r.t_result_ms, r.source, p.model_id, p.class_key, p.confidence,
               int(r.accepted), r.reject_reason, r.threshold, p.feature_ms, p.inference_ms,
               r.clip_ms, r.clip_rms_dbfs, cap, rta, o.action, int(o.executed),
               r.expected_class, r.t_received_ms, network, r.edge_clock_offset_ms,
               int(r.timing_valid), r.clock_quality)
        self.db.execute(f"INSERT OR REPLACE INTO commands VALUES ({','.join('?' * len(row))})", row)
        roles = [("primary", p)] + [("compare", c) for c in r.compare] \
            + [("member", c) for c in r.members]
        for role, pred in roles:
            self.db.execute(
                "INSERT INTO predictions VALUES (?,?,?,?,?,?,?)",
                (r.command_id, pred.model_id, role, pred.class_key, pred.confidence,
                 pred.inference_ms, json.dumps(pred.top_k)),
            )
        self.db.commit()
        return dict(zip(COLUMNS, row))

    def label(self, command_id: str, expected_class: Optional[str]) -> None:
        self.db.execute("UPDATE commands SET expected_class=? WHERE command_id=?",
                        (expected_class, command_id))
        self.db.commit()

    def recent(self, n: int = 15) -> list[dict]:
        rows = self.db.execute(
            f"SELECT {','.join(COLUMNS)} FROM commands ORDER BY ts_ms DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(zip(COLUMNS, r)) for r in rows]

    def scoreboard(self) -> list[dict]:
        """Per-model accuracy on labelled, non-simulated clips.

        ``accuracy`` compares full class keys (intent + slot); ``intent_accuracy``
        compares intents only, so intent-mode models (no slots) can be ranked
        against leaf-mode models on the same clips.
        """
        rows = self.db.execute(
            "SELECT p.model_id, p.class_key, p.confidence, p.inference_ms, c.expected_class "
            "FROM predictions p JOIN commands c USING (command_id) "
            "WHERE c.source != 'simulated'"
        ).fetchall()
        agg: dict[str, dict] = {}
        for mid, key, conf, ms, expected in rows:
            a = agg.setdefault(mid, {"model_id": mid, "n_clips": 0, "n_labelled": 0,
                                     "n_correct": 0, "n_intent_correct": 0, "_conf": 0.0,
                                     "_ms": 0.0})
            a["n_clips"] += 1
            a["_conf"] += conf or 0.0
            a["_ms"] += ms or 0.0
            if expected:
                a["n_labelled"] += 1
                a["n_correct"] += int(key == expected)
                a["n_intent_correct"] += int(key.split("|")[0] == expected.split("|")[0])
        out = []
        for a in sorted(agg.values(), key=lambda x: x["model_id"]):
            n, nl = a["n_clips"], a["n_labelled"]
            out.append({
                "model_id": a["model_id"], "n_clips": n, "n_labelled": nl,
                "n_correct": a["n_correct"], "n_intent_correct": a["n_intent_correct"],
                "accuracy": a["n_correct"] / nl if nl else None,
                "intent_accuracy": a["n_intent_correct"] / nl if nl else None,
                "mean_conf": round(a["_conf"] / n, 3) if n else None,
                "mean_inference_ms": round(a["_ms"] / n, 2) if n else None,
            })
        return out

    def export_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(COLUMNS)
        for r in self.db.execute(f"SELECT {','.join(COLUMNS)} FROM commands ORDER BY ts_ms"):
            w.writerow(r)
        return buf.getvalue()

    def export_predictions_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["command_id", "model_id", "role", "class_key", "confidence",
                    "inference_ms", "expected_class", "source"])
        for r in self.db.execute(
            "SELECT p.command_id, p.model_id, p.role, p.class_key, p.confidence, "
            "p.inference_ms, c.expected_class, c.source FROM predictions p "
            "JOIN commands c USING (command_id) ORDER BY c.ts_ms"
        ):
            w.writerow(r)
        return buf.getvalue()
