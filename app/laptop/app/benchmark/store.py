"""SQLite store for live benchmark sessions/trials/false wakes (runtime/benchmark.sqlite)."""
from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
  session_id TEXT PRIMARY KEY, alias TEXT, created_ms REAL, finished_ms REAL,
  status TEXT, mode TEXT, set_type TEXT, set_seed INTEGER,
  model_id TEXT, threshold REAL,
  cond_noise TEXT, cond_distance TEXT, network_offline INTEGER,
  consent_results INTEGER, consent_audio INTEGER, note TEXT);
CREATE TABLE IF NOT EXISTS trials(
  trial_id TEXT PRIMARY KEY, session_id TEXT, idx INTEGER, attempt INTEGER,
  kind TEXT, prompt TEXT, expected_class TEXT,
  outcome TEXT, intent_correct INTEGER, misspoken INTEGER DEFAULT 0, superseded INTEGER DEFAULT 0,
  command_id TEXT, predicted_class TEXT, confidence REAL, accepted INTEGER,
  t_prompt_ms REAL, t_wake_ms REAL, t_wake_received_ms REAL, t_result_ms REAL,
  inference_ms REAL, feature_ms REAL, capture_to_result_ms REAL, result_to_action_ms REAL,
  t_received_ms REAL, network_ms REAL, timing_valid INTEGER, clock_quality TEXT);
CREATE TABLE IF NOT EXISTS false_wakes(session_id TEXT, t_ms REAL);
"""

SCORED_OUTCOMES = {"correct", "not_understood", "wrong", "false_accept", "correct_reject"}
EXCLUDED_OUTCOMES = {"wake_miss", "no_result", "skipped"}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_ms() -> float:
    return time.time() * 1000.0


class BenchmarkStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Add clock-sync columns to an existing trials table; flag pre-fix rows.
        Trials recorded before ``clock_quality`` existed get 'unknown' (excluded
        from end-to-end latency, kept for accuracy)."""
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(trials)")}
        for col, decl in (("t_wake_received_ms", "REAL"), ("t_received_ms", "REAL"),
                          ("network_ms", "REAL"), ("timing_valid", "INTEGER"),
                          ("clock_quality", "TEXT")):
            if col not in cols:
                self.db.execute(f"ALTER TABLE trials ADD COLUMN {col} {decl}")
        self.db.commit()
        self.db.execute("UPDATE trials SET timing_valid=0 WHERE timing_valid IS NULL")
        self.db.execute("UPDATE trials SET clock_quality='unknown' "
                        "WHERE clock_quality IS NULL AND outcome IS NOT NULL")
        self.db.commit()

    # ---- sessions -----------------------------------------------------------
    def next_alias(self) -> str:
        n = self.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] + 1
        return f"Guest {n:02d}"

    def create_session(self, *, alias: str, mode: str, set_type: str, set_seed: Optional[int],
                       model_id: str, threshold: float, cond_noise: str, cond_distance: str,
                       network_offline: bool, consent_results: bool, consent_audio: bool,
                       note: Optional[str], prompts: list[dict]) -> str:
        sid = new_id("sess")
        self.db.execute(
            "INSERT INTO sessions (session_id, alias, created_ms, status, mode, set_type, set_seed, "
            "model_id, threshold, cond_noise, cond_distance, network_offline, consent_results, "
            "consent_audio, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, alias, now_ms(), "created", mode, set_type, set_seed, model_id, threshold,
             cond_noise, cond_distance, int(network_offline), int(consent_results),
             int(consent_audio), note))
        for i, p in enumerate(prompts):
            self.db.execute(
                "INSERT INTO trials (trial_id, session_id, idx, attempt, kind, prompt, expected_class) "
                "VALUES (?,?,?,?,?,?,?)",
                (new_id("trial"), sid, i, 0, p["kind"], p["say"], p.get("expected_class")))
        self.db.commit()
        return sid

    def get_session(self, sid: str) -> Optional[dict]:
        row = self.db.execute("SELECT * FROM sessions WHERE session_id=?", (sid,)).fetchone()
        return dict(row) if row else None

    def list_sessions(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM sessions ORDER BY created_ms")]

    def set_status(self, sid: str, status: str, finished_ms: Optional[float] = None) -> None:
        if finished_ms is not None:
            self.db.execute("UPDATE sessions SET status=?, finished_ms=? WHERE session_id=?",
                            (status, finished_ms, sid))
        else:
            self.db.execute("UPDATE sessions SET status=? WHERE session_id=?", (status, sid))
        self.db.commit()

    def update_alias(self, sid: str, alias: str) -> None:
        self.db.execute("UPDATE sessions SET alias=? WHERE session_id=?", (alias, sid))
        self.db.commit()

    # ---- trials -------------------------------------------------------------
    def trials_for_session(self, sid: str) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM trials WHERE session_id=? ORDER BY idx, attempt", (sid,))]

    def trial_at(self, sid: str, idx: int, attempt: int) -> str | None:
        row = self.db.execute(
            "SELECT trial_id FROM trials WHERE session_id=? AND idx=? AND attempt=?",
            (sid, idx, attempt)).fetchone()
        return row["trial_id"] if row else None

    def add_attempt(self, sid: str, idx: int, attempt: int, kind: str, prompt: str,
                    expected_class: Optional[str]) -> str:
        tid = new_id("trial")
        self.db.execute(
            "INSERT INTO trials (trial_id, session_id, idx, attempt, kind, prompt, expected_class) "
            "VALUES (?,?,?,?,?,?,?)", (tid, sid, idx, attempt, kind, prompt, expected_class))
        self.db.commit()
        return tid

    def update_trial(self, trial_id: str, **fields) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE trials SET {cols} WHERE trial_id=?",
                        (*fields.values(), trial_id))
        self.db.commit()

    def mark_superseded(self, trial_id: str) -> None:
        self.db.execute("UPDATE trials SET superseded=1 WHERE trial_id=?", (trial_id,))
        self.db.commit()

    def record_false_wake(self, sid: str, t_ms: float) -> None:
        self.db.execute("INSERT INTO false_wakes (session_id, t_ms) VALUES (?,?)", (sid, t_ms))
        self.db.commit()

    def false_wake_count(self, sid: str) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM false_wakes WHERE session_id=?", (sid,)).fetchone()[0]

    # ---- read / export ------------------------------------------------------
    def all_scored_trials(self) -> list[dict]:
        """Trials joined with their session, for metrics. Excludes nothing here;
        exclusion is the metrics module's job."""
        return [dict(r) for r in self.db.execute(
            "SELECT t.*, s.alias, s.set_type, s.cond_noise, s.cond_distance, s.model_id, "
            "s.status, s.created_ms FROM trials t JOIN sessions s USING (session_id) ORDER BY s.created_ms, t.idx")]

    def delete_session(self, sid: str) -> list[str]:
        """Delete a session + trials + false wakes; return the command_ids it labelled."""
        cids = [r["command_id"] for r in self.db.execute(
            "SELECT command_id FROM trials WHERE session_id=? AND command_id IS NOT NULL", (sid,))]
        self.db.execute("DELETE FROM false_wakes WHERE session_id=?", (sid,))
        self.db.execute("DELETE FROM trials WHERE session_id=?", (sid,))
        self.db.execute("DELETE FROM sessions WHERE session_id=?", (sid,))
        self.db.commit()
        return cids
