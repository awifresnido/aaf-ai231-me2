"""Reminders (SQLite, persistent across restarts)."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class ReminderStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS reminders ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL,"
            " source TEXT NOT NULL, created_ms REAL NOT NULL, done INTEGER NOT NULL DEFAULT 0)"
        )
        self.db.commit()
        self.focus_until_ms = 0.0

    def add(self, text: str, source: str) -> int:
        cur = self.db.execute(
            "INSERT INTO reminders(text, source, created_ms) VALUES (?, ?, ?)",
            (text.strip(), source, time.time() * 1000),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def set_done(self, rid: int, done: bool) -> None:
        self.db.execute("UPDATE reminders SET done=? WHERE id=?", (int(done), rid))
        self.db.commit()

    def delete(self, rid: int) -> None:
        self.db.execute("DELETE FROM reminders WHERE id=?", (rid,))
        self.db.commit()

    def all(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, text, source, created_ms, done FROM reminders ORDER BY done, id DESC"
        ).fetchall()
        return [dict(zip(("id", "text", "source", "created_ms", "done"), r)) for r in rows]

    def snapshot(self) -> dict:
        return {"items": self.all(), "focus_until_ms": self.focus_until_ms}
