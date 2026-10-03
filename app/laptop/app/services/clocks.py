"""Timer and alarm services (laptop clock; no network)."""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta
from typing import Awaitable, Callable, Optional
from zoneinfo import ZoneInfo

Notify = Callable[[str, dict], Awaitable[None]]

_DURATION_RE = re.compile(r"^\s*(\d+)\s*(second|seconds|minute|minutes|hour|hours)\s*$", re.I)
_UNIT_S = {"second": 1, "minute": 60, "hour": 3600}


def parse_duration_s(slot: str) -> int:
    """'30 seconds' -> 30, '1 minute' -> 60."""
    m = _DURATION_RE.match(slot)
    if not m:
        raise ValueError(f"cannot parse duration '{slot}'")
    unit = m.group(2).lower().rstrip("s")
    return int(m.group(1)) * _UNIT_S[unit]


def next_alarm(slot: str, now: datetime) -> datetime:
    """Next occurrence of '6:00 AM' strictly after ``now`` (tz-aware)."""
    t = datetime.strptime(slot.strip().upper(), "%I:%M %p").time()
    candidate = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


class TimerService:
    def __init__(self, notify: Notify):
        self.notify = notify
        self.label: Optional[str] = None
        self.duration_s = 0
        self.ends_at_ms: Optional[float] = None
        self.status = "idle"   # idle | running | done
        self._task: Optional[asyncio.Task] = None

    def start(self, slot: str, now_ms: float) -> str:
        self.cancel()
        self.duration_s = parse_duration_s(slot)
        self.label = slot
        self.ends_at_ms = now_ms + self.duration_s * 1000
        self.status = "running"
        self._task = asyncio.create_task(self._run(self.duration_s))
        return f"timer {slot}"

    async def _run(self, seconds: int) -> None:
        await asyncio.sleep(seconds)
        self.status = "done"
        await self.notify("timer_done", {"label": self.label})

    def cancel(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        self.status, self.ends_at_ms = "idle", None

    def snapshot(self) -> dict:
        return {"status": self.status, "label": self.label, "duration_s": self.duration_s,
                "ends_at_ms": self.ends_at_ms}


class AlarmService:
    CHECK_INTERVAL_S = 2.0

    def __init__(self, notify: Notify, tz: str):
        self.notify = notify
        self.tz = ZoneInfo(tz)
        self.slot: Optional[str] = None
        self.at: Optional[datetime] = None
        self.status = "none"   # none | set | ringing

    def set(self, slot: str) -> str:
        self.slot = slot
        self.at = next_alarm(slot, datetime.now(self.tz))
        self.status = "set"
        return f"alarm {slot} ({self.at:%a %d %b})"

    def dismiss(self) -> None:
        self.status, self.at, self.slot = "none", None, None

    async def loop(self) -> None:
        while True:
            await asyncio.sleep(self.CHECK_INTERVAL_S)
            if self.status == "set" and self.at and datetime.now(self.tz) >= self.at:
                self.status = "ringing"
                await self.notify("alarm_ringing", {"slot": self.slot})

    def snapshot(self) -> dict:
        return {"status": self.status, "slot": self.slot,
                "at_iso": self.at.isoformat() if self.at else None}
