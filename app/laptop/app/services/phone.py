"""Deterministic phone simulation. The VCM has no contact or message-text slot,
so CALL and MESSAGE go to the configured demo contact with a fixed text."""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional

CONNECT_DELAY_S = 3.0

Notify = Callable[[str, dict], Awaitable[None]]


class PhoneService:
    def __init__(self, contact: dict, default_text: str, notify: Notify):
        self.contact = contact
        self.default_text = default_text
        self.notify = notify
        self.call_status = "idle"   # idle | calling | connected
        self.call_started_ms: Optional[float] = None
        self.messages: list[dict] = []
        self._task: Optional[asyncio.Task] = None

    def call(self, now_ms: float) -> str:
        self.end()
        self.call_status, self.call_started_ms = "calling", now_ms
        self._task = asyncio.create_task(self._connect())
        return f"calling {self.contact['name']}"

    async def _connect(self) -> None:
        await asyncio.sleep(CONNECT_DELAY_S)
        if self.call_status == "calling":
            self.call_status = "connected"
            await self.notify("call_connected", {"contact": self.contact["name"]})

    def end(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        self.call_status, self.call_started_ms = "idle", None

    def message(self, now_ms: float) -> str:
        self.messages.append({"text": self.default_text, "ts_ms": now_ms, "to": self.contact["name"]})
        self.messages = self.messages[-20:]
        return f"message sent to {self.contact['name']}"

    def snapshot(self) -> dict:
        return {"contact": self.contact, "call_status": self.call_status,
                "call_started_ms": self.call_started_ms, "messages": self.messages}
