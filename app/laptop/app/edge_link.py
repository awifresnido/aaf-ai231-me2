"""Connection from the laptop app to the edge pipeline.

LocalEdgeLink  -- pipeline runs in this process (development without the Pi).
RemoteEdgeLink -- WebSocket client to the Pi (or WSL), with automatic reconnect.
Both expose the same three calls, so nothing above this layer changes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

log = logging.getLogger(__name__)

OnMessage = Callable[[dict], Awaitable[None]]
RECONNECT_DELAY_S = 2.0
PING_INTERVAL_S = 2.0


class EdgeLink:
    mode = "base"
    connected = False
    rtt_ms: Optional[float] = None
    url = ""

    async def start(self) -> None: ...
    async def send(self, msg: dict) -> None: ...
    async def send_audio(self, pcm: bytes) -> None: ...


class LocalEdgeLink(EdgeLink):
    mode = "local"

    def __init__(self, cfg: dict, repo_root: Path, on_message: OnMessage):
        from edge.factory import build_pipeline  # imported lazily: remote mode never needs it

        self.pipeline = build_pipeline(cfg, repo_root, on_message)
        self.connected = True
        self.url = "in-process"
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        await self.pipeline.startup()
        self._tasks.append(asyncio.create_task(self.pipeline.health_loop()))
        await self.pipeline.publish_state()

    async def send(self, msg: dict) -> None:
        await self.pipeline.handle(msg)

    async def send_audio(self, pcm: bytes) -> None:
        await self.pipeline.feed_pcm(pcm)


class RemoteEdgeLink(EdgeLink):
    mode = "remote"

    def __init__(self, host: str, port: int, on_message: OnMessage,
                 on_status: Callable[[], Awaitable[None]], clock=None):
        self.url = f"ws://{host}:{port}/ws/edge"
        self.on_message = on_message
        self.on_status = on_status
        self.clock = clock
        self._ws = None
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        import websockets

        while True:
            try:
                async with websockets.connect(self.url, max_size=8 * 2**20,
                                              open_timeout=3) as ws:
                    self._ws, self.connected = ws, True
                    await self.on_status()
                    pinger = asyncio.create_task(self._ping_loop())
                    try:
                        async for raw in ws:
                            if isinstance(raw, str):
                                msg = json.loads(raw)
                                if msg.get("type") == "pong":
                                    t3 = time.time() * 1000
                                    self.rtt_ms = round(t3 - float(msg["t"]), 1)
                                    if self.clock is not None and "t_edge" in msg:
                                        self.clock.feed(float(msg["t"]), float(msg["t_edge"]), t3)
                                    continue
                                await self.on_message(msg)
                    finally:
                        pinger.cancel()
            except Exception as exc:  # noqa: BLE001 -- keep reconnecting
                log.info("edge link down (%s); retrying in %.0f s", exc, RECONNECT_DELAY_S)
            was_connected = self.connected
            self._ws, self.connected, self.rtt_ms = None, False, None
            if was_connected:
                if self.clock is not None:
                    self.clock.reset()
                await self.on_status()
            await asyncio.sleep(RECONNECT_DELAY_S)

    async def _ping_loop(self) -> None:
        while True:
            await self.send({"type": "ping", "t": time.time() * 1000})
            await asyncio.sleep(PING_INTERVAL_S)

    async def send(self, msg: dict) -> None:
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps(msg))
            except Exception:  # noqa: BLE001
                pass

    async def send_audio(self, pcm: bytes) -> None:
        if self._ws is not None:
            try:
                await self._ws.send(pcm)
            except Exception:  # noqa: BLE001
                pass
