"""Edge WebSocket service (runs on the Raspberry Pi; in WSL during development).

    python -m edge.server --config config/demo.yaml

Binary frames in  = 16 kHz mono PCM16 microphone audio.
JSON frames in    = control messages (see EdgePipeline.handle).
JSON frames out   = phase / event / inference_result / edge_state / level / pong.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .factory import build_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("edge.server")


def create_app(cfg: dict) -> FastAPI:
    clients: set[WebSocket] = set()

    async def emit(msg: dict) -> None:
        dead = []
        text = json.dumps(msg)
        for ws in list(clients):
            try:
                await ws.send_text(text)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)

    pipeline = build_pipeline(cfg, REPO_ROOT, emit)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await pipeline.startup()
        task = asyncio.create_task(pipeline.health_loop())
        yield
        task.cancel()

    app = FastAPI(title="TinyVCM edge", lifespan=lifespan)
    app.state.pipeline = pipeline

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ready", **pipeline.health()}

    @app.get("/models")
    async def models() -> dict:
        return {"active_vcm": pipeline.s.active_vcm, "models": pipeline.registry.summaries()}

    @app.websocket("/ws/edge")
    async def ws_edge(ws: WebSocket) -> None:
        await ws.accept()
        clients.add(ws)
        await pipeline.publish_state()
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await pipeline.feed_pcm(msg["bytes"])
                elif msg.get("text") is not None:
                    try:
                        await pipeline.handle(json.loads(msg["text"]))
                    except json.JSONDecodeError:
                        log.warning("dropped malformed control message")
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(ws)

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "demo.yaml"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    e = cfg["edge_service"]
    uvicorn.run(create_app(cfg), host=e.get("bind_host", "0.0.0.0"), port=int(e.get("port", 8765)),
                ws_max_size=8 * 2**20)


if __name__ == "__main__":
    main()
