"""Laptop application: FastAPI backend + compiled React UI on one URL.

    python -m laptop.app.main --config config/demo.yaml            # all-in-WSL (local edge)
    python -m laptop.app.main --config config/laptop-windows.yaml  # Windows + mic, remote edge
    open http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
import yaml
from fastapi import (FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket,
                    WebSocketDisconnect)
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .hub import Hub

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = REPO_ROOT / "laptop" / "frontend" / "dist"
MAX_REPLAY_BYTES = 4 * 2**20


class ClassKey(BaseModel):
    class_key: str


class ModelId(BaseModel):
    model_id: str


class ModelIds(BaseModel):
    model_ids: list[str]


class Threshold(BaseModel):
    value: Optional[float] = None


class WakeId(BaseModel):
    wake_id: str


class LabelBody(BaseModel):
    expected_class: Optional[str] = None


class ReminderBody(BaseModel):
    text: str


class BenchCreate(BaseModel):
    set_type: str
    cond_noise: str = "quiet"
    cond_distance: str = "near"
    consent_results: bool = False
    consent_audio: bool = False
    alias: Optional[str] = None
    note: Optional[str] = None


def create_app(cfg: dict) -> FastAPI:
    hub = Hub(cfg, REPO_ROOT)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await hub.start()
        yield

    app = FastAPI(title="TinyVCM Control Center", lifespan=lifespan)
    app.state.hub = hub

    @app.websocket("/ws/ui")
    async def ws_ui(ws: WebSocket) -> None:
        await ws.accept()
        hub.clients.add(ws)
        await ws.send_json({"kind": "state", "state": hub.snapshot()})
        try:
            while True:
                await ws.receive_text()  # UI sends nothing on this socket; keep it open
        except WebSocketDisconnect:
            hub.clients.discard(ws)

    @app.get("/api/state")
    async def state() -> dict:
        return hub.snapshot()

    @app.get("/api/about")
    async def about() -> dict:
        about_path = REPO_ROOT / "laptop" / "app" / "about" / "about_data.json"
        if not about_path.exists():
            raise HTTPException(404, "about data not generated (run scripts/build_about_data.py)")
        data = json.loads(about_path.read_text(encoding="utf-8"))
        # The GitHub URL is driven by config, never baked into the snapshot.
        data["repo_url"] = cfg.get("about", {}).get("repo_url")
        return data

    # ---- assistant / models (forwarded to the edge) ------------------------------
    @app.post("/api/wake")
    async def wake() -> dict:
        await hub.to_edge({"type": "wake_manual"})
        return {"ok": True}

    @app.post("/api/simulate")
    async def simulate(body: ClassKey) -> dict:
        if body.class_key not in hub.ontology.leaf_labels():
            raise HTTPException(400, f"unknown class {body.class_key}")
        await hub.to_edge({"type": "simulate", "class_key": body.class_key})
        return {"ok": True}

    @app.post("/api/replay")
    async def replay(files: list[UploadFile] = File(...),
                     expected_class: Optional[str] = Form(None)) -> dict:
        items = []
        for f in files:
            data = await f.read()
            if len(data) > MAX_REPLAY_BYTES:
                raise HTTPException(413, f"{f.filename} is larger than 4 MB")
            items.append({"wav_b64": base64.b64encode(data).decode(), "name": f.filename,
                          "expected_class": expected_class or None})
        asyncio.create_task(hub.replay_batch(items))
        return {"queued": len(items)}

    @app.post("/api/models/select")
    async def select_model(body: ModelId) -> dict:
        if hub.benchmark.running:
            raise HTTPException(409, "model locked during a live benchmark session")
        await hub.to_edge({"type": "select_model", "model_id": body.model_id})
        return {"ok": True}

    @app.post("/api/models/compare")
    async def set_compare(body: ModelIds) -> dict:
        await hub.to_edge({"type": "set_compare", "model_ids": body.model_ids})
        return {"ok": True}

    @app.post("/api/models/rescan")
    async def rescan() -> dict:
        await hub.to_edge({"type": "rescan"})
        return {"ok": True}

    @app.post("/api/wake/select")
    async def select_wake(body: WakeId) -> dict:
        await hub.to_edge({"type": "select_wake", "wake_id": body.wake_id})
        return {"ok": True}

    @app.post("/api/threshold")
    async def threshold(body: Threshold) -> dict:
        if hub.benchmark.running:
            raise HTTPException(409, "threshold locked during a live benchmark session")
        await hub.to_edge({"type": "set_threshold", "value": body.value})
        return {"ok": True}

    @app.post("/api/media/{action}")
    async def media(action: str) -> dict:
        if action not in {"play", "pause", "stop", "next", "volume_up", "volume_down"}:
            raise HTTPException(400, "unknown media action")
        await hub.to_edge({"type": "media", "action": action})
        return {"ok": True}

    # ---- beta-testing labels and exports -----------------------------------------
    @app.post("/api/commands/{command_id}/label")
    async def label(command_id: str, body: LabelBody) -> dict:
        known = set(hub.ontology.leaf_labels()) | set(hub.ontology.intent_labels())
        if body.expected_class and body.expected_class not in known:
            raise HTTPException(400, "unknown class")
        hub.log.label(command_id, body.expected_class)
        await hub.push_state()
        return {"ok": True}

    @app.get("/api/export/commands.csv", response_class=PlainTextResponse)
    async def export_commands() -> str:
        return hub.log.export_csv()

    @app.get("/api/export/predictions.csv", response_class=PlainTextResponse)
    async def export_predictions() -> str:
        return hub.log.export_predictions_csv()

    # ---- subsystems with manual controls -----------------------------------------
    @app.post("/api/reminders")
    async def add_reminder(body: ReminderBody) -> dict:
        if not body.text.strip():
            raise HTTPException(400, "empty reminder")
        rid = hub.s.reminders.add(body.text, source="manual")
        await hub.push_state()
        return {"id": rid}

    @app.post("/api/reminders/{rid}/toggle")
    async def toggle_reminder(rid: int, done: bool = True) -> dict:
        hub.s.reminders.set_done(rid, done)
        await hub.push_state()
        return {"ok": True}

    @app.delete("/api/reminders/{rid}")
    async def delete_reminder(rid: int) -> dict:
        hub.s.reminders.delete(rid)
        await hub.push_state()
        return {"ok": True}

    @app.post("/api/phone/end")
    async def end_call() -> dict:
        hub.s.phone.end()
        await hub.laptop_event("call_ended")
        await hub.push_state()
        return {"ok": True}

    @app.post("/api/timer/cancel")
    async def cancel_timer() -> dict:
        hub.s.timer.cancel()
        await hub.push_state()
        return {"ok": True}

    @app.post("/api/alarm/dismiss")
    async def dismiss_alarm() -> dict:
        hub.s.alarm.dismiss()
        await hub.push_state()
        return {"ok": True}


    # ---- live benchmark ---------------------------------------------------------
    @app.post("/api/bench/sessions")
    async def bench_create(body: BenchCreate) -> dict:
        if body.set_type not in {"fixed", "random"}:
            raise HTTPException(400, "set_type must be 'fixed' or 'random'")
        if not body.consent_results:
            raise HTTPException(400, "consent_results is required")
        try:
            return await hub.benchmark.create(
                set_type=body.set_type, cond_noise=body.cond_noise,
                cond_distance=body.cond_distance, consent_results=body.consent_results,
                consent_audio=body.consent_audio, alias=body.alias, note=body.note)
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/bench/sessions/{sid}/start")
    async def bench_start(sid: str) -> dict:
        try:
            await hub.benchmark.start(sid)
        except (RuntimeError, KeyError) as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @app.post("/api/bench/trial/retry")
    async def bench_retry() -> dict:
        try:
            await hub.benchmark.retry()
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @app.post("/api/bench/trial/misspoken")
    async def bench_misspoken() -> dict:
        try:
            await hub.benchmark.misspoken()
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @app.post("/api/bench/trial/skip")
    async def bench_skip() -> dict:
        try:
            await hub.benchmark.skip()
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @app.post("/api/bench/sessions/{sid}/finish")
    async def bench_finish(sid: str) -> dict:
        if hub.benchmark.session_id != sid:
            raise HTTPException(409, "session is not the running one")
        try:
            await hub.benchmark.finish()
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @app.post("/api/bench/sessions/{sid}/abandon")
    async def bench_abandon(sid: str) -> dict:
        if hub.benchmark.session_id != sid:
            raise HTTPException(409, "session is not the running one")
        try:
            await hub.benchmark.abandon()
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @app.delete("/api/bench/sessions/{sid}")
    async def bench_delete(sid: str) -> dict:
        if hub.benchmark.running and hub.benchmark.session_id == sid:
            raise HTTPException(409, "cannot delete a running session")
        cids = hub.benchmark_store.delete_session(sid)
        for cid in cids:
            hub.log.label(cid, None)
        await hub.push_state()
        return {"ok": True, "cleared": len(cids)}

    @app.get("/api/bench/sessions")
    async def bench_list() -> list[dict]:
        return hub.benchmark_store.list_sessions()

    @app.get("/api/bench/summary")
    async def bench_summary(model: str = "", set_type: str = Query("", alias="set"),
                            noise: str = "", distance: str = "") -> dict:
        return hub.benchmark_summary(model, set_type, noise, distance)

    @app.get("/api/bench/export.csv", response_class=PlainTextResponse)
    async def bench_export_csv() -> str:
        rows = hub.benchmark_export_rows()
        if not rows:
            return ""
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        return buf.getvalue()

    @app.get("/api/bench/export.json")
    async def bench_export_json() -> list[dict]:
        return hub.benchmark_export_rows()

    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="ui")

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "demo.yaml"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    lc = cfg["laptop"]
    uvicorn.run(create_app(cfg), host=lc.get("host", "127.0.0.1"), port=int(lc.get("port", 8080)))


if __name__ == "__main__":
    main()
