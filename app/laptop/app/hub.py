"""Application hub: receives edge messages, dispatches actions, keeps the
authoritative UI state, and pushes it to every browser over WebSocket."""
from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections import deque
from pathlib import Path
from typing import Optional

from fastapi import WebSocket

from vcm_common.display_names import DisplayNames
from vcm_common.ontology import get_ontology
from vcm_common.protocol import Event, InferenceResult, now_ms

from .clock_sync import ClockOffset
from .command_log import CommandLog
from .dispatcher import Dispatcher, Outcome, Services
from .benchmark import metrics as benchmark_metrics
from .benchmark.config import load_benchmark_config
from .benchmark.runner import BenchmarkRunner
from .benchmark.store import BenchmarkStore
from .edge_link import EdgeLink, LocalEdgeLink, RemoteEdgeLink
from .mic import MicStreamer
from .services.clocks import AlarmService, TimerService
from .services.devices import LightService, ThermostatService
from .services.phone import PhoneService
from .services.reminders import ReminderStore
from .services.weather import WeatherService

log = logging.getLogger(__name__)
TIMELINE_MAX = 60
REPLAY_WAIT_S = 20.0


class Hub:
    def __init__(self, cfg: dict, repo_root: Path):
        self.cfg = cfg
        lcfg = cfg["laptop"]
        data_dir = Path(lcfg.get("data_dir", "runtime")).expanduser()
        data_dir = (data_dir if data_dir.is_absolute() else repo_root / data_dir).resolve()
        self.ontology = get_ontology()
        self.display_names = DisplayNames.load(repo_root / "config" / "display_names.yaml")
        missing = self.display_names.missing_model_names(
            [p.parent.name for p in (repo_root / "models" / "vcm").glob("*/manifest.yaml")])
        if missing:
            raise RuntimeError(
                f"display_names.yaml has no entry for registered models: {sorted(missing)}")
        self.clients: set[WebSocket] = set()
        self.timeline: deque[dict] = deque(maxlen=TIMELINE_MAX)
        self.edge_state: Optional[dict] = None
        self.phase: dict = {"phase": "disconnected"}
        self.level_dbfs: Optional[float] = None
        self.last_result: Optional[dict] = None
        self._idle = asyncio.Event()
        self._idle.set()

        self.log = CommandLog(data_dir / "commands.sqlite")
        self.clock = ClockOffset()
        self._clock_jumps_seen = 0
        self.benchmark_cfg = load_benchmark_config(
            repo_root / "config" / "benchmark.yaml", self.ontology)
        self.benchmark_store = BenchmarkStore(data_dir / "benchmark.sqlite")
        self.benchmark = BenchmarkRunner(self.benchmark_store, self.benchmark_cfg,
                                         self.ontology, self.display_names, self)
        services = Services(
            light=LightService(self.ontology),
            thermostat=ThermostatService(self.ontology),
            timer=TimerService(self.notify),
            alarm=AlarmService(self.notify, lcfg.get("timezone", "Asia/Manila")),
            reminders=ReminderStore(data_dir / "reminders.sqlite"),
            phone=PhoneService(lcfg["contacts"]["default"], lcfg["message"]["default_text"],
                               self.notify),
            weather=WeatherService(lcfg["weather"], repo_root / "data" / "weather_fixture.json"),
        )
        self.s = services
        self.dispatcher = Dispatcher(self.ontology, services, lcfg.get("timezone", "Asia/Manila"))

        link_cfg = cfg["edge_link"]
        self.link: EdgeLink
        if link_cfg.get("mode", "local") == "remote":
            self.link = RemoteEdgeLink(link_cfg["host"], int(link_cfg["port"]),
                                       self.on_edge_message, self.on_link_status,
                                       self.clock)
        else:
            self.clock.mark_local()
            self.link = LocalEdgeLink(cfg, repo_root, self.on_edge_message)
        mic_cfg = lcfg.get("mic", {})
        self.mic = MicStreamer(bool(mic_cfg.get("enabled")), mic_cfg.get("device"),
                               self.link.send_audio)

    async def start(self) -> None:
        await self.link.start()
        await self.mic.start()
        asyncio.create_task(self.s.alarm.loop())

    # ------------------------------------------------------------------ browser I/O
    async def _send_all(self, msg: dict) -> None:
        text = json.dumps(msg, default=str)
        for ws in list(self.clients):
            try:
                await ws.send_text(text)
            except Exception:  # noqa: BLE001
                self.clients.discard(ws)

    async def push_state(self) -> None:
        await self._send_all({"kind": "state", "state": self.snapshot()})

    async def add_event(self, ev: dict) -> None:
        self.timeline.append(ev)
        await self._send_all({"kind": "event", "event": ev})

    async def laptop_event(self, type_: str, command_id: Optional[str] = None, **payload) -> None:
        ev = Event(type=type_, origin="laptop", command_id=command_id, payload=payload)
        await self.add_event(ev.model_dump())

    async def notify(self, type_: str, payload: dict) -> None:
        """Callback for services (timer done, alarm ringing, call connected)."""
        await self.laptop_event(type_, **payload)
        await self.push_state()

    # ------------------------------------------------------------------ edge I/O
    async def on_link_status(self) -> None:
        if not self.link.connected:
            self.phase = {"phase": "disconnected"}
            self.edge_state = None
            self._idle.set()
        await self.laptop_event("edge_connected" if self.link.connected else "edge_disconnected",
                                url=self.link.url)
        await self.push_state()

    # ------------------------------------------------------------------ clock sync
    def _normalize_event(self, event: dict) -> dict:
        """Convert an edge event ts_ms to laptop time; keep the raw Pi time.

        The displayed time is converted as soon as any offset exists
        (``has_estimate``); cross-device timing is only *valid* once the offset
        is settled (``ready``) -- the same rule as results. ``clock_quality``
        says how far to trust it (local / accurate / approximate / syncing).
        """
        if event.get("origin") != "edge":
            return event
        ev = dict(event)
        ts_edge = ev.get("ts_ms")
        ev["ts_edge_ms"] = ts_edge
        ev["t_received_ms"] = now_ms()
        # Ingress sample (only used while the edge sends no ``t_edge`` pong):
        # an edge event's own wall clock vs our receive time.
        if ts_edge is not None:
            self.clock.feed_ingress(ts_edge, ev["t_received_ms"])
        if ts_edge is not None and self.clock.has_estimate:
            ev["ts_ms"] = ts_edge - self.clock.offset_ms
        ev["timing_valid"] = bool(self.clock.ready)
        ev["clock_quality"] = self.clock.quality
        return ev

    def _normalize_result(self, result: dict) -> dict:
        """Convert an InferenceResult edge wall-clock fields to laptop time,
        keeping the originals in edge_times. The offset is measured from
        ping/pong when available and from ingress otherwise; latency metrics
        are only trusted once the offset is settled (``ready``)."""
        r = dict(result)
        r["t_received_ms"] = now_ms()
        if r.get("t_result_ms") is not None:
            self.clock.feed_ingress(r["t_result_ms"], r["t_received_ms"])
        if self.clock.has_estimate:
            offset = self.clock.offset_ms
            r["edge_times"] = {
                "t_capture_end_ms": r.get("t_capture_end_ms"),
                "t_result_ms": r.get("t_result_ms"),
                "t_wake_ms": r.get("t_wake_ms"),
            }
            r["t_capture_end_ms"] = r["t_capture_end_ms"] - offset
            r["t_result_ms"] = r["t_result_ms"] - offset
            if r.get("t_wake_ms") is not None:
                r["t_wake_ms"] = r["t_wake_ms"] - offset
            r["edge_clock_offset_ms"] = offset
            r["timing_valid"] = self.clock.ready
        else:
            r["edge_clock_offset_ms"] = None
            r["timing_valid"] = False
        r["clock_quality"] = self.clock.quality
        return r

    async def _check_clock_jump(self) -> None:
        """Surface a detected Pi clock step in the trace (once per jump)."""
        if self.clock.jumps != self._clock_jumps_seen:
            self._clock_jumps_seen = self.clock.jumps
            await self.laptop_event("edge_clock_jump",
                                    delta_ms=round(self.clock.last_jump_ms or 0.0, 1),
                                    offset_ms=round(self.clock.offset_ms, 1))

    async def on_edge_message(self, msg: dict) -> None:
        t = msg.get("type")
        # pongs are consumed by the link; a jump they reveal shows up here on the
        # next edge message (level meters arrive every ~100 ms)
        await self._check_clock_jump()
        if t == "level":
            self.level_dbfs = msg.get("dbfs")
            await self._send_all({"kind": "level", "dbfs": self.level_dbfs})
        elif t == "event":
            ev = self._normalize_event(msg["event"])
            await self._check_clock_jump()        # an ingress sample may have confirmed a jump
            await self.add_event(ev)
            if ev.get("type") == "wake_detected":
                await self.benchmark.on_wake(ev)
        elif t == "phase":
            self.phase = {k: v for k, v in msg.items() if k != "type"}
            if self.phase.get("phase") == "idle":
                self._idle.set()
            else:
                self._idle.clear()
            await self.push_state()
        elif t == "edge_state":
            self.edge_state = msg["state"]
            if self.phase.get("phase") == "disconnected":
                self.phase = {"phase": self.edge_state.get("phase", "idle")}
            await self.push_state()
        elif t == "inference_result":
            normalized = self._normalize_result(msg["result"])
            await self._check_clock_jump()
            await self._on_result(InferenceResult(**normalized))

    async def _on_result(self, r: InferenceResult) -> None:
        received = now_ms()
        p = r.primary
        slot_txt = f" [{p.slot}]" if p.slot else ""
        await self.laptop_event("result_received", r.command_id, model_id=p.model_id,
                                intent=p.intent, slot=p.slot, confidence=round(p.confidence, 3),
                                accepted=r.accepted, source=r.source,
                                summary=f"{p.intent}{slot_txt}")
        try:
            outcome = await self.dispatcher.dispatch(r, received)
        except Exception as exc:  # noqa: BLE001 -- a bad command must not crash the app
            log.exception("dispatch failed")
            outcome = Outcome(False, "dispatch error", detail=str(exc))
        done = now_ms()
        await self.laptop_event("action_completed" if outcome.executed else "action_skipped",
                                r.command_id, action=outcome.action, target=outcome.target,
                                detail=outcome.detail, latency_ms=round(done - received, 1))
        row = self.log.record(r, outcome, done)
        self.last_result = {
            **r.model_dump(), "outcome": outcome.__dict__, "received_ms": received,
            "action_ms": done, "log": row,
        }
        await self.benchmark.on_result(r, done)
        await self.push_state()

    # ------------------------------------------------------------------ commands from UI
    async def to_edge(self, msg: dict) -> None:
        await self.link.send(msg)

    async def replay_batch(self, items: list[dict]) -> None:
        """Send WAVs one at a time; wait for the edge to return to idle between them."""
        for item in items:
            try:
                await asyncio.wait_for(self._idle.wait(), REPLAY_WAIT_S)
            except asyncio.TimeoutError:
                await self.laptop_event("replay_aborted", detail="edge stayed busy")
                return
            self._idle.clear()
            await self.link.send({"type": "replay", **item})
            await asyncio.sleep(0.2)
        await self.laptop_event("replay_batch_sent", count=len(items))

    # ------------------------------------------------------------------ state
    def _scrub_dataset_ids(self, edge: dict) -> dict:
        """Return the edge payload with internal dataset ids aliased to their
        public ``display_id`` (e.g. ``mark_optionb`` -> ``synthetic_tts``). The
        manifests keep the internal id; only the UI-facing snapshot is scrubbed."""
        if not edge or "models" not in edge:
            return edge
        edge = copy.deepcopy(edge)
        for m in edge.get("models", []):
            lt = m.get("lineage") or {}
            if lt.get("training_data"):
                lt["training_data"] = self.display_names.dataset_display_id(lt["training_data"])
            for split in ("training", "validation", "test"):
                sc = (m.get("scores") or {}).get(split)
                if sc and sc.get("dataset_id"):
                    sc["dataset_id"] = self.display_names.dataset_display_id(sc["dataset_id"])
        return edge

    def snapshot(self) -> dict:
        now = now_ms()
        d = self.dispatcher
        weather = self.s.weather.last if now < d.weather_card_until_ms else None
        return {
            "server_ms": now,
            "link": {"mode": self.link.mode, "connected": self.link.connected,
                     "url": self.link.url, "rtt_ms": self.link.rtt_ms},
            "edge_clock": self.clock.snapshot(),
            "edge": self._scrub_dataset_ids(self.edge_state),
            "phase": self.phase,
            "level_dbfs": self.level_dbfs,
            "last_result": self.last_result,
            "mic": self.mic.snapshot(),
            "internet_ok": self.s.weather.internet_ok,
            "devices": {
                "light": self.s.light.snapshot(),
                "thermostat": self.s.thermostat.snapshot(),
                "timer": self.s.timer.snapshot(),
                "alarm": self.s.alarm.snapshot(),
                "reminders": self.s.reminders.snapshot(),
                "phone": self.s.phone.snapshot(),
                "weather": weather,
                "weather_last": self.s.weather.last,
                "toast": d.toast if d.toast and now < d.toast["until_ms"] else None,
            },
            "commands": self.log.recent(),
            "scoreboard": self.log.scoreboard(),
            "benchmark": self.benchmark.snapshot(),
            "display_names": self.display_names.as_dict(),
            "ontology": {"leaf_labels": self.ontology.leaf_labels(),
                         "slot_values": {k: list(v) for k, v in self.ontology.slot_values.items()},
                         "schema_version": self.ontology.schema_version},
            "timeline": list(self.timeline),
        }

    # ------------------------------------------------------------------ benchmark
    def _model_facts(self, model_id: Optional[str]) -> dict:
        es = self.edge_state or {}
        m = next((x for x in es.get("models", []) if x.get("id") == model_id), None)
        engine = m.get("engine") if m else None
        runtime = {"onnx_tinyvcm": "ONNX Runtime", "torch_tinyvcm": "PyTorch",
                   "agreement": "Agreement", "scripted": "Simulated"}.get(engine or "", engine)
        size_bytes = m.get("size_bytes") if m else None
        return {"params": m.get("params") if m else None,
                "size_kb": round(size_bytes / 1024, 1) if size_bytes else None,
                "runtime": runtime,
                "name": self.display_names.model_name(model_id) if model_id else None}

    def benchmark_summary(self, model: str = "", set_type: str = "", noise: str = "",
                          distance: str = "") -> dict:
        trials = self.benchmark_store.all_scored_trials()
        if model:
            trials = [t for t in trials if t["model_id"] == model]
        if set_type:
            trials = [t for t in trials if t["set_type"] == set_type]
        if noise:
            trials = [t for t in trials if t["cond_noise"] == noise]
        if distance:
            trials = [t for t in trials if t["cond_distance"] == distance]
        target_model = model or (self.edge_state or {}).get("active_vcm")
        sessions = self.benchmark_store.list_sessions()
        false_wakes = sum(self.benchmark_store.false_wake_count(s["session_id"])
                          for s in sessions if (not model or s["model_id"] == model))
        fixed_labels = {i: p.say for i, p in enumerate(self.benchmark_cfg.fixed_set)}
        return benchmark_metrics.compute(trials, self.benchmark_cfg.targets,
                                         self._model_facts(target_model), false_wakes,
                                         fixed_labels)

    def benchmark_export_rows(self) -> list[dict]:
        cols = ["session_id", "alias", "set_type", "model_id", "cond_noise", "cond_distance",
                "created_ms", "idx", "attempt", "kind", "prompt", "expected_class", "outcome",
                "intent_correct", "misspoken", "superseded", "predicted_class", "confidence",
                "inference_ms", "feature_ms", "capture_to_result_ms", "result_to_action_ms"]
        return [{c: t.get(c) for c in cols} for t in self.benchmark_store.all_scored_trials()]

