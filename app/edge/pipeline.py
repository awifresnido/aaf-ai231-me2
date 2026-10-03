"""Edge finite-state machine: wake -> duck -> capture -> VCM -> result.

Transport-agnostic: the WebSocket server (on the Pi) and the laptop's
in-process 'local' mode both drive this same class, so what you test on the
laptop is what runs on the Pi.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Optional

import numpy as np

from vcm_common.manifest import ModelManifest
from vcm_common.ontology import get_ontology
from vcm_common.protocol import (
    AUDIO_SAMPLE_RATE_HZ,
    Event,
    InferenceResult,
    ModelPrediction,
    new_command_id,
    now_ms,
)

from . import telemetry
from .benchmark_log import BenchmarkLog
from .audio import decode_wav, pcm16_to_float, rms_dbfs
from .chime import EdgeChime
from .engines.base import EngineUnavailable, RawOutput
from .engines.scripted import SCRIPTED_MODEL_ID
from .media import MediaController
from .registry import ModelRegistry
from .wake import WakeEngine

log = logging.getLogger(__name__)

Emit = Callable[[dict], Awaitable[None]]

AUDIO_ACTIVE_TIMEOUT_S = 0.5
LEVEL_EMIT_INTERVAL_S = 0.1
TOP_K = 3


class Phase(str, Enum):
    IDLE = "idle"
    WAKE = "wake_detected"
    LISTENING = "listening"
    INFERENCING = "inferencing"
    RESULT = "result"


@dataclass
class EdgeSettings:
    window_s: float = 4.0
    active_vcm: Optional[str] = None
    compare_ids: list[str] = field(default_factory=list)
    active_wake: str = "manual"
    threshold_override: Optional[float] = None
    result_hold_s: float = 1.5
    pre_capture_delay_s: float = 0.35   # let the wake chime finish before recording [s]
    benchmark_mode: bool = False        # vcm-benchmark: log lines on, music + chime off


class EdgePipeline:
    def __init__(self, registry: ModelRegistry, wake_engines: dict[str, WakeEngine],
                 media: MediaController, settings: EdgeSettings, emit: Emit,
                 device_label: str = "edge", chime: Optional[EdgeChime] = None,
                 endpointer=None, endpointing_status: str = "disabled (fixed window)"):
        self.registry = registry
        self.wake_engines = wake_engines
        self.media = media
        self.s = settings
        self.emit = emit
        self.device_label = device_label
        self.chime = chime or EdgeChime(enabled=False)
        self.endpointer = endpointer
        self.endpointing_status = endpointing_status
        self.benchmark_log = BenchmarkLog(enabled=settings.benchmark_mode)
        self.ontology = get_ontology()
        self.phase = Phase.IDLE
        self.started_at = time.time()
        self._busy = asyncio.Lock()
        self._capture: Optional[list[np.ndarray]] = None
        self._endpoint_evt: Optional[asyncio.Event] = None
        self._capture_cid: Optional[str] = None
        self._last_audio_t = 0.0
        self._last_level_emit = 0.0
        self._level_dbfs = -120.0
        self.last_error: Optional[str] = None
        if self.s.active_vcm is None:
            ids = registry.selectable_ids()
            self.s.active_vcm = ids[0] if ids else SCRIPTED_MODEL_ID

    # ------------------------------------------------------------------ emit
    async def _event(self, type_: str, command_id: Optional[str] = None, **payload) -> None:
        ev = Event(type=type_, origin="edge", command_id=command_id, payload=payload)
        await self.emit({"type": "event", "event": ev.model_dump()})

    async def _set_phase(self, phase: Phase, **extra) -> None:
        self.phase = phase
        await self.emit({"type": "phase", "phase": phase.value, "ts_ms": now_ms(), **extra})

    async def publish_state(self) -> None:
        await self.emit({"type": "edge_state", "state": self.snapshot()})

    # ------------------------------------------------------------------ audio
    @property
    def audio_active(self) -> bool:
        return (time.time() - self._last_audio_t) < AUDIO_ACTIVE_TIMEOUT_S

    async def feed_pcm(self, data: bytes) -> None:
        """Continuous 16 kHz PCM16 stream from the microphone."""
        self._last_audio_t = time.time()
        x = pcm16_to_float(data)
        wake = self.wake_engines.get(self.s.active_wake)
        fired: Optional[float] = None
        if self._capture is not None:
            self._capture.append(x)
            if self.endpointer is not None and self.endpointer.active:
                for ev, at_ms in self.endpointer.push(x):
                    asyncio.create_task(self._event(f"vad_{ev}", self._capture_cid, at_ms=at_ms))
                if self.endpointer.done and self._endpoint_evt is not None:
                    self._endpoint_evt.set()
        elif wake is not None and self.phase == Phase.IDLE and not self._busy.locked():
            fired = wake.process(np.frombuffer(data, dtype="<i2"))
        now = time.time()
        if now - self._last_level_emit >= LEVEL_EMIT_INTERVAL_S:
            self._level_dbfs = rms_dbfs(x)
            self._last_level_emit = now
            msg: dict = {"type": "level", "dbfs": round(self._level_dbfs, 1)}
            peak = wake.pop_peak() if wake is not None else None
            if peak is not None:
                msg["wake_score"] = round(peak, 3)
            await self.emit(msg)
        if fired is not None:
            asyncio.create_task(self.run_command("live", wake_confidence=fired,
                                                 wake_source=wake.info.id))

    # ------------------------------------------------------------------ command
    async def run_command(self, source: str, *, wake_confidence: Optional[float] = None,
                          wake_source: str = "manual", clip: Optional[np.ndarray] = None,
                          clip_sr: int = AUDIO_SAMPLE_RATE_HZ, sim_class: Optional[str] = None,
                          expected_class: Optional[str] = None, clip_name: Optional[str] = None
                          ) -> Optional[InferenceResult]:
        if self._busy.locked():
            await self._event("busy_ignored", source=source)
            return None
        async with self._busy:
            return await self._run(source, wake_confidence, wake_source, clip, clip_sr,
                                   sim_class, expected_class, clip_name)

    async def _run(self, source, wake_confidence, wake_source, clip, clip_sr, sim_class,
                   expected_class, clip_name) -> Optional[InferenceResult]:
        cid = new_command_id()
        t_wake = now_ms()
        await self._set_phase(Phase.WAKE, command_id=cid)
        wake_info = self.wake_engines.get(wake_source)
        await self._event("wake_detected", cid, source=source, wake_engine=wake_source,
                          confidence=None if wake_confidence is None else round(wake_confidence, 3),
                          phrase=wake_info.info.phrase if wake_info else None,
                          clip_name=clip_name)
        if self.s.benchmark_mode and source == "live":
            self.benchmark_log.wake()
        if source == "live":
            self.chime.play()
        if self.media.duck():
            await self._event("media_ducked", cid, level=self.media.level)
            await self.publish_state()

        # ---- capture
        if source == "live":
            if not self.audio_active:
                await self._event("capture_failed", cid,
                                  reason="no microphone audio is reaching the edge")
                self.media.apply(None)
                await self._finish()
                return None
            if self.s.pre_capture_delay_s > 0:   # chime + wake-word tail stay out of the clip
                await asyncio.sleep(self.s.pre_capture_delay_s)
            use_vad = self.endpointer is not None
            max_s = self.endpointer.params.max_window_ms / 1000 if use_vad else self.s.window_s
            self._capture_cid = cid
            if use_vad:
                self._endpoint_evt = asyncio.Event()
                self.endpointer.start()
            self._capture = []
            ends = now_ms() + max_s * 1000
            await self._set_phase(Phase.LISTENING, command_id=cid, ends_at_ms=ends,
                                  window_s=max_s, mode="vad" if use_vad else "fixed")
            await self._event("capture_started", cid, window_s=max_s,
                              mode="vad" if use_vad else "fixed",
                              pre_delay_s=self.s.pre_capture_delay_s)
            vad = None
            if use_vad:
                try:
                    await asyncio.wait_for(self._endpoint_evt.wait(), timeout=max_s + 0.5)
                except asyncio.TimeoutError:
                    pass
                self.endpointer.stop()
                vad = self.endpointer.summary()
                await self._event("capture_endpoint", cid, **vad)
            else:
                await asyncio.sleep(self.s.window_s)
            frames, self._capture = self._capture, None
            self._endpoint_evt = None
            if vad and vad["reason"] == "no_speech":
                await self._event("capture_failed", cid, reason="no speech heard")
                self.media.apply(None)
                await self._finish()
                return None
            clip = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
            clip_sr = AUDIO_SAMPLE_RATE_HZ
        else:
            await self._set_phase(Phase.LISTENING, command_id=cid, ends_at_ms=now_ms(),
                                  window_s=0.0)
            await self._event("capture_started", cid, window_s=0.0,
                              note=f"{source} clip, no live capture")
            if clip is None:
                clip = np.zeros(int(0.5 * AUDIO_SAMPLE_RATE_HZ), dtype=np.float32)
            max_n = int(self.s.window_s * clip_sr)
            clip = clip[:max_n]
        t_cap_end = now_ms()
        clip_ms = 1000.0 * clip.size / clip_sr
        await self._event("capture_finished", cid, clip_ms=round(clip_ms, 1),
                          rms_dbfs=round(rms_dbfs(clip), 1))

        # ---- inference
        await self._set_phase(Phase.INFERENCING, command_id=cid)
        await self._event("inference_started", cid)
        primary_id = SCRIPTED_MODEL_ID if source == "simulated" else self.s.active_vcm
        members: list[ModelPrediction] = []
        try:
            entry = self.registry.entries[primary_id]
            manifest = entry.manifest
            agreement = manifest.engine == "agreement" and source != "simulated"
            if source == "simulated":
                self.registry.scripted().next_class = sim_class
            if agreement:
                primary, members = await self._predict_agreement(manifest, clip, clip_sr)
            else:
                primary = await self._predict(primary_id, clip, clip_sr)
        except (EngineUnavailable, Exception) as exc:  # noqa: BLE001 -- surface every failure
            self.last_error = f"{primary_id}: {exc}"
            log.exception("inference failed")
            await self._event("inference_error", cid, model_id=primary_id, detail=str(exc))
            self.media.apply(None)
            await self._finish()
            return None

        compare: list[ModelPrediction] = []
        if source != "simulated" and not agreement:
            for mid in self.s.compare_ids:
                if mid == primary_id:
                    continue
                try:
                    compare.append(await self._predict(mid, clip, clip_sr))
                except Exception as exc:  # noqa: BLE001
                    await self._event("compare_error", cid, model_id=mid, detail=str(exc))

        threshold = (self.s.threshold_override
                     if self.s.threshold_override is not None and source != "simulated"
                     else manifest.reject.min_confidence)
        if agreement:
            accepted, reason = self._accept_agreement(members, manifest, threshold)
        else:
            accepted, reason = self._accept(primary, manifest, threshold)
        media_handled = self.media.apply(primary.intent if accepted else None)

        result = InferenceResult(
            command_id=cid, source=source, primary=primary, accepted=accepted,
            reject_reason=reason, threshold=threshold, compare=compare, members=members,
            clip_ms=clip_ms, clip_rms_dbfs=round(rms_dbfs(clip), 1),
            media_handled=media_handled and accepted, t_wake_ms=t_wake,
            t_capture_end_ms=t_cap_end, t_result_ms=now_ms(), expected_class=expected_class,
        )
        if self.s.benchmark_mode and source == "live":
            if accepted:
                self.benchmark_log.decision(primary.intent, primary.slot or "",
                                            primary.feature_ms + primary.inference_ms, clip_ms)
            else:
                self.benchmark_log.decision("OUT_OF_SCOPE", "",
                                            primary.feature_ms + primary.inference_ms, clip_ms)
        await self.emit({"type": "inference_result", "result": result.model_dump()})
        if media_handled and accepted:
            await self._event("media_action", cid, intent=primary.intent,
                              status=self.media.status, level=self.media.level)
        elif self.media.status == "playing":
            await self._event("media_restored", cid, level=self.media.level)
        await self._set_phase(Phase.RESULT, command_id=cid, accepted=accepted)
        await self.publish_state()
        await asyncio.sleep(self.s.result_hold_s)
        await self._finish()
        return result

    async def _finish(self) -> None:
        self._capture = None
        if self.endpointer is not None:
            self.endpointer.stop()
        wake = self.wake_engines.get(self.s.active_wake)
        if wake:
            wake.reset()
        await self._set_phase(Phase.IDLE)
        await self.publish_state()

    async def _predict(self, model_id: str, clip: np.ndarray, sr: int) -> ModelPrediction:
        engine = self.registry.get(model_id)
        raw: RawOutput = await asyncio.to_thread(engine.infer, clip, sr)
        labels = engine.manifest.labels
        order = np.argsort(raw.probs)[::-1]
        best = int(order[0])
        key = labels[best]
        intent, slot = self.ontology.decode(key)
        return ModelPrediction(
            model_id=model_id, class_key=key, intent=intent, slot=slot,
            confidence=float(raw.probs[best]),
            top_k=[(labels[int(i)], round(float(raw.probs[int(i)]), 4)) for i in order[:TOP_K]],
            feature_ms=round(raw.feature_ms, 2), inference_ms=round(raw.inference_ms, 2),
        )

    async def _predict_agreement(self, m: ModelManifest, clip: np.ndarray, sr: int
                                 ) -> tuple[ModelPrediction, list[ModelPrediction]]:
        """Run every member on the same clip; build the agreement primary.

        The primary is only for display when there is no agreement; acceptance
        is decided separately by ``_accept_agreement``.
        """
        members: list[ModelPrediction] = []
        for mid in m.members:
            members.append(await self._predict(mid, clip, sr))

        # agreed class_key when every member agrees; else the higher-confidence member's
        keys = {p.class_key for p in members}
        if len(keys) == 1:
            agreed_key = members[0].class_key
        else:
            hi = max(members, key=lambda p: p.confidence)
            agreed_key = hi.class_key
        intent, slot = self.ontology.decode(agreed_key)

        # mean top-k over members, summed timings, min confidence
        mean_top: dict[str, float] = {}
        for p in members:
            for k, v in p.top_k:
                mean_top[k] = mean_top.get(k, 0.0) + v / len(members)
        top_k = sorted(mean_top.items(), key=lambda kv: -kv[1])[:TOP_K]

        primary = ModelPrediction(
            model_id=m.id, class_key=agreed_key, intent=intent, slot=slot,
            confidence=round(min(p.confidence for p in members), 4),
            top_k=[(k, round(v, 4)) for k, v in top_k],
            feature_ms=round(sum(p.feature_ms for p in members), 2),
            inference_ms=round(sum(p.inference_ms for p in members), 2),
        )
        return primary, members

    def _accept_agreement(self, members: list[ModelPrediction], m: ModelManifest,
                          threshold: float) -> tuple[bool, Optional[str]]:
        """Accept only if every member agrees on a command class at >= threshold."""
        if not members:
            return False, "agreement has no members"
        keys = {p.class_key for p in members}
        if len(keys) != 1:
            desc = ", ".join(
                f"{self.registry.entries[p.model_id].manifest.display_name} → "
                f"{p.intent} ({p.confidence:.2f})" for p in members)
            return False, f"models disagree: {desc}"
        p0 = members[0]
        if p0.intent in m.reject.non_command_labels or not self.ontology.is_command(p0.intent):
            return False, f"not a command ({p0.intent})"
        for p in members:
            if p.confidence < threshold:
                name = self.registry.entries[p.model_id].manifest.display_name
                return False, f"{name} confidence {p.confidence:.2f} below threshold {threshold:.2f}"
        err = self.ontology.validate(p0.intent, p0.slot)
        if err:
            return False, err
        return True, None

    def _accept(self, p: ModelPrediction, m: ModelManifest, threshold: float
                ) -> tuple[bool, Optional[str]]:
        if p.intent in m.reject.non_command_labels or not self.ontology.is_command(p.intent):
            return False, f"not a command ({p.intent})"
        if p.confidence < threshold:
            return False, f"confidence {p.confidence:.2f} below threshold {threshold:.2f}"
        err = self.ontology.validate(p.intent, p.slot)
        if err:
            return False, err
        return True, None

    # ------------------------------------------------------------------ control
    async def handle(self, msg: dict) -> None:
        """Control messages from the laptop (same schema remote or in-process)."""
        t = msg.get("type")
        if t == "wake_manual":
            asyncio.create_task(self.run_command("live", wake_source="manual"))
        elif t == "simulate":
            key = msg.get("class_key", "")
            if key not in self.registry.scripted().manifest.labels:
                await self._event("simulate_rejected", detail=f"unknown class '{key}'")
                return
            asyncio.create_task(self.run_command("simulated", wake_source="simulator",
                                                 sim_class=key))
        elif t == "replay":
            try:
                clip, sr = decode_wav(base64.b64decode(msg["wav_b64"]))
            except Exception as exc:  # noqa: BLE001
                await self._event("replay_rejected", detail=f"cannot read WAV: {exc}")
                return
            asyncio.create_task(self.run_command(
                "replay", wake_source="replay", clip=clip, clip_sr=sr,
                expected_class=msg.get("expected_class"), clip_name=msg.get("name")))
        elif t == "select_model":
            mid = msg.get("model_id")
            try:
                await asyncio.to_thread(self.registry.get, mid)
                self.s.active_vcm = mid
                m = self.registry.entries[mid].manifest
                excluded = {mid, *m.members}          # agreement members drop out of compare
                self.s.compare_ids = [i for i in self.s.compare_ids if i not in excluded]
                await self._event("model_selected", model_id=mid)
            except Exception as exc:  # noqa: BLE001
                await self._event("model_load_failed", model_id=mid, detail=str(exc))
            await self.publish_state()
        elif t == "set_compare":
            ids = [i for i in msg.get("model_ids", [])
                   if i in self.registry.entries
                   and self.registry.entries[i].manifest.engine != "agreement"]
            for mid in ids:
                try:
                    await asyncio.to_thread(self.registry.get, mid)
                except Exception as exc:  # noqa: BLE001
                    await self._event("model_load_failed", model_id=mid, detail=str(exc))
            self.s.compare_ids = [i for i in ids if self.registry.entries[i].status == "ready"]
            await self.publish_state()
        elif t == "select_wake":
            wid = msg.get("wake_id")
            if wid in self.wake_engines and self.wake_engines[wid].info.status == "ready":
                self.wake_engines[wid].reset()
                self.s.active_wake = wid
                await self._event("wake_selected", wake_id=wid)
            await self.publish_state()
        elif t == "set_threshold":
            v = msg.get("value")
            self.s.threshold_override = None if v is None else max(0.0, min(1.0, float(v)))
            await self.publish_state()
        elif t == "media":
            action = msg.get("action")
            fn = {"play": self.media.play, "pause": self.media.pause, "stop": self.media.stop,
                  "next": self.media.next, "volume_up": self.media.volume_up,
                  "volume_down": self.media.volume_down}.get(action)
            if fn:
                fn()
                if not self.media.ducked:
                    self.media.backend.set_volume_percent(self.media.level_pct)
                await self._event("media_manual", action=action, status=self.media.status,
                                  level=self.media.level)
                await self.publish_state()
        elif t == "rescan":
            await asyncio.to_thread(self.registry.scan)
            await self._event("registry_rescanned", count=len(self.registry.selectable_ids()))
            await self.publish_state()
        elif t == "ping":
            await self.emit({"type": "pong", "t": msg.get("t"), "t_edge": time.time() * 1000})
        elif t == "get_state":
            await self.publish_state()

    async def startup(self) -> None:
        """Load the active and challenger models so the first command is not slow."""
        for mid in [self.s.active_vcm, *self.s.compare_ids]:
            try:
                await asyncio.to_thread(self.registry.get, mid)
            except Exception as exc:  # noqa: BLE001 -- stay up; the UI shows why
                self.last_error = f"{mid}: {exc}"
                log.warning("model %s not loaded: %s", mid, exc)
        self.s.compare_ids = [i for i in self.s.compare_ids
                              if i in self.registry.entries
                              and self.registry.entries[i].status == "ready"]

    # ------------------------------------------------------------------ state
    def health(self) -> dict:
        active = self.registry.entries.get(self.s.active_vcm or "")
        wake = self.wake_engines.get(self.s.active_wake)
        return {
            "device": self.device_label,
            "uptime_s": round(time.time() - self.started_at),
            "audio_stream": self.audio_active,
            "level_dbfs": round(self._level_dbfs, 1) if self.audio_active else None,
            "cpu_percent": telemetry.cpu_percent(),
            "process_rss_mb": telemetry.process_rss_mb(),
            "system_mem_used_mb": telemetry.system_mem_used_mb(),
            "vcm_ready": bool(active and active.status == "ready"),
            "wake_ready": bool(wake and wake.info.status == "ready"),
            "media_ok": self.media.backend.available,
            "vad": self.endpointing_status,
            "last_error": self.last_error,
        }

    def snapshot(self) -> dict:
        return {
            "phase": self.phase.value,
            "window_s": self.s.window_s,
            "active_vcm": self.s.active_vcm,
            "compare_ids": list(self.s.compare_ids),
            "active_wake": self.s.active_wake,
            "threshold_override": self.s.threshold_override,
            "models": self.registry.summaries(),
            "wake_engines": [w.summary() for w in self.wake_engines.values()],
            "pre_capture_delay_s": self.s.pre_capture_delay_s,
            "edge_chime": self.chime.enabled,
            "endpointing": self.endpointing_status,
            "media": self.media.snapshot(),
            "health": self.health(),
        }

    async def health_loop(self, interval_s: float = 2.0) -> None:
        while True:
            await asyncio.sleep(interval_s)
            await self.publish_state()
