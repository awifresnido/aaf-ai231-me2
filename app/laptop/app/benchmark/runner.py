"""Live benchmark runner: the per-session state machine, owned by the Hub.

Flow per trial: PROMPTING -> AWAIT_WAKE -> AWAIT_RESULT -> SCORED -> (auto-advance)
-> next trial, or finished. Operator actions (retry / misspoken / skip / abandon)
can interrupt at any point.
"""
from __future__ import annotations

import asyncio
import random
from typing import Optional

from vcm_common.ontology import Ontology
from vcm_common.protocol import InferenceResult, now_ms

from ..clock_sync import timing_trusted
from .config import BenchmarkConfig, Prompt, generate_random_set
from .store import BenchmarkStore

DEFAULT_AUTO_ADVANCE_S = 1.5


def score_trial(kind: str, expected: Optional[str], accepted: bool,
                predicted: Optional[str]) -> tuple[str, bool]:
    """Pure scoring. Returns (outcome, intent_correct)."""
    if kind == "command":
        if accepted and predicted == expected:
            return "correct", True
        if not accepted:
            return "not_understood", False
        intent_correct = (predicted is not None and expected is not None
                          and predicted.split("|")[0] == expected.split("|")[0])
        return "wrong", intent_correct
    # negative prompt
    if accepted:
        return "false_accept", False
    return "correct_reject", False


class BenchmarkRunner:
    """Owned by the Hub. ``host`` must provide ``edge_state`` (dict), ``log``
    (with ``.label``) and an async ``push_state()``."""

    def __init__(self, store: BenchmarkStore, cfg: BenchmarkConfig, ontology: Ontology,
                 display_names, host, auto_advance_s: float = DEFAULT_AUTO_ADVANCE_S):
        self.store = store
        self.cfg = cfg
        self.o = ontology
        self.dn = display_names
        self.host = host
        self.running = False
        self.session_id: Optional[str] = None
        self.prompts: list[Prompt] = []
        self.idx = 0
        self.attempt = 0
        self.phase = "idle"
        self.current_trial_id: Optional[str] = None
        self.t_prompt_ms: Optional[float] = None
        self.t_wake_ms: Optional[float] = None
        self.last_outcome: Optional[dict] = None
        self.locked_model_id: Optional[str] = None
        self.locked_threshold: Optional[float] = None
        self._timers: list[asyncio.Task] = []
        self.auto_advance_s = auto_advance_s

    # ------------------------------------------------------------------ helpers
    def _lock(self) -> tuple[Optional[str], float]:
        es = getattr(self.host, "edge_state", None) or {}
        model_id = es.get("active_vcm")
        thr = es.get("threshold_override")
        if thr is None:
            for m in es.get("models", []):
                if m.get("id") == model_id:
                    thr = m.get("min_confidence")
                    break
        return model_id, (float(thr) if thr is not None else 0.7)

    async def _network_offline(self) -> bool:
        try:
            _, w = await asyncio.wait_for(
                asyncio.get_running_loop().create_connection(
                    lambda: asyncio.Protocol(), "1.1.1.1", 53), timeout=1.0)
            w.close()
            return False
        except Exception:
            return True

    def _cancel_timers(self) -> None:
        for t in self._timers:
            if not t.done():
                t.cancel()
        self._timers = []

    def _arm(self, seconds: float, coro_fn) -> None:
        async def _run():
            await asyncio.sleep(seconds)
            await coro_fn()
        self._timers.append(asyncio.create_task(_run()))

    async def _push(self) -> None:
        try:
            await self.host.push_state()
        except Exception:  # noqa: BLE001 -- pushing UI state must not kill the runner
            pass

    # ------------------------------------------------------------------ lifecycle
    async def create(self, *, set_type: str, cond_noise: str, cond_distance: str,
                     consent_results: bool, consent_audio: bool,
                     alias: Optional[str] = None, note: Optional[str] = None) -> dict:
        if set_type == "fixed":
            prompts = list(self.cfg.fixed_set)
            seed = None
        elif set_type == "random":
            seed = random.getrandbits(31)
            rs = self.cfg.random_set
            prompts = generate_random_set(self.o, self.cfg.phrases, rs["n_commands"],
                                          rs["n_negatives"], self.cfg.negatives, seed)
        else:
            raise ValueError(f"unknown set_type {set_type!r}")
        model_id, threshold = self._lock()
        offline = await self._network_offline()
        alias = alias or self.store.next_alias()
        sid = self.store.create_session(
            alias=alias, mode="live", set_type=set_type, set_seed=seed,
            model_id=model_id, threshold=threshold, cond_noise=cond_noise,
            cond_distance=cond_distance, network_offline=offline,
            consent_results=consent_results, consent_audio=consent_audio, note=note,
            prompts=[{"kind": p.kind, "say": p.say, "expected_class": p.expected_class}
                     for p in prompts])
        await self._push()
        return self.store.get_session(sid)

    async def start(self, session_id: str) -> None:
        if self.running:
            raise RuntimeError("a session is already running")
        sess = self.store.get_session(session_id)
        if sess is None:
            raise KeyError(session_id)
        self.running = True
        self.session_id = session_id
        self.idx = 0
        self.attempt = 0
        self.locked_model_id = sess["model_id"]
        self.locked_threshold = sess["threshold"]
        self.prompts = self._prompts(session_id)
        self.store.set_status(session_id, "running")
        self._begin_trial()
        await self._push()

    def _prompts(self, sid: str) -> list[Prompt]:
        return [Prompt(kind=t["kind"], say=t["prompt"], expected_class=t["expected_class"])
                for t in self.store.trials_for_session(sid) if t["attempt"] == 0]

    def _begin_trial(self) -> None:
        self._cancel_timers()
        if self.idx >= len(self.prompts):
            self._finish()
            return
        self.attempt = 0
        p = self.prompts[self.idx]
        self.current_trial_id = self.store.trial_at(self.session_id, self.idx, 0)
        self.t_prompt_ms = now_ms()
        self.t_wake_ms = None
        self.last_outcome = None
        self.store.update_trial(self.current_trial_id, t_prompt_ms=self.t_prompt_ms)
        self.phase = "await_wake"
        self._arm(self.cfg.wake_timeout_s, self._wake_timeout)

    # ------------------------------------------------------------------ timers
    async def _wake_timeout(self) -> None:
        if self.phase != "await_wake":
            return
        self.store.update_trial(self.current_trial_id, outcome="wake_miss")
        await self._advance()

    async def _result_timeout(self) -> None:
        if self.phase != "await_result":
            return
        self.store.update_trial(self.current_trial_id, outcome="no_result")
        await self._advance()

    async def _advance(self) -> None:
        self._cancel_timers()
        self.idx += 1
        self.attempt = 0
        self._begin_trial()
        await self._push()

    def _finish(self) -> None:
        self._cancel_timers()
        self.store.set_status(self.session_id, "finished", now_ms())
        self.running = False
        self.phase = "idle"
        self.session_id = None
        self.prompts = []
        self.current_trial_id = None
        self.last_outcome = None

    # ------------------------------------------------------------------ events (from Hub)
    async def on_wake(self, event: dict) -> None:
        if not self.running:
            return
        if self.phase != "await_wake":
            self.store.record_false_wake(self.session_id, event.get("ts_ms") or now_ms())
            await self._push()
            return
        self.t_wake_ms = event.get("ts_ms") or now_ms()
        self.store.update_trial(self.current_trial_id, t_wake_ms=self.t_wake_ms,
                                t_wake_received_ms=event.get("t_received_ms"))
        self.phase = "await_result"
        self._cancel_timers()
        self._arm(self.cfg.result_timeout_s, self._result_timeout)
        await self._push()

    async def on_result(self, r: InferenceResult, action_ms: Optional[float] = None) -> None:
        if not self.running or self.phase != "await_result":
            return
        if r.source != "live":
            return  # ignore replay / simulate
        self._cancel_timers()
        p = self.prompts[self.idx]
        outcome, intent_correct = score_trial(p.kind, p.expected_class, r.accepted,
                                              r.primary.class_key)
        cap = (r.edge_times["t_result_ms"] - r.edge_times["t_capture_end_ms"]
               if r.edge_times else r.t_result_ms - r.t_capture_end_ms)
        rta = (action_ms - r.t_received_ms
               if action_ms is not None and r.t_received_ms is not None else None)
        network = (r.t_received_ms - r.t_result_ms
                   if r.t_received_ms is not None
                   and timing_trusted(r.timing_valid, r.clock_quality) else None)
        fields = dict(
            outcome=outcome, intent_correct=int(intent_correct),
            predicted_class=r.primary.class_key, confidence=r.primary.confidence,
            accepted=int(r.accepted), command_id=r.command_id,
            t_result_ms=r.t_result_ms, inference_ms=r.primary.inference_ms,
            feature_ms=r.primary.feature_ms,
            capture_to_result_ms=cap, result_to_action_ms=rta,
            t_received_ms=r.t_received_ms, network_ms=network,
            timing_valid=int(r.timing_valid), clock_quality=r.clock_quality,
        )
        self.store.update_trial(self.current_trial_id, **fields)
        if p.kind == "command" and p.expected_class:
            self.host.log.label(r.command_id, p.expected_class)
        self.last_outcome = {"outcome": outcome, "predicted": r.primary.class_key,
                             "confidence": round(r.primary.confidence, 3),
                             "inference_ms": r.primary.inference_ms}
        self.phase = "scored"
        self._arm(self.auto_advance_s, self._advance)
        await self._push()

    # ------------------------------------------------------------------ operator actions
    async def retry(self) -> None:
        if not self.running:
            raise RuntimeError("no running session")
        self._cancel_timers()
        self.store.mark_superseded(self.current_trial_id)
        self.attempt += 1
        p = self.prompts[self.idx]
        self.current_trial_id = self.store.add_attempt(
            self.session_id, self.idx, self.attempt, p.kind, p.say, p.expected_class)
        self.t_prompt_ms = now_ms()
        self.t_wake_ms = None
        self.last_outcome = None
        self.store.update_trial(self.current_trial_id, t_prompt_ms=self.t_prompt_ms)
        self.phase = "await_wake"
        self._arm(self.cfg.wake_timeout_s, self._wake_timeout)
        await self._push()

    async def misspoken(self) -> None:
        if not self.running:
            raise RuntimeError("no running session")
        self._cancel_timers()
        self.store.update_trial(self.current_trial_id, misspoken=1)
        self.idx += 1
        self.attempt = 0
        self._begin_trial()
        await self._push()

    async def skip(self) -> None:
        if not self.running:
            raise RuntimeError("no running session")
        self._cancel_timers()
        self.store.update_trial(self.current_trial_id, outcome="skipped")
        self.idx += 1
        self.attempt = 0
        self._begin_trial()
        await self._push()

    async def abandon(self) -> None:
        if not self.running:
            raise RuntimeError("no running session")
        self._cancel_timers()
        self.store.set_status(self.session_id, "abandoned", now_ms())
        self.running = False
        self.phase = "idle"
        self.session_id = None
        self.prompts = []
        self.current_trial_id = None
        self.last_outcome = None
        await self._push()

    async def finish(self) -> None:
        """Operator ends the session early (scores the remaining as skipped)."""
        if not self.running:
            raise RuntimeError("no running session")
        self._cancel_timers()
        for i in range(self.idx, len(self.prompts)):
            tid = self.store.trial_at(self.session_id, i, 0)
            if tid:
                self.store.update_trial(tid, outcome="skipped")
        self._finish()
        await self._push()

    # ------------------------------------------------------------------ state
    def snapshot(self) -> dict:
        if not self.running:
            return {"running": False, "session_id": None, "phase": "idle",
                    "locked": None}
        trials = self.store.trials_for_session(self.session_id)
        counts = {k: 0 for k in ("correct", "not_understood", "wrong", "false_accept",
                                 "correct_reject", "wake_miss", "no_result", "skipped")}
        for t in trials:
            if t["superseded"] or t["misspoken"]:
                continue
            o = t["outcome"]
            if o in counts:
                counts[o] += 1
        p = self.prompts[self.idx] if 0 <= self.idx < len(self.prompts) else None
        return {
            "running": True,
            "session_id": self.session_id,
            "trial_idx": self.idx + 1,
            "trial_total": len(self.prompts),
            "phase": self.phase,
            "prompt": {"kind": p.kind, "say": p.say, "expected_class": p.expected_class}
                      if p else None,
            "last_outcome": self.last_outcome,
            "counts": counts,
            "false_wakes": self.store.false_wake_count(self.session_id),
            "locked": {"model_id": self.locked_model_id, "threshold": self.locked_threshold},
        }
