"""NTP-style clock-offset estimator for the edge link.

The Raspberry Pi has no RTC battery and no NTP on a direct Ethernet link, so it
boots with a stale wall clock -- and on a network WITH internet its clock is
stepped (often by hours) when NTP syncs, possibly mid-session. The laptop
measures the offset and converts every edge timestamp into laptop time where
it enters the app.

Two sample sources:

* ``feed``         -- NTP-style ping/pong round trip (edge echoes ``t_edge``).
                      Unbiased: offset = t_edge - (t0 + t3) / 2. Source "ntp".
* ``feed_ingress`` -- an edge message's own wall clock vs the laptop receive
                      time. Biased by the one-way delivery delay (Wi-Fi: ms to
                      tens of ms) but needs no edge support. Source "ingress".
                      Ignored as soon as any NTP sample exists.

Jump detection (2026-10-02): a sample more than ``jump_ms`` away from the
current estimate is held as *pending*. Three consecutive pending samples that
agree with each other mean the Pi's clock really stepped: the window is
replaced by them (``jumps`` += 1, ``last_jump_ms`` = new - old). A lone
outlier is dropped; a slow (high-rtt) NTP outlier is dropped immediately.

``offset_ms`` is ``edge_time - laptop_time`` (negative when the Pi is behind).
``has_estimate`` -> an offset exists (enough to correct displayed clock times).
``ready``        -> settled (>= 3 samples; NTP also needs spread <= max_spread_ms);
                    gates every cross-device latency.
``quality``      -> "local" | "accurate" (NTP, ready) | "approximate" (ingress,
                    ready) | "syncing". Only local/accurate count for
                    end-to-end latency metrics.
"""
from __future__ import annotations

import statistics
from typing import Optional

TRUSTED_QUALITIES = ("local", "accurate")


def timing_trusted(valid, quality: Optional[str]) -> bool:
    """Cross-device latency is trustworthy: timing_valid and an accurate clock.
    ``quality is None`` = recorded before quality existed; judged by validity alone."""
    return bool(valid) and (quality is None or quality in TRUSTED_QUALITIES)


class ClockOffset:
    def __init__(self, max_samples: int = 30, jump_ms: float = 1000.0,
                 confirm: int = 3, max_spread_ms: float = 100.0, min_ready: int = 3):
        self._ntp: list[tuple[float, float]] = []   # (offset_ms, rtt_ms)
        self._ingress: list[float] = []             # offset_ms
        self._pend_ntp: list[tuple[float, float]] = []
        self._pend_ingress: list[float] = []
        self._max = max_samples
        self.jump_ms = jump_ms
        self.confirm = confirm
        self.max_spread_ms = max_spread_ms
        self.min_ready = min_ready
        self._local = False
        self.offset_ms = 0.0
        self.ready = False
        self.rtt_ms: Optional[float] = None
        self.spread_ms: Optional[float] = None
        self.n_samples = 0
        self.jumps = 0                      # cumulative; survives reset() (hub compares counts)
        self.last_jump_ms: Optional[float] = None

    # ------------------------------------------------------------------ status
    @property
    def has_estimate(self) -> bool:
        """True when an offset is available to convert timestamps."""
        return self.ready or self.n_samples >= 1

    @property
    def source(self) -> Optional[str]:
        if self._local:
            return "local"
        if self._ntp:
            return "ntp"
        if self._ingress:
            return "ingress"
        return None

    @property
    def quality(self) -> str:
        if self._local:
            return "local"
        if not self.ready:
            return "syncing"
        src = self.source
        if src == "ntp":
            return "accurate"
        if src == "ingress":
            return "approximate"
        return "unknown"            # ready set externally without samples (tests/tools)

    # ------------------------------------------------------------------ input
    def feed(self, t0: float, t_edge: float, t3: float) -> None:
        """One ping/pong: t0 = laptop send, t_edge = edge reply wall time, t3 = laptop receive."""
        if self._local:
            return
        rtt = t3 - t0
        self._admit("ntp", t_edge - (t0 + t3) / 2.0, rtt)
        self._recompute()

    def feed_ingress(self, ts_edge: float, t_received: float) -> None:
        """Coarse sample from an edge message's wall clock vs laptop receive time."""
        if self._local or self._ntp:          # NTP present -> ingress not needed
            return
        self._admit("ingress", ts_edge - t_received, None)
        self._recompute()

    # ------------------------------------------------------------------ internals
    def _ntp_used(self, window: list[tuple[float, float]]) -> tuple[list[float], Optional[float]]:
        if not window:
            return [], None
        min_rtt = min(r for _, r in window)
        return [o for o, r in window if r <= 1.5 * min_rtt], min_rtt

    def _estimate(self, kind: str) -> Optional[float]:
        if kind == "ntp":
            used, _ = self._ntp_used(self._ntp)
        else:
            used = list(self._ingress)
        return statistics.median(used) if used else None

    def _admit(self, kind: str, offset: float, rtt: Optional[float]) -> None:
        win = self._ntp if kind == "ntp" else self._ingress
        pend = self._pend_ntp if kind == "ntp" else self._pend_ingress
        item = (offset, rtt) if kind == "ntp" else offset
        current = self._estimate(kind)
        if current is None or abs(offset - current) <= self.jump_ms:
            win.append(item)
            pend.clear()
            del win[:-self._max]
            return
        # far from the current estimate: a slow NTP round trip is just noise
        if kind == "ntp":
            _, min_rtt = self._ntp_used(self._ntp)
            if min_rtt is not None and rtt is not None and rtt > 1.5 * min_rtt:
                return
        pend.append(item)
        del pend[:-self.confirm]
        offs = [p[0] for p in pend] if kind == "ntp" else list(pend)
        if len(pend) == self.confirm and max(offs) - min(offs) <= self.jump_ms:
            new = statistics.median(offs)
            self.jumps += 1
            self.last_jump_ms = new - current
            win.clear()
            win.extend(pend)
            pend.clear()
            # the other source's samples predate the jump too
            other_win = self._ingress if kind == "ntp" else self._ntp
            other_win.clear()
            (self._pend_ingress if kind == "ntp" else self._pend_ntp).clear()

    def _recompute(self) -> None:
        if self._ntp:
            used, min_rtt = self._ntp_used(self._ntp)
            self.rtt_ms = min_rtt
        elif self._ingress:
            used = list(self._ingress)
            self.rtt_ms = None
        else:
            used = []
        if not used:
            self.offset_ms = 0.0
            self.ready = False
            self.rtt_ms = None
            self.spread_ms = None
            self.n_samples = 0
            return
        self.offset_ms = statistics.median(used)
        self.n_samples = len(used)
        self.spread_ms = (max(used) - min(used)) if len(used) >= 2 else 0.0
        enough = self.n_samples >= self.min_ready
        if self._ntp:
            self.ready = enough and self.spread_ms <= self.max_spread_ms
        else:
            self.ready = enough

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """New connection: forget samples (jump counter is kept)."""
        self._ntp.clear()
        self._ingress.clear()
        self._pend_ntp.clear()
        self._pend_ingress.clear()
        self.offset_ms = 0.0
        self.ready = False
        self.rtt_ms = None
        self.spread_ms = None
        self.n_samples = 0

    def mark_local(self) -> None:
        """Same process = same clock: offset is exactly 0 and always ready."""
        self._local = True
        self.reset()
        self.offset_ms = 0.0
        self.ready = True
        self.rtt_ms = 0.0
        self.spread_ms = 0.0
        self.n_samples = 0

    def snapshot(self) -> dict:
        return {
            "offset_ms": round(self.offset_ms, 2),
            "ready": self.ready,
            "has_estimate": self.has_estimate,
            "n_samples": self.n_samples,
            "rtt_ms": round(self.rtt_ms, 2) if self.rtt_ms is not None else None,
            "spread_ms": round(self.spread_ms, 2) if self.spread_ms is not None else None,
            "source": self.source,
            "quality": self.quality,
            "jumps": self.jumps,
            "last_jump_ms": round(self.last_jump_ms, 1) if self.last_jump_ms is not None else None,
        }
