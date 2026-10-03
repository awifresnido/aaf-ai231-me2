"""NTP-style clock offset estimator: known offsets, rtt outliers, readiness, reset.

Also covers the ingress fallback (``feed_ingress``): an edge message's own wall
clock vs the laptop receive time, which lets the trace be corrected even when
the edge does not echo ``t_edge`` in its pong.
"""
import random

from laptop.app.clock_sync import ClockOffset


def _feed(c: ClockOffset, offset_ms: float, rtt_ms: float, t0: float, jitter: float = 0.0) -> None:
    t3 = t0 + rtt_ms
    t_edge = (t0 + t3) / 2.0 + offset_ms + random.uniform(-jitter, jitter)
    c.feed(t0, t_edge, t3)


def test_estimate_offset_13h_within_5ms():
    c = ClockOffset()
    offset = -13 * 3600 * 1000.0
    t0 = 1_000_000.0
    for _ in range(30):
        _feed(c, offset, 10 + random.uniform(0, 2), t0, jitter=1.0)
        t0 += 2000
    assert c.ready
    assert abs(c.offset_ms - offset) < 5


def test_estimate_positive_offset():
    c = ClockOffset()
    offset = 50.0
    t0 = 1_000_000.0
    for _ in range(30):
        _feed(c, offset, 10.0, t0, jitter=0.5)
        t0 += 2000
    assert c.ready
    assert abs(c.offset_ms - offset) < 5


def test_outlier_rtt_ignored():
    c = ClockOffset()
    offset = 50.0
    t0 = 1_000_000.0
    for _ in range(25):
        _feed(c, offset, 10 + random.uniform(0, 2), t0)
        t0 += 2000
    for _ in range(5):  # slow round trips with a wildly wrong offset
        _feed(c, offset + 10_000, 200.0, t0)
        t0 += 2000
    assert c.ready
    assert abs(c.offset_ms - offset) < 5


def test_not_ready_before_3_samples():
    c = ClockOffset()
    t0 = 1_000_000.0
    for _ in range(2):
        _feed(c, 0.0, 10.0, t0)
        t0 += 2000
    assert not c.ready
    assert c.has_estimate          # an offset exists, just not settled
    _feed(c, 0.0, 10.0, t0)
    assert c.ready


def test_ingress_estimates_offset_after_three_samples():
    c = ClockOffset()
    offset = 11 * 3600 * 1000.0     # Pi 11 h ahead of the laptop
    t = 1_700_000_000_000.0
    for _ in range(3):
        c.feed_ingress(t + offset, t)
    assert c.ready
    assert abs(c.offset_ms - offset) < 5


def test_has_estimate_after_one_ingress_sample():
    c = ClockOffset()
    offset = 11 * 3600 * 1000.0
    c.feed_ingress(1_700_000_000_000.0 + offset, 1_700_000_000_000.0)
    assert c.has_estimate
    assert not c.ready              # not settled yet
    assert abs(c.offset_ms - offset) < 5


def test_ntp_preferred_over_ingress():
    c = ClockOffset()
    offset = 50.0
    t0 = 1_000_000.0
    for i in range(10):             # ingress samples biased by ~-5 ms (one-way delay)
        c.feed_ingress(t0 + offset - 5.0 + i * 2000.0, t0 + i * 2000.0)
    c.feed(t0, t0 + offset + 5.0, t0 + 10.0)  # one accurate NTP sample wins
    assert abs(c.offset_ms - offset) < 5
    assert abs(c.rtt_ms - 10.0) < 0.01


def test_reset_clears_ready():
    c = ClockOffset()
    t0 = 1_000_000.0
    for _ in range(10):
        _feed(c, 0.0, 10.0, t0)
        t0 += 2000
    assert c.ready
    c.reset()
    assert not c.ready
    assert c.offset_ms == 0.0


def test_mark_local_ready_zero():
    c = ClockOffset()
    c.mark_local()
    assert c.ready
    assert c.has_estimate
    assert c.offset_ms == 0.0
    assert c.snapshot()["ready"] is True


# ---------------------------------------------------------------- 2026-10-02: jumps, quality
SKEW = 13 * 3600 * 1000.0


def test_ntp_jump_mid_stream_resyncs_after_3_samples():
    """Pi boots 13 h behind, then NTP steps its clock to the right time."""
    c = ClockOffset()
    t0 = 1_000_000.0
    for _ in range(20):
        _feed(c, -SKEW, 10 + random.uniform(0, 2), t0, jitter=1.0)
        t0 += 2000
    assert abs(c.offset_ms + SKEW) < 5 and c.jumps == 0
    for i in range(3):                       # clock stepped: offset now ~ +3 ms
        _feed(c, 3.0, 10 + random.uniform(0, 2), t0, jitter=1.0)
        t0 += 2000
        if i < 2:
            assert abs(c.offset_ms + SKEW) < 5   # not yet confirmed
    assert c.jumps == 1
    assert abs(c.offset_ms - 3.0) < 5
    assert abs(c.last_jump_ms - SKEW) < 10
    assert c.n_samples == 3                   # pre-jump samples discarded
    assert c.ready and c.quality == "accurate"


def test_single_outlier_dropped():
    c = ClockOffset()
    t0 = 1_000_000.0
    for _ in range(10):
        _feed(c, 50.0, 10.0, t0)
        t0 += 2000
    _feed(c, 5050.0, 10.0, t0)                # one wild sample at normal rtt
    t0 += 2000
    for _ in range(5):
        _feed(c, 50.0, 10.0, t0)
        t0 += 2000
    assert abs(c.offset_ms - 50.0) < 5 and c.jumps == 0


def test_inconsistent_pending_samples_do_not_jump():
    c = ClockOffset()
    t0 = 1_000_000.0
    for _ in range(10):
        _feed(c, 0.0, 10.0, t0)
        t0 += 2000
    for off in (5000.0, 9000.0, 13000.0):     # far apart from each other
        _feed(c, off, 10.0, t0)
        t0 += 2000
    assert c.jumps == 0 and abs(c.offset_ms) < 5


def test_ingress_jump_resyncs():
    c = ClockOffset()
    t = 1_700_000_000_000.0
    for i in range(10):
        c.feed_ingress(t + i * 1000 - SKEW, t + i * 1000)
    for i in range(10, 13):
        c.feed_ingress(t + i * 1000 + 20.0, t + i * 1000)   # stepped clock, ~20 ms delay
    assert c.jumps == 1 and abs(c.offset_ms - 20.0) < 5
    assert c.quality == "approximate"


def test_quality_progression_ntp():
    c = ClockOffset()
    assert c.quality == "syncing" and c.source is None
    t0 = 1_000_000.0
    _feed(c, 0.0, 10.0, t0)
    assert c.quality == "syncing" and c.source == "ntp"
    for _ in range(2):
        t0 += 2000
        _feed(c, 0.0, 10.0, t0)
    assert c.quality == "accurate"


def test_ingress_never_accurate_and_ignored_once_ntp_exists():
    c = ClockOffset()
    t = 1_700_000_000_000.0
    for i in range(5):
        c.feed_ingress(t + i + 30.0, t + i)
    assert c.quality == "approximate"
    c.feed(t, t + 5.0 + 0.0, t + 10.0)          # NTP offset 0
    n_before = len(c._ingress)
    c.feed_ingress(t + 999.0, t)                # ignored now
    assert len(c._ingress) == n_before
    assert c.source == "ntp"


def test_ntp_not_ready_when_spread_too_large():
    c = ClockOffset(max_spread_ms=100.0)
    t0 = 1_000_000.0
    for off in (0.0, 300.0, 600.0):             # within jump_ms of each other, but spread 600
        _feed(c, off, 10.0, t0)
        t0 += 2000
    assert not c.ready and c.quality == "syncing"


def test_jump_counter_survives_reset_and_snapshot_fields():
    c = ClockOffset()
    t = 1_700_000_000_000.0
    for i in range(5):
        c.feed_ingress(t + i - SKEW, t + i)
    for i in range(5, 8):
        c.feed_ingress(t + i, t + i)
    assert c.jumps == 1
    c.reset()
    s = c.snapshot()
    assert s["jumps"] == 1 and s["quality"] == "syncing" and s["source"] is None
    assert set(s) >= {"offset_ms", "ready", "has_estimate", "n_samples", "rtt_ms", "spread_ms",
                      "source", "quality", "jumps", "last_jump_ms"}


def test_local_quality():
    c = ClockOffset()
    c.mark_local()
    assert c.quality == "local" and c.source == "local"
    c.feed(0.0, 1e9, 10.0)                      # ignored in local mode
    assert c.offset_ms == 0.0
