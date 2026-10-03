"""Edge timestamps are converted to laptop time at ingress (13 h skew).

Also covers the ingress fallback: with no ``t_edge`` pong, the first edge event
of a command still establishes the offset and gets converted.
"""
import yaml

from laptop.app.clock_sync import ClockOffset
from laptop.app.hub import Hub
from laptop.app.main import REPO_ROOT
from vcm_common.protocol import now_ms

SKEW_MS = 13 * 3600 * 1000.0


def _hub(tmp_path):
    c = yaml.safe_load((REPO_ROOT / "config" / "demo.yaml").read_text())
    c["edge_service"]["models_dir"] = str(tmp_path / "models")
    c["edge_service"]["compare_ids"] = []
    c["laptop"]["data_dir"] = str(tmp_path / "runtime")
    return Hub(c, REPO_ROOT)


def test_event_13h_behind_converted(tmp_path):
    hub = _hub(tmp_path)
    hub.clock.offset_ms = -SKEW_MS
    hub.clock.ready = True
    ts_laptop = 1_700_000_000_000.0
    ts_edge = ts_laptop - SKEW_MS
    out = hub._normalize_event({"type": "wake_detected", "origin": "edge", "ts_ms": ts_edge,
                                "command_id": "c1", "payload": {}})
    assert abs(out["ts_ms"] - ts_laptop) < 5
    assert out["ts_edge_ms"] == ts_edge
    assert out["timing_valid"] is True
    assert "t_received_ms" in out


def test_not_ready_marks_timing_invalid(tmp_path):
    hub = _hub(tmp_path)
    hub.clock.ready = False
    out = hub._normalize_event({"type": "wake_detected", "origin": "edge", "ts_ms": 12345.0,
                                "payload": {}})
    assert out["timing_valid"] is False
    assert out["ts_ms"] == 12345.0  # unconverted


def test_result_13h_behind_converted(tmp_path):
    hub = _hub(tmp_path)
    hub.clock.offset_ms = -SKEW_MS
    hub.clock.ready = True
    ts_laptop = 1_700_000_000_000.0
    cap_edge = ts_laptop - SKEW_MS
    res_edge = cap_edge + 40.0
    r = {"command_id": "c1", "source": "live", "accepted": True, "threshold": 0.8,
         "clip_ms": 500.0, "t_capture_end_ms": cap_edge, "t_result_ms": res_edge}
    out = hub._normalize_result(r)
    assert abs(out["t_result_ms"] - (ts_laptop + 40.0)) < 5
    assert out["edge_times"]["t_result_ms"] == res_edge
    assert out["edge_times"]["t_capture_end_ms"] == cap_edge
    assert out["timing_valid"] is True
    assert out["edge_clock_offset_ms"] == -SKEW_MS


def test_result_not_ready_no_conversion(tmp_path):
    hub = _hub(tmp_path)
    hub.clock.ready = False
    r = {"command_id": "c1", "source": "live", "accepted": True, "threshold": 0.8,
         "clip_ms": 500.0, "t_capture_end_ms": 1000.0, "t_result_ms": 1040.0}
    out = hub._normalize_result(r)
    assert out["timing_valid"] is False
    assert out["edge_clock_offset_ms"] is None
    assert out["t_result_ms"] == 1040.0  # unconverted


def test_ingress_event_establishes_offset_and_converts(tmp_path):
    """Old edge code (no ``t_edge``): the first event feeds an ingress sample,
    so the very first event's displayed time is converted -- but cross-device
    timing is not *valid* until the offset is settled (2026-10-02: same rule as
    results; one coarse sample is not enough)."""
    hub = _hub(tmp_path)
    hub.clock = ClockOffset()          # fresh, remote-style (not local)
    skew = 11 * 3600 * 1000.0          # Pi 11 h ahead
    ts_edge = now_ms() + skew
    out = hub._normalize_event({"type": "wake_detected", "origin": "edge",
                                "ts_ms": ts_edge, "command_id": "c1", "payload": {}})
    assert out["timing_valid"] is False
    assert out["clock_quality"] == "syncing"
    assert out["ts_edge_ms"] == ts_edge
    # converted to ~receive time (laptop clock), within a few ms
    assert abs(out["ts_ms"] - out["t_received_ms"]) < 5
    # the clock now carries the coarse offset
    assert abs(hub.clock.offset_ms - skew) < 5
    assert hub.clock.has_estimate


# ---------------------------------------------------------------- 2026-10-02 follow-up
def _edge_event(ts_edge: float, type_: str = "capture_started", cid: str = "c1") -> dict:
    return {"type": "event", "event": {"type": type_, "origin": "edge", "ts_ms": ts_edge,
                                       "command_id": cid, "payload": {}}}


def test_ingress_events_valid_after_three_but_only_approximate(tmp_path):
    hub = _hub(tmp_path)
    hub.clock = ClockOffset()
    outs = [hub._normalize_event(_edge_event(now_ms() - SKEW_MS)["event"]) for _ in range(3)]
    assert [o["timing_valid"] for o in outs] == [False, False, True]
    assert outs[-1]["clock_quality"] == "approximate"


def test_clock_jump_surfaces_in_timeline(tmp_path):
    import asyncio

    hub = _hub(tmp_path)
    hub.clock = ClockOffset()

    async def run():
        for _ in range(5):                                  # Pi 13 h behind
            await hub.on_edge_message(_edge_event(now_ms() - SKEW_MS))
        for _ in range(3):                                  # NTP stepped the Pi clock
            await hub.on_edge_message(_edge_event(now_ms()))

    asyncio.run(run())
    jumps = [e for e in hub.timeline if e["type"] == "edge_clock_jump"]
    assert len(jumps) == 1
    assert abs(jumps[0]["payload"]["delta_ms"] - SKEW_MS) < 1000
    assert abs(hub.clock.offset_ms) < 1000
    # the event after the jump is converted with the new offset
    last_edge = [e for e in hub.timeline if e.get("origin") == "edge"][-1]
    assert abs(last_edge["ts_ms"] - last_edge["t_received_ms"]) < 1000


def test_approximate_clock_keeps_network_out_of_command_log(tmp_path):
    import asyncio

    hub = _hub(tmp_path)
    hub.clock = ClockOffset()
    t_pi = now_ms() - SKEW_MS
    raw = {
        "command_id": "c9", "source": "live",
        "primary": {"model_id": "B2_s0", "class_key": "LIGHT_ON", "intent": "LIGHT_ON",
                    "confidence": 0.9, "feature_ms": 12.0, "inference_ms": 30.0, "top_k": []},
        "accepted": True, "threshold": 0.8, "clip_ms": 500.0,
        "t_capture_end_ms": t_pi - 40.0, "t_result_ms": t_pi,
    }

    async def run():
        for _ in range(3):
            await hub.on_edge_message(_edge_event(now_ms() - SKEW_MS))
        await hub.on_edge_message({"type": "inference_result", "result": raw})

    asyncio.run(run())
    row = hub.log.recent(1)[0]
    assert row["timing_valid"] == 1
    assert row["clock_quality"] == "approximate"
    assert row["network_ms"] is None               # cross-device hop not trusted
    assert row["result_to_action_ms"] is not None and row["result_to_action_ms"] >= 0
    assert abs(row["capture_to_result_ms"] - 40.0) < 1.0
