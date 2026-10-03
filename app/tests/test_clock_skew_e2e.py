"""End-to-end: a skewed result through the hub gives realistic latencies."""
import asyncio

import yaml

from laptop.app.hub import Hub
from laptop.app.main import REPO_ROOT
from vcm_common.protocol import now_ms

SKEW = 13 * 3600 * 1000.0


def _hub(tmp_path):
    c = yaml.safe_load((REPO_ROOT / "config" / "demo.yaml").read_text())
    c["edge_service"]["models_dir"] = str(tmp_path / "models")
    c["edge_service"]["compare_ids"] = []
    c["laptop"]["data_dir"] = str(tmp_path / "runtime")
    return Hub(c, REPO_ROOT)


def test_skewed_result_through_hub_gives_realistic_latency(tmp_path):
    hub = _hub(tmp_path)
    # simulate a measured remote offset of -13 h (the Pi behind the laptop)
    hub.clock.offset_ms = -SKEW
    hub.clock.ready = True

    t_laptop = now_ms()
    t_result_pi = t_laptop - SKEW
    t_capture_pi = t_result_pi - 40.0
    raw = {
        "command_id": "c1", "source": "live",
        "primary": {"model_id": "B2_s0", "class_key": "LIGHT_ON", "intent": "LIGHT_ON",
                    "confidence": 0.9, "feature_ms": 12.0, "inference_ms": 30.0, "top_k": []},
        "accepted": True, "threshold": 0.8, "clip_ms": 500.0,
        "t_capture_end_ms": t_capture_pi, "t_result_ms": t_result_pi,
    }
    asyncio.run(hub.on_edge_message({"type": "inference_result", "result": raw}))

    row = hub.log.recent(1)[0]
    assert row["class_key"] == "LIGHT_ON"
    assert row["timing_valid"] == 1
    # result_to_action_ms is a laptop-only interval: small, positive
    assert row["result_to_action_ms"] is not None
    assert 0 <= row["result_to_action_ms"] < 1000
    # capture_to_result_ms is a Pi-only interval: ~40 ms
    assert abs(row["capture_to_result_ms"] - 40.0) < 1.0
    # network_ms is the (estimated) cross-machine hop
    assert row["network_ms"] is not None
