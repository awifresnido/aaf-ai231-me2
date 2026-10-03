"""End-to-end through the HTTP API in local mode with the simulator."""
import time

import yaml
from fastapi.testclient import TestClient

from laptop.app.main import REPO_ROOT, create_app


def cfg(tmp_path):
    c = yaml.safe_load((REPO_ROOT / "config" / "demo.yaml").read_text())
    c["edge_service"]["result_hold_s"] = 0.05
    c["edge_service"]["models_dir"] = str(tmp_path / "models")  # no torch needed
    c["edge_service"]["compare_ids"] = []
    c["laptop"]["data_dir"] = str(tmp_path / "runtime")
    return c


def wait_for(client, pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get("/api/state").json()
        if pred(s):
            return s
        time.sleep(0.05)
    raise AssertionError("condition not reached")


def test_simulated_light_on(tmp_path):
    with TestClient(create_app(cfg(tmp_path))) as c:
        assert c.post("/api/simulate", json={"class_key": "LIGHT_ON"}).status_code == 200
        s = wait_for(c, lambda s: s["devices"]["light"]["power"])
        assert s["last_result"]["source"] == "simulated"
        assert s["last_result"]["outcome"]["executed"]
        s = wait_for(c, lambda s: s["phase"].get("phase") == "idle")
        assert s["commands"][0]["class_key"] == "LIGHT_ON"
        assert c.post("/api/simulate", json={"class_key": "NOPE"}).status_code == 400
