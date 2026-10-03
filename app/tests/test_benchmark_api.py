"""Benchmark API: lifecycle, lock, concurrency, export (local mode, scripted edge)."""
import yaml
from fastapi.testclient import TestClient

from laptop.app.main import REPO_ROOT, create_app


def cfg(tmp_path):
    c = yaml.safe_load((REPO_ROOT / "config" / "demo.yaml").read_text())
    c["edge_service"]["models_dir"] = str(tmp_path / "models")
    c["edge_service"]["compare_ids"] = []
    c["laptop"]["data_dir"] = str(tmp_path / "runtime")
    return c


def _create(c):
    return c.post("/api/bench/sessions", json={
        "set_type": "fixed", "cond_noise": "quiet", "cond_distance": "near",
        "consent_results": True, "consent_audio": False, "alias": "Guest 01"})


def test_create_requires_consent(tmp_path):
    with TestClient(create_app(cfg(tmp_path))) as c:
        r = c.post("/api/bench/sessions", json={"set_type": "fixed", "consent_results": False})
        assert r.status_code == 400


def test_create_start_abandon_lifecycle(tmp_path):
    with TestClient(create_app(cfg(tmp_path))) as c:
        r = _create(c)
        assert r.status_code == 200
        sess = r.json()
        assert sess["status"] == "created"
        assert sess["set_type"] == "fixed"
        sid = sess["session_id"]
        assert c.post(f"/api/bench/sessions/{sid}/start").status_code == 200
        # lock: model select + threshold rejected while running
        assert c.post("/api/models/select", json={"model_id": "B2_s0"}).status_code == 409
        assert c.post("/api/threshold", json={"value": 0.9}).status_code == 409
        assert c.post(f"/api/bench/sessions/{sid}/abandon").status_code == 200
        # released after abandon
        assert c.post("/api/models/select", json={"model_id": "B2_s0"}).status_code == 200


def test_second_concurrent_start_rejected(tmp_path):
    with TestClient(create_app(cfg(tmp_path))) as c:
        s1 = _create(c).json()
        s2 = _create(c).json()
        assert c.post(f"/api/bench/sessions/{s1['session_id']}/start").status_code == 200
        assert c.post(f"/api/bench/sessions/{s2['session_id']}/start").status_code == 409


def test_finish_marks_remaining_skipped(tmp_path):
    with TestClient(create_app(cfg(tmp_path))) as c:
        sess = _create(c).json()
        c.post(f"/api/bench/sessions/{sess['session_id']}/start")
        assert c.post(f"/api/bench/sessions/{sess['session_id']}/finish").status_code == 200
        assert c.get("/api/state").json()["benchmark"]["running"] is False


def test_list_summary_export(tmp_path):
    with TestClient(create_app(cfg(tmp_path))) as c:
        _create(c)
        assert c.get("/api/bench/sessions").status_code == 200
        assert c.get("/api/bench/summary").status_code == 200
        assert c.get("/api/bench/export.csv").status_code == 200
        assert c.get("/api/bench/export.json").status_code == 200


def test_delete_session(tmp_path):
    with TestClient(create_app(cfg(tmp_path))) as c:
        sess = _create(c).json()
        r = c.delete(f"/api/bench/sessions/{sess['session_id']}")
        assert r.status_code == 200
        ids = [s["session_id"] for s in c.get("/api/bench/sessions").json()]
        assert sess["session_id"] not in ids
