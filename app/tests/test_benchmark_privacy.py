"""Benchmark privacy: no audio stored, exports carry no classmate name, delete clears."""
import re
import yaml
from fastapi.testclient import TestClient

from laptop.app.benchmark.config import load_benchmark_config
from laptop.app.benchmark.store import BenchmarkStore
from laptop.app.main import REPO_ROOT, create_app
from vcm_common.ontology import get_ontology

MARK_RE = re.compile(r"\bMark\b")


def _cfg(tmp_path):
    c = yaml.safe_load((REPO_ROOT / "config" / "demo.yaml").read_text())
    c["edge_service"]["models_dir"] = str(tmp_path / "models")
    c["laptop"]["data_dir"] = str(tmp_path / "runtime")
    return c


def test_no_audio_written_by_store(tmp_path):
    store = BenchmarkStore(tmp_path / "benchmark.sqlite")
    prompts = [{"kind": "command", "say": "Play music", "expected_class": "PLAY_MUSIC"}]
    sid = store.create_session(alias="Guest 01", mode="live", set_type="fixed", set_seed=None,
                               model_id="B2_s0", threshold=0.8, cond_noise="quiet",
                               cond_distance="near", network_offline=False,
                               consent_results=True, consent_audio=False, note=None, prompts=prompts)
    tid = store.trial_at(sid, 0, 0)
    store.update_trial(tid, outcome="correct", command_id="cid_x", inference_ms=34.0)
    audio = [p for p in tmp_path.rglob("*") if p.suffix in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}]
    assert audio == []
    # only the sqlite store was written
    assert {p.name for p in tmp_path.rglob("*") if p.is_file()} == {"benchmark.sqlite"}


def test_export_has_no_classmate_name(tmp_path):
    with TestClient(create_app(_cfg(tmp_path))) as c:
        c.post("/api/bench/sessions", json={"set_type": "fixed", "consent_results": True})
        csv = c.get("/api/bench/export.csv").text
        js = c.get("/api/bench/export.json").text
        assert "mark_optionb" not in csv and "mark_optionb" not in js
        assert not MARK_RE.search(csv) and not MARK_RE.search(js)


def test_delete_returns_labelled_command_ids(tmp_path):
    store = BenchmarkStore(tmp_path / "benchmark.sqlite")
    o = get_ontology()
    cfg = load_benchmark_config(REPO_ROOT / "config" / "benchmark.yaml", o)
    prompts = [{"kind": p.kind, "say": p.say, "expected_class": p.expected_class}
               for p in cfg.fixed_set]
    sid = store.create_session(alias="Guest 01", mode="live", set_type="fixed", set_seed=None,
                               model_id="B2_s0", threshold=0.8, cond_noise="quiet",
                               cond_distance="near", network_offline=False,
                               consent_results=True, consent_audio=False, note=None, prompts=prompts)
    tid = store.trial_at(sid, 0, 0)
    store.update_trial(tid, outcome="correct", command_id="cid_abc")
    cids = store.delete_session(sid)
    assert cids == ["cid_abc"]
    assert store.get_session(sid) is None
