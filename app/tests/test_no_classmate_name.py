"""The synthetic dataset's contributor (``mark_optionb``) must never reach a user.

Covers every surface the UI reads: the built frontend bundle, the About data
snapshot (§3), and the API responses the UI consumes (``/api/state``,
``/api/about``). ``mark_optionb`` is matched case-sensitively; the prose name is
matched with a word-boundary regex so ``benchmark`` and ``brand-mark`` are fine.
"""
import json
import re
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from laptop.app.main import REPO_ROOT, create_app

MARK_ID = "mark_optionb"
MARK_RE = re.compile(r"\bMark\b")
# File extensions of the built bundle whose bytes may contain UI strings.
TEXT_SUFFIXES = {".js", ".html", ".css", ".json", ".txt", ".svg", ".map"}


def _cfg(tmp_path: Path) -> dict:
    c = yaml.safe_load((REPO_ROOT / "config" / "demo.yaml").read_text())
    # Real manifests (A2/A7 carry mark_optionb in lineage/scores) but a scripted
    # active model so the test needs no torch / ONNX weights.
    c["edge_service"]["models_dir"] = str(REPO_ROOT / "models")
    c["edge_service"]["active_vcm"] = "simulator"
    c["edge_service"]["compare_ids"] = []
    c["laptop"]["data_dir"] = str(tmp_path / "runtime")
    return c


def _problems(text: str) -> list[str]:
    out = []
    if MARK_ID in text:
        out.append(f"contains '{MARK_ID}'")
    if MARK_RE.search(text):
        out.append("matches \\bMark\\b")
    return out


def test_api_state_has_no_classmate_name(tmp_path):
    with TestClient(create_app(_cfg(tmp_path))) as c:
        state = c.get("/api/state").json()
        assert state["edge"]["models"], "expected the real model registry"
        assert _problems(json.dumps(state)) == []


def test_api_about_has_no_classmate_name(tmp_path):
    with TestClient(create_app(_cfg(tmp_path))) as c:
        r = c.get("/api/about")
        if r.status_code == 404:
            pytest.skip("GET /api/about not implemented yet")
        assert r.status_code == 200
        assert _problems(r.text) == []


def test_about_data_json_has_no_classmate_name():
    p = REPO_ROOT / "laptop" / "app" / "about" / "about_data.json"
    if not p.exists():
        pytest.skip("about_data.json not generated yet")
    assert _problems(p.read_text(encoding="utf-8")) == []


def test_built_bundle_has_no_classmate_name():
    dist = REPO_ROOT / "laptop" / "frontend" / "dist"
    if not dist.exists():
        pytest.skip("frontend not built (run `npm run build`)")
    found = [f for f in dist.rglob("*") if f.is_file() and f.suffix in TEXT_SUFFIXES]
    assert found, "built bundle is empty"
    for f in found:
        problems = _problems(f.read_text(encoding="utf-8", errors="ignore"))
        assert not problems, f"{f.relative_to(REPO_ROOT)}: {problems}"
