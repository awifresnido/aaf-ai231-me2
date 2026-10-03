"""Parity: the edge engine must give the same probabilities as tiny-vcm's own
preprocessing + model path, and every registered checkpoint must load."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from vcm_common.ontology import get_ontology  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def test_engine_matches_reference(tmp_path, tiny_vcm_root):
    sys.path.insert(0, str(tiny_vcm_root))
    from src.vcm_data_loader import AudioProcessor, fit_frames
    from src.vcm_models_v2 import build_model

    from edge.registry import ModelRegistry

    labels = get_ontology().leaf_labels()
    torch.manual_seed(0)
    model = build_model({"arch": "tcresnet", "width_mult": 1.0}, len(labels), 40).eval()
    folder = tmp_path / "vcm" / "t"
    folder.mkdir(parents=True)
    torch.save({"model_state_dict": model.state_dict()}, folder / "model.pt")
    m = yaml.safe_load((REPO / "models/vcm/B2_s0/manifest.yaml").read_text())
    m["id"] = "t"
    m["weights"] = "model.pt"
    m["engine"] = "torch_tinyvcm"   # this test exercises the torch parity path
    (folder / "manifest.yaml").write_text(yaml.safe_dump(m))

    reg = ModelRegistry(tmp_path, {"tiny_vcm_root": str(tiny_vcm_root)})
    reg.scan()
    engine = reg.get("t")

    rng = np.random.default_rng(0)
    t = np.arange(24000) / 16000
    clip = (0.3 * np.sin(2 * np.pi * 440 * t) * (t > 0.4) * (t < 1.1)
            + 0.001 * rng.standard_normal(t.size)).astype(np.float32)
    got = engine.infer(clip, 16000).probs

    proc = AudioProcessor()
    w = proc.trim_silence(torch.from_numpy(clip).unsqueeze(0))
    feat = fit_frames(proc.process(w), int(3.0 * 16000 / 160) + 1, random_offset=False)
    with torch.no_grad():
        ref = torch.softmax(model(feat.transpose(1, 2).unsqueeze(0)), -1)[0].numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)
    assert reg.entries["t"].params and reg.entries["t"].status == "ready"


@pytest.mark.parametrize("manifest", sorted((REPO / "models" / "vcm").glob("*/manifest.yaml")),
                         ids=lambda p: p.parent.name)
def test_registered_checkpoints_load(manifest, tiny_vcm_root):
    """Every committed manifest must load its real checkpoint (skips if weights not synced)."""
    from edge.registry import ModelRegistry

    reg = ModelRegistry(REPO / "models", {"tiny_vcm_root": str(tiny_vcm_root)})
    reg.scan()
    mid = manifest.parent.name
    entry = reg.entries[mid]
    if entry.manifest.engine == "agreement":
        # no weights of its own: loading it loads every member (and fails loudly
        # on a labels mismatch), and params are the sum of the members'.
        engine = reg.get(mid)
        assert entry.status == "ready"
        assert entry.params == sum(reg.entries[x].params for x in entry.manifest.members)
        return
    w = entry.manifest.weights_path()
    if w is None or not w.exists():
        pytest.skip(f"weights for {mid} not synced")
    engine = reg.get(mid)  # load_state_dict fails loudly on any arch/class-count mismatch
    out = engine.infer(np.zeros(16000, dtype=np.float32), 16000)
    assert out.probs.shape == (len(engine.manifest.labels),)
    assert abs(float(out.probs.sum()) - 1.0) < 1e-4
