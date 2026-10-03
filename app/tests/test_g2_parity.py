"""Parity: G2_s0 (DS-CNN, the benchmark model) through the app's engines must match
tiny-vcm's own VCMInferencer (which reads the model_config.json sidecar).

Two checks on 5 real validation WAVs (personal s02 = the Tuning Session):
  * the app's *torch* engine vs VCMInferencer
  * the *registered* engine (ONNX, what the app actually runs) vs VCMInferencer
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")

from vcm_common.ontology import get_ontology  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TVCM = REPO.parent / "tiny-vcm"
CKPT = TVCM / "checkpoints" / "G2_s0" / "best.pt"


def _personal_wavs() -> list[Path]:
    s02 = TVCM / "data" / "personal" / "raw" / "awi01" / "awi01_s02"
    wavs = sorted(s02.glob("*.wav")) if s02.exists() else []
    return wavs[:5]


def _reference(tiny_vcm_root, monkeypatch):
    sys.path.insert(0, str(tiny_vcm_root))
    from src.vcm_infer import VCMInferencer  # noqa: E402

    # VCMInferencer resolves its ontology path relative to CWD; run it from tiny-vcm
    monkeypatch.chdir(tiny_vcm_root)
    return VCMInferencer(CKPT)


def _ref_probs(ref, wav, torch):
    from scipy.io import wavfile as _wavfile  # noqa: E402
    sr, pcm = _wavfile.read(str(wav))
    x = (pcm / 32768.0).astype(np.float32)
    wav_t = torch.from_numpy(x).unsqueeze(0)
    w = ref.processor.to_mono_16k(wav_t, sr)
    w = ref.processor.trim_silence(w)
    feat = ref.processor.process(w)
    from src.vcm_data_loader import fit_frames  # noqa: E402
    feat = fit_frames(feat, ref.n_frames, random_offset=False)
    feat = feat.transpose(1, 2).unsqueeze(0).to(ref.device)
    with torch.no_grad():
        return torch.softmax(ref.model(feat), dim=-1)[0].cpu().numpy()


@pytest.mark.skipif(not CKPT.exists(), reason="G2_s0 weights not synced")
def test_g2_torch_engine_matches_vcm_inferencer(tiny_vcm_root, monkeypatch):
    """The app's torch engine (manifest architecture) vs VCMInferencer: < 1e-5."""
    from edge.engines.torch_tinyvcm import TorchTinyVCMEngine  # noqa: E402
    from vcm_common.manifest import ModelManifest  # noqa: E402

    wavs = _personal_wavs()
    if not wavs:
        pytest.skip("no personal validation WAVs available")

    # same manifest, but pointed at the checkpoint so the torch engine is exercised
    m = ModelManifest.load(REPO / "models" / "vcm" / "G2_s0")
    m.weights = "../../../../tiny-vcm/checkpoints/G2_s0/best.pt"   # relative to the manifest folder
    engine = TorchTinyVCMEngine(m, tiny_vcm_root=str(tiny_vcm_root))
    engine.load()

    ref = _reference(tiny_vcm_root, monkeypatch)

    from scipy.io import wavfile as _wavfile  # noqa: E402
    for wav in wavs:
        sr, pcm = _wavfile.read(str(wav))
        x = (pcm / 32768.0).astype(np.float32)
        got = engine.infer(x, sr).probs
        want = _ref_probs(ref, wav, torch)
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=0)
        assert m.labels[int(got.argmax())] == ref.idx_to_key[int(want.argmax())], wav.name


@pytest.mark.skipif(not CKPT.exists(), reason="G2_s0 weights not synced")
def test_g2_registered_engine_matches_vcm_inferencer(tiny_vcm_root, monkeypatch):
    """The registered (ONNX) engine the app runs vs VCMInferencer."""
    from edge.registry import ModelRegistry  # noqa: E402
    from scipy.io import wavfile as _wavfile  # noqa: E402

    wavs = _personal_wavs()
    if not wavs:
        pytest.skip("no personal validation WAVs available")

    from edge.engines.base import EngineUnavailable  # noqa: E402

    reg = ModelRegistry(REPO / "models", {"tiny_vcm_root": str(tiny_vcm_root)})
    reg.scan()
    try:
        engine = reg.get("G2_s0")          # loads the registered engine (ONNX)
    except EngineUnavailable as exc:
        pytest.skip(f"G2_s0 engine unavailable: {exc}")

    ref = _reference(tiny_vcm_root, monkeypatch)
    worst = 0.0
    for wav in wavs:
        sr, pcm = _wavfile.read(str(wav))
        x = (pcm / 32768.0).astype(np.float32)
        got = engine.infer(x, sr).probs
        want = _ref_probs(ref, wav, torch)
        worst = max(worst, float(np.abs(got - want).max()))
        assert engine.manifest.labels[int(got.argmax())] == ref.idx_to_key[int(want.argmax())], wav.name
    # fp32 ONNX: parity is ~1e-7; int8 would be far looser, so assert only what fp32 gives
    assert worst < 1e-5, f"ONNX vs VCMInferencer max |dprob| = {worst:.2e}"


def test_g2_registered_leaf_labels():
    import yaml
    m = yaml.safe_load((REPO / "models" / "vcm" / "G2_s0" / "manifest.yaml").read_text())
    assert m["label_scheme"] == "leaf"
    assert m["labels"] == get_ontology().leaf_labels()
    assert m["architecture"]["arch"] == "dscnn"
    assert m["architecture"]["channels"] == 112
    assert m["architecture"]["n_blocks"] == 4
    # no keys belonging to another architecture (the E1 manifest had them)
    assert not {"width_mult", "kernel"} & set(m["architecture"])


def test_g2_registered_checkpoint_loads(tiny_vcm_root):
    """test_registered_checkpoints_load[G2_s0] as its own case (the parametrised test
    already covers it; this pins the arch/params contract)."""
    from edge.registry import ModelRegistry  # noqa: E402

    from edge.engines.base import EngineUnavailable  # noqa: E402

    reg = ModelRegistry(REPO / "models", {"tiny_vcm_root": str(tiny_vcm_root)})
    reg.scan()
    entry = reg.entries["G2_s0"]
    w = entry.manifest.weights_path()
    if w is None or not w.exists():
        pytest.skip("G2_s0 weights not synced (exports/G2_s0/model.onnx)")
    try:
        engine = reg.get("G2_s0")
    except EngineUnavailable as exc:
        pytest.fail(f"G2_s0 failed to load: {exc}")
    assert entry.status == "ready", entry.detail
    assert entry.params == 68129
    out = engine.infer(np.zeros(16000, dtype=np.float32), 16000)
    assert out.probs.shape == (33,)
