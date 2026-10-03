"""Parity: E1_s0 (CRNN-Attn) through the app's torch engine must match
tiny-vcm's own VCMInferencer (which reads the model_config.json sidecar)."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")

from vcm_common.ontology import get_ontology  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TVCM = REPO.parent / "tiny-vcm"


def _personal_wavs() -> list[Path]:
    s02 = TVCM / "data" / "personal" / "raw" / "awi01" / "awi01_s02"
    wavs = sorted(s02.glob("*.wav")) if s02.exists() else []
    return wavs[:5]


@pytest.mark.skipif(not (TVCM / "checkpoints" / "E1_s0" / "best.pt").exists(),
                    reason="E1_s0 weights not synced")
def test_e1_engine_matches_vcm_inferencer(tiny_vcm_root, monkeypatch):
    wavs = _personal_wavs()
    if not wavs:
        pytest.skip("no personal validation WAVs available")

    sys.path.insert(0, str(tiny_vcm_root))
    from src.vcm_infer import VCMInferencer  # noqa: E402

    from edge.registry import ModelRegistry  # noqa: E402

    reg = ModelRegistry(REPO / "models", {"tiny_vcm_root": str(tiny_vcm_root)})
    reg.scan()
    engine = reg.get("E1_s0")

    # VCMInferencer resolves its ontology path relative to CWD; run it from tiny-vcm
    monkeypatch.chdir(tiny_vcm_root)
    ref = VCMInferencer(TVCM / "checkpoints" / "E1_s0" / "best.pt")

    from scipy.io import wavfile as _wavfile  # noqa: E402

    labels = engine.manifest.labels
    for wav in wavs:
        sr, pcm = _wavfile.read(str(wav))
        x = (pcm / 32768.0).astype(np.float32)          # 1D: the engine unsqueezes internally
        got = engine.infer(x, sr).probs

        # reference: full softmax through VCMInferencer's own pipeline
        wav_t = torch.from_numpy(x).unsqueeze(0)         # (1, samples) for the processor
        w = ref.processor.to_mono_16k(wav_t, sr)
        w = ref.processor.trim_silence(w)
        feat = ref.processor.process(w)
        from src.vcm_data_loader import fit_frames  # noqa: E402
        feat = fit_frames(feat, ref.n_frames, random_offset=False)
        feat = feat.transpose(1, 2).unsqueeze(0).to(ref.device)
        with torch.no_grad():
            ref_probs = torch.softmax(ref.model(feat), dim=-1)[0].cpu().numpy()

        np.testing.assert_allclose(got, ref_probs, atol=1e-5, rtol=0)
        assert labels[int(got.argmax())] == ref.idx_to_key[int(ref_probs.argmax())], \
            f"argmax mismatch on {wav.name}"


def test_e1_registered_leaf_labels():
    from vcm_common.ontology import get_ontology
    import yaml
    m = yaml.safe_load((REPO / "models" / "vcm" / "E1_s0" / "manifest.yaml").read_text())
    assert m["label_scheme"] == "leaf"
    assert m["labels"] == get_ontology().leaf_labels()
