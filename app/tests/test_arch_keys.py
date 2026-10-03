"""architecture: in a registered manifest records exactly one architecture's keys."""
import pytest

from vcm_common.arch_keys import ARCHITECTURE_KEYS, prune_architecture


def test_dscnn_keeps_only_dscnn_keys():
    # the sidecar model_cfg of a DS-CNN run copies B2's config: only the DS-CNN keys
    cfg = {"arch": "dscnn", "width_mult": 1.0, "kernel": 9, "channels": 112,
           "n_blocks": 4, "dropout_rate": 0.1, "gradient_clip": 1.0}
    kept, dropped = prune_architecture(cfg)
    assert kept == {"arch": "dscnn", "channels": 112, "n_blocks": 4,
                    "dropout_rate": 0.1, "gradient_clip": 1.0}
    assert dropped == ["kernel", "width_mult"]


def test_tcresnet_keeps_only_tcresnet_keys():
    cfg = {"arch": "tcresnet", "width_mult": 1.0, "kernel": 9, "channels": 64, "n_blocks": 4,
           "dropout_rate": 0.1}
    kept, dropped = prune_architecture(cfg)
    assert kept["width_mult"] == 1.0 and kept["kernel"] == 9
    assert "channels" not in kept and "n_blocks" not in kept
    assert dropped == ["channels", "n_blocks"]


def test_crnn_attn_keeps_crnn_keys():
    cfg = {"arch": "crnn_attn", "width_mult": 1.0, "gru_hidden": 48, "pool": "attention",
           "dropout_rate": 0.1}
    kept, dropped = prune_architecture(cfg)
    assert kept == {"arch": "crnn_attn", "gru_hidden": 48, "pool": "attention",
                    "dropout_rate": 0.1}
    assert dropped == ["width_mult"]


def test_unknown_arch_is_left_alone():
    cfg = {"arch": "something_new", "width_mult": 1.0, "weird_key": 3}
    kept, dropped = prune_architecture(cfg)
    assert kept == cfg and dropped == []


def test_every_arch_keeps_its_own_arch_key():
    for arch, keys in ARCHITECTURE_KEYS.items():
        assert "arch" in keys, arch
