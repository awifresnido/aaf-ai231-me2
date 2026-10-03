"""Per-architecture ``architecture:`` keys a model manifest may carry.

tiny-vcm's ``build_model`` reads only the keys its own arch needs, but the ``model:``
block of a train_config.yaml holds every arch's keys (``width_mult``/``kernel`` belong
to TC-ResNet, ``channels``/``n_blocks`` to DS-CNN, ``crnn_*`` to CRNN-Attn). Copying that
whole block into a manifest records configuration belonging to another architecture —
that is how the E1 (CRNN-Attn) manifest ended up carrying ``width_mult`` and ``kernel``.

``prune_architecture`` uses this table so a registered manifest describes exactly one
architecture. Unknown arch: return the dict unchanged (fail open, never silently drop
keys we do not understand).
"""
from __future__ import annotations

ARCHITECTURE_KEYS: dict[str, frozenset[str]] = {
    "tcresnet": frozenset({"arch", "width_mult", "kernel", "dropout_rate", "gradient_clip"}),
    "dscnn": frozenset({"arch", "channels", "n_blocks", "dropout_rate", "gradient_clip"}),
    "crnn_attn": frozenset({"arch", "crnn_channels", "proj_dim", "gru_hidden", "bidirectional",
                            "attn_dim", "pool", "dropout_rate", "gradient_clip"}),
    "legacy": frozenset({"arch", "dropout_rate", "gradient_clip"}),
}


def prune_architecture(model_cfg: dict) -> tuple[dict, list[str]]:
    """Return ``(kept, dropped)`` for the arch named in ``model_cfg``."""
    arch = str(model_cfg.get("arch", ""))
    allowed = ARCHITECTURE_KEYS.get(arch)
    if allowed is None:
        return dict(model_cfg), []
    kept = {k: v for k, v in model_cfg.items() if k in allowed}
    dropped = sorted(k for k in model_cfg if k not in allowed)
    return kept, dropped
