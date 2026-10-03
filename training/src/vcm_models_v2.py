"""
VCM models, v2 (2026-09-25).

Why this file exists
--------------------
The legacy ``VCMModel`` (src/vcm_model.py) opens with
``SeparableConvBlock(1, 32)``. A depthwise conv on a 1-channel input is a
single 3x3 filter; the following 1x1 pointwise only rescales that one map
32 ways. The whole network therefore sees the spectrogram through ONE
learned filter, then a ~9-frame (~90 ms) receptive field, then global
average pooling. That explains the measured behaviour in
VCM_TRAINING_TROUBLESHOOTING.md (CNN < MLP-on-mean-mel; widening
32-64-128 -> 64-128-256 did nothing, because the 1-filter stem was unchanged).

Two replacements, both standard small-footprint KWS designs:

* ``TCResNet``  -- temporal convolution over time with mel bins as channels
  (Choi et al., 2019, "Temporal Convolution for Real-time Keyword Spotting
  on Mobile Devices"). The first layer sees all mel bins at once; the
  receptive field after the last block is ~171 frames (~1.7 s at 10 ms hop),
  which covers the median command (1.68 s in mark_optionb). Primary choice.
* ``DSCNN``     -- Hello-Edge DS-CNN (Zhang et al., 2017) with the correct
  full-conv stem (1 -> C, kernel 10x4, stride 2x2) followed by
  depthwise-separable blocks. Kept as the 2-D ablation.

Both accept the loader's existing layout ``(batch, 1, time, n_mels)`` so the
data pipeline and trainer are unchanged.

Pooling is mean+max over time instead of plain mean, so a short, decisive
event ("on" vs "off", "up" vs "down") is not diluted by the rest of the
utterance.

Parameter counts (40 mels, 33 leaf classes), computed by hand -- confirm
with ``python -m src.vcm_models_v2``:
    tcresnet width 1.0 : ~67.8 k   (~68 KB int8)
    tcresnet width 0.5 : ~18.4 k
    dscnn 64ch x 4     : ~26.7 k
    crnn_attn (bi-GRU 48, default) : ~69.2 k  (parameter-matched to tcresnet 1.0)
    crnn_attn (uni-GRU 72)         : ~63.9 k

CRNN-Attn (added 2026-09-26)
----------------------------
A non-TC-ResNet challenger for the same job, same data (composite_v1),
same footprint. TC-ResNet pools over time with mean+max, which is
order-blind: "timer for 10 seconds" vs "30 seconds" differ only in *which*
sounds occur, not *where*. CRNN-Attn keeps order:

    2-D conv stem (full conv, not depthwise -- see the legacy-model lesson)
    -> 2 depthwise-separable 2-D blocks (downsample frequency 40 -> 5)
    -> flatten (channels x freq) per frame -> linear projection + LayerNorm
    -> GRU over time (bidirectional by default; we classify a complete
       3-s window after capture, so the backward pass is free)
    -> additive attention pooling with a PADDING MASK
    -> linear head

The padding mask is derived from the input itself: fit_frames pads with
exact zeros after CMVN, so a frame whose 40 mel values are all exactly 0
is padding. This keeps the mask ONNX-exportable (no extra input) and
identical on the Pi. SpecAugment time masks also zero whole frames; those
are masked too, which is harmless (they carry no information).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# TC-ResNet
# --------------------------------------------------------------------------
class _TCResBlock(nn.Module):
    """Two 1-D convs (kernel k) with BN/ReLU and a residual shortcut."""

    def __init__(self, c_in: int, c_out: int, stride: int, kernel: int, dropout: float):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        if stride != 1 or c_in != c_out:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.conv1(x)))
        y = self.drop(y)
        y = self.bn2(self.conv2(y))
        return self.act(y + self.shortcut(x))


class TCResNet(nn.Module):
    """TC-ResNet8-style classifier.

    Parameters
    ----------
    n_classes : int
        Output classes (derive from the label mapping, never a literal).
    n_mels : int
        Mel bins; used as the input channel count.
    width_mult : float
        Channel multiplier on (16, 24, 32, 48).
    kernel : int
        Temporal kernel size of the residual blocks.
    dropout : float
        Dropout inside blocks and before the head.
    """

    BASE_CHANNELS = (16, 24, 32, 48)

    def __init__(
        self,
        n_classes: int,
        n_mels: int = 40,
        width_mult: float = 1.0,
        kernel: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()
        ch = [max(8, int(round(c * width_mult))) for c in self.BASE_CHANNELS]
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, ch[0], 3, padding=1, bias=False),
            nn.BatchNorm1d(ch[0]),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            _TCResBlock(ch[0], ch[1], stride=2, kernel=kernel, dropout=dropout),
            _TCResBlock(ch[1], ch[2], stride=2, kernel=kernel, dropout=dropout),
            _TCResBlock(ch[2], ch[3], stride=2, kernel=kernel, dropout=dropout),
        )
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * ch[3], n_classes)  # mean + max pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, time, n_mels) -> logits (batch, n_classes)."""
        x = x.squeeze(1).transpose(1, 2)  # (batch, n_mels, time)
        x = self.blocks(self.stem(x))     # (batch, C, time/8)
        x = x.float()                     # pool in fp32 under AMP
        pooled = torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)
        return self.head(self.head_drop(pooled))


# --------------------------------------------------------------------------
# DS-CNN (Hello Edge) with a correct stem
# --------------------------------------------------------------------------
class _DSBlock(nn.Module):
    def __init__(self, c: int, dropout: float):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return self.drop(x)


class DSCNN(nn.Module):
    """DS-CNN-S style: full-conv stem (the fix) + depthwise-separable blocks."""

    def __init__(
        self,
        n_classes: int,
        n_mels: int = 40,
        channels: int = 64,
        n_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(10, 4), stride=(2, 2), padding=(5, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_DSBlock(channels, dropout) for _ in range(n_blocks)])
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, time, n_mels) -> logits (batch, n_classes)."""
        x = self.blocks(self.stem(x))      # (batch, C, time/2, n_mels/2)
        x = x.float().mean(dim=3)          # collapse frequency -> (batch, C, time')
        pooled = torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)
        return self.head(self.head_drop(pooled))


# --------------------------------------------------------------------------
# CRNN with attention pooling
# --------------------------------------------------------------------------
def _conv_out(n: int, kernel: int, stride: int, pad: int) -> int:
    return (n + 2 * pad - kernel) // stride + 1


class _DSConv2d(nn.Module):
    """Depthwise 3x3 (with stride) + pointwise 1x1, BN/ReLU after each."""

    def __init__(self, c_in: int, c_out: int, stride: tuple[int, int], dropout: float):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, 3, stride=stride, padding=1, groups=c_in, bias=False)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return self.drop(x)


class CRNNAttn(nn.Module):
    """Conv front-end + GRU + masked attention pooling.

    Parameters
    ----------
    n_classes : int
        Output classes (from the label mapping).
    n_mels : int
        Mel bins of the input.
    channels : (int, int, int)
        Stem, block-1, block-2 output channels.
    proj_dim : int
        Per-frame projection size fed to the GRU.
    gru_hidden : int
        GRU hidden size per direction.
    bidirectional : bool
        Bi-GRU (default). Uni-GRU is the streaming-friendly ablation.
    attn_dim : int
        Hidden size of the additive attention scorer.
    pool : str
        "attention" (default) or "meanmax" (masked mean+max; isolates the
        effect of attention vs the GRU itself).
    dropout : float
    """

    def __init__(
        self,
        n_classes: int,
        n_mels: int = 40,
        channels: tuple[int, int, int] = (32, 48, 64),
        proj_dim: int = 64,
        gru_hidden: int = 48,
        bidirectional: bool = True,
        attn_dim: int = 64,
        pool: str = "attention",
        dropout: float = 0.1,
    ):
        super().__init__()
        if pool not in ("attention", "meanmax"):
            raise ValueError(f"pool must be 'attention' or 'meanmax', got {pool!r}")
        c0, c1, c2 = channels
        self.pool = pool

        # time stride 2 happens ONLY in the stem (kernel 5, pad 2); the mask
        # downsampling in forward() uses the same (5, 2, 2) geometry.
        self.stem = nn.Sequential(
            nn.Conv2d(1, c0, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1), bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )
        self.block1 = _DSConv2d(c0, c1, stride=(1, 2), dropout=dropout)
        self.block2 = _DSConv2d(c1, c2, stride=(1, 2), dropout=dropout)

        f = _conv_out(n_mels, 3, 2, 1)   # stem
        f = _conv_out(f, 3, 2, 1)        # block1
        f = _conv_out(f, 3, 2, 1)        # block2
        self.proj = nn.Sequential(
            nn.Linear(c2 * f, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(proj_dim, gru_hidden, batch_first=True, bidirectional=bidirectional)
        d = gru_hidden * (2 if bidirectional else 1)
        self.attn = nn.Sequential(nn.Linear(d, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1))
        head_in = d if pool == "attention" else 2 * d
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(head_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, time, n_mels) -> logits (batch, n_classes)."""
        # padding mask from the input: fit_frames pads with exact zeros post-CMVN
        valid_in = (x.abs().sum(dim=(1, 3)) > 0).float().unsqueeze(1)   # (B, 1, T)

        h = self.block2(self.block1(self.stem(x)))                         # (B, C, T', F')
        b, c, t, f = h.shape
        h = h.permute(0, 2, 1, 3).reshape(b, t, c * f)                     # (B, T', C*F')
        h = self.proj(h)
        h, _ = self.gru(h)                                                 # (B, T', d)
        h = h.float()                                                      # pool in fp32 under AMP

        valid = F.max_pool1d(valid_in, kernel_size=5, stride=2, padding=2)  # same geometry as stem
        valid = valid.squeeze(1)[:, :t] > 0                                 # (B, T')
        # a fully-masked row (never expected) falls back to attending everywhere
        valid = valid | ~valid.any(dim=1, keepdim=True)

        if self.pool == "attention":
            scores = self.attn(h).squeeze(-1).float()                      # (B, T')
            scores = scores.masked_fill(~valid, -1e9)
            w = torch.softmax(scores, dim=1)
            pooled = (w.unsqueeze(-1) * h).sum(dim=1)                      # (B, d)
        else:
            m = valid.unsqueeze(-1).float()
            mean = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
            mx = h.masked_fill(~valid.unsqueeze(-1), -1e9).amax(dim=1)
            pooled = torch.cat([mean, mx], dim=1)
        return self.head(self.head_drop(pooled))


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def build_model(model_cfg: dict[str, Any], n_classes: int, n_mels: int) -> nn.Module:
    """Build a model from the ``model:`` block of train_config.yaml.

    ``arch`` is one of ``tcresnet`` (default), ``dscnn``, ``crnn_attn``,
    ``legacy``. ``legacy`` is the original VCMModel, kept only as the
    baseline row of the ablation table.
    """
    arch = model_cfg.get("arch", "tcresnet")
    dropout = float(model_cfg.get("dropout_rate", 0.1))
    if arch == "tcresnet":
        return TCResNet(
            n_classes=n_classes,
            n_mels=n_mels,
            width_mult=float(model_cfg.get("width_mult", 1.0)),
            kernel=int(model_cfg.get("kernel", 9)),
            dropout=dropout,
        )
    if arch == "dscnn":
        return DSCNN(
            n_classes=n_classes,
            n_mels=n_mels,
            channels=int(model_cfg.get("channels", 64)),
            n_blocks=int(model_cfg.get("n_blocks", 4)),
            dropout=dropout,
        )
    if arch == "crnn_attn":
        return CRNNAttn(
            n_classes=n_classes,
            n_mels=n_mels,
            channels=tuple(model_cfg.get("crnn_channels", (32, 48, 64))),
            proj_dim=int(model_cfg.get("proj_dim", 64)),
            gru_hidden=int(model_cfg.get("gru_hidden", 48)),
            bidirectional=bool(model_cfg.get("bidirectional", True)),
            attn_dim=int(model_cfg.get("attn_dim", 64)),
            pool=str(model_cfg.get("pool", "attention")),
            dropout=dropout,
        )
    if arch == "legacy":
        from .vcm_model import VCMModel
        return VCMModel(n_intents=n_classes, n_mel_bins=n_mels, dropout_rate=dropout)
    raise ValueError(f"Unknown model.arch {arch!r} (expected tcresnet | dscnn | crnn_attn | legacy)")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Shape + parameter smoke test: python -m src.vcm_models_v2
    n_mels, frames, n_cls = 40, 301, 33   # 3.0 s window
    dummy = torch.randn(4, 1, frames, n_mels)
    dummy[:, :, 250:, :] = 0.0            # simulate right padding for the CRNN mask
    for cfg in (
        {"arch": "tcresnet", "width_mult": 1.0},
        {"arch": "tcresnet", "width_mult": 0.5},
        {"arch": "dscnn"},
        {"arch": "crnn_attn"},
        {"arch": "crnn_attn", "bidirectional": False, "gru_hidden": 72},
        {"arch": "crnn_attn", "pool": "meanmax"},
        {"arch": "legacy"},
    ):
        m = build_model(cfg, n_cls, n_mels).eval()
        out = m(dummy)
        assert out.shape == (4, n_cls), out.shape
        assert torch.isfinite(out).all(), cfg
        print(f"{cfg}: params={count_parameters(m):,} out={tuple(out.shape)}")

    # mask check: EXTRA right padding should barely change a CRNN prediction
    m = build_model({"arch": "crnn_attn"}, n_cls, n_mels).eval()
    with torch.no_grad():
        a = m(dummy)
        longer = torch.cat([dummy, torch.zeros(4, 1, 40, n_mels)], dim=2)
        b = m(longer)
    print(f"crnn_attn extra-padding max |dlogit| = {(a - b).abs().max().item():.2e} "
          f"(small but nonzero is expected: the bi-GRU's backward pass reads through padding)")
