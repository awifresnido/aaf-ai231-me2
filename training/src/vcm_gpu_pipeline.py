"""
GPU feature pipeline (ME2 Part 0 step 6).

The DataLoader workers only return raw (cached) waveforms; everything else —
waveform augmentation, silence trim, log-mel + CMVN, frame fit, SpecAugment —
runs on the device inside the collate function, so the CPU workers are no
longer the bottleneck and all runs share one path.

The CPU path (WaveAugment + AudioProcessor.process in the worker) is kept
intact and is the reference for GATE 0c parity:

  (i)  no-augmentation GPU features vs CPU features on 500 clips: max abs diff <= 1e-4;
  (ii) augmentation parameter histograms (SNR, gain, RT60, speed factor)
       within 5% of the CPU path.

GPUWaveAugment draws the SAME parameter distributions as WaveAugment but with
torch.rand on the device (GATE 0c(ii) checks histograms, not per-clip identity,
so the RNG source is irrelevant; the distributions are identical).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader, TensorDataset

from .vcm_data_loader import SpecAugment, fit_frames


class GPUWaveAugment:
    """On-device waveform augmentation matching WaveAugment's statistics."""

    def __init__(self, cfg, noise_bank, sample_rate, device):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg)
        self.sample_rate = sample_rate
        self.device = device
        self.noise_bank = [n.to(device) for n in (noise_bank or [])]
        self.c_noise = dict(cfg.get("noise_mix", {}))
        self.c_reverb = dict(cfg.get("reverb", {}))
        self.c_speed = dict(cfg.get("speed_perturb", {}))
        self.c_gain = dict(cfg.get("gain", {}))
        self.c_band = dict(cfg.get("mic_bandlimit", {}))
        self.aug_stats = {"snr": [], "gain_db": [], "rt60": [], "speed_factor": []}

    @staticmethod
    def _p(d, default):
        return float(d.get("p", default))

    def __call__(self, waves: List[torch.Tensor]) -> List[torch.Tensor]:
        """waves: list of (1, L_i) on device -> augmented list of (1, L_i')."""
        if not self.enabled:
            return waves
        dev = waves[0].device
        B = len(waves)

        # noise mix
        p = self._p(self.c_noise, 0.6)
        if self.noise_bank and p > 0.0:
            for i in range(B):
                if torch.rand(1, device=dev) >= p:
                    continue
                noise = self.noise_bank[int(torch.randint(len(self.noise_bank), (1,), device=dev))]
                L = waves[i].shape[-1]
                n_len = noise.shape[-1]
                if n_len >= L:
                    off = int(torch.randint(n_len - L + 1, (1,), device=dev))
                    n = noise[..., off:off + L]
                else:
                    n = noise.repeat(1, (L + n_len - 1) // n_len)[..., :L]
                snr = float(torch.empty(1, device=dev).uniform_(*self.c_noise.get("snr_db", [5, 25])))
                self.aug_stats["snr"].append(snr)
                p_sig = (waves[i] ** 2).mean() + 1e-10
                p_noise = (n ** 2).mean() + 1e-10
                waves[i] = waves[i] + ((p_sig / (p_noise * (10 ** (snr / 10)))) ** 0.5) * n

        # reverb (FFT convolution)
        p = self._p(self.c_reverb, 0.3)
        if p > 0.0:
            for i in range(B):
                if torch.rand(1, device=dev) >= p:
                    continue
                rt60 = float(torch.empty(1, device=dev).uniform_(*self.c_reverb.get("rt60_s", [0.15, 0.6])))
                self.aug_stats["rt60"].append(rt60)
                n = max(8, int(rt60 * self.sample_rate))
                t = torch.arange(n, device=dev).float() / self.sample_rate
                rir = torch.randn(n, device=dev) * torch.exp(-6.91 * t / rt60)
                rir[0] = 1.0
                rir = rir / (rir.abs().max() + 1e-8)
                L = waves[i].shape[-1]
                waves[i] = torchaudio.functional.fftconvolve(waves[i], rir.unsqueeze(0))[..., :L]

        # speed perturbation (group by factor -> one batched resample per factor)
        p = self._p(self.c_speed, 0.5)
        if p > 0.0:
            factors = [float(f) for f in self.c_speed.get("factors", [0.9, 1.0, 1.1])]
            chosen: List[float] = []
            for _ in range(B):
                if torch.rand(1, device=dev) < p:
                    f = factors[int(torch.randint(len(factors), (1,), device=dev))]
                    self.aug_stats["speed_factor"].append(f)
                    chosen.append(f)
                else:
                    chosen.append(1.0)
            groups: dict = {}
            order: List[Tuple[int, float]] = []
            for i, f in enumerate(chosen):
                if abs(f - 1.0) >= 1e-6:
                    groups.setdefault(f, []).append(i)
                    order.append((i, f))
            for f, idx in groups.items():
                max_len = max(waves[i].shape[-1] for i in idx)
                grp = torch.zeros(len(idx), 1, max_len, device=dev)
                for j, i in enumerate(idx):
                    grp[j, 0, :waves[i].shape[-1]] = waves[i][0]
                new_len = max(1, int(round(max_len * f)))
                res = torchaudio.functional.resample(grp, self.sample_rate, round(self.sample_rate * f))
                for j, i in enumerate(idx):
                    # trim back to the clip's own resampled length (drop shared padding)
                    own = max(1, int(round(waves[i].shape[-1] * f)))
                    waves[i] = res[j, 0, :own].unsqueeze(0)

        # gain
        p = self._p(self.c_gain, 0.5)
        if p > 0.0:
            db = float(self.c_gain.get("db", 6.0))
            for i in range(B):
                if torch.rand(1, device=dev) >= p:
                    continue
                g_db = float(torch.empty(1, device=dev).uniform_(-db, db))
                self.aug_stats["gain_db"].append(g_db)
                waves[i] = torch.clamp(waves[i] * (10 ** (g_db / 20.0)), -1.0, 1.0)

        # low-pass (mic bandlimit)
        p = self._p(self.c_band, 0.2)
        if p > 0.0:
            lo, hi = self.c_band.get("cutoff_hz", [3500, 7500])
            for i in range(B):
                if torch.rand(1, device=dev) >= p:
                    continue
                cutoff = float(torch.empty(1, device=dev).uniform_(lo, hi))
                waves[i] = torchaudio.functional.lowpass_biquad(waves[i], self.sample_rate, cutoff)

        return waves


class GPUBatchProcessor:
    """Waveform list -> (B, 1, time, n_mels) features, all on the device."""

    def __init__(self, audio_processor, wave_cfg, spec_cfg, n_frames, noise_bank, device):
        import torchaudio.transforms as T
        self.ap = audio_processor
        self.n_frames = n_frames
        self.device = device
        # Build a FRESH on-device mel front-end (deepcopy would share torchaudio's
        # lazily-initialised STFT window buffer and move the CPU path's window to CUDA).
        ms = audio_processor.mel_spectrogram
        self.mel = T.MelSpectrogram(
            sample_rate=audio_processor.sample_rate,
            n_mels=audio_processor.n_mels,
            n_fft=ms.n_fft,
            hop_length=audio_processor.hop_length,
            f_min=ms.f_min,
            f_max=ms.f_max,
        ).to(device).double()
        self.amp_to_db = T.AmplitudeToDB(stype="power", top_db=audio_processor.amp_to_db.top_db)
        self.wave = GPUWaveAugment(wave_cfg, noise_bank, audio_processor.sample_rate, device) \
            if wave_cfg else None
        sa = dict(spec_cfg or {})
        use_sa = bool(sa.pop("spec_augment", True))
        self.spec_augment = SpecAugment(**sa) if use_sa else None

    def __call__(self, waves: List[torch.Tensor], augment: bool) -> torch.Tensor:
        if augment and self.wave is not None:
            waves = self.wave(waves)
        waves = [self.ap.trim_silence(w) for w in waves]
        max_len = max(w.shape[-1] for w in waves) if waves else 0
        padded = torch.zeros(len(waves), 1, max_len, device=self.device)
        for i, w in enumerate(waves):
            padded[i, 0, :w.shape[-1]] = w[0]
        mel_db = self.amp_to_db(self.mel(padded.double()).float())  # (B, n_mels, T)
        mean = mel_db.mean(dim=-1, keepdim=True)
        std = mel_db.std(dim=-1, keepdim=True, unbiased=False)
        feat = (mel_db - mean) / (std + 1e-5)
        feats = [fit_frames(feat[i:i + 1], self.n_frames, random_offset=augment) for i in range(len(waves))]
        feat = torch.cat(feats, dim=0)
        if augment and self.spec_augment is not None:
            feat = self.spec_augment(feat)
        return feat.transpose(1, 2)  # (B, 1, time, n_mels)


def make_gpu_collate(processor: GPUBatchProcessor, augment: bool, device):
    """DataLoader collate: raw (wave, label) -> (features, label) on device."""
    def collate(batch):
        waves = [w.to(device) for w, _ in batch]
        labels = torch.tensor([y for _, y in batch])
        feats = processor(waves, augment=augment)
        return feats, labels.to(device)
    return collate


def cache_val_features(val_dataset, processor: GPUBatchProcessor, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute validation features once (no augmentation) on the device."""
    feats, labels = [], []
    for w, y in val_dataset:
        f = processor([w.to(device)], augment=False).cpu()
        feats.append(f)
        labels.append(y)
    return torch.cat(feats, dim=0), torch.tensor(labels)


def make_val_tensor_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int,
                           num_workers: int, seed=None) -> DataLoader:
    ds = TensorDataset(features, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
