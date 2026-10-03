"""
VCM Data Loader: Audio preprocessing and dataset management for training pipeline
Handles mel-spectrogram generation, data augmentation, and manifest-based dataset loading

Reconciled 2026-09-24 (see ../MODEL_DATASET_LABEL_DISCREPANCY.md): labels and
manifest format come from the canonical ontology (configs/ontology.json) and
the CSV manifests under data/manifests/ (keyed on `training_label`,
`slot_value`, `relative_path`).

Revised 2026-09-25 (see ../VCM_TRAINING_TROUBLESHOOTING.md, Step 7):
* Padding moved from the waveform domain to the FEATURE domain. Before, clips
  were zero-padded to 4.0 s (~58 % zeros for a 1.68 s median), log-mel of
  digital silence is ~-100 dB, and per-sample mean/std was computed over that
  padding -- so the dominant feature was utterance LENGTH, not content.
  Now: trim leading/trailing silence -> log-mel (top_db clamp) -> per-mel
  CMVN over the real frames only -> pad with 0 (= the normalized mean).
* max_duration, n_mels, n_fft, hop, SpecAugment params are now actually read
  from train_config.yaml (they were silently ignored before).
* label_mode="leaf" gives one class per (intent, slot value), e.g.
  "BRIGHTNESS|100 percent", so the demo app gets the slot value it needs.
  label_mode="intent" keeps the 21-class ontology contract.
* Resamplers are cached per source rate (Chatterbox TTS is 24 kHz; building
  a new Resample kernel per clip was a data-loader CPU hot spot).
"""

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

from .vcm_data.common import load_ontology, load_slot_values

LEAF_SEP = "|"

SLOTTED_INTENTS = {"TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER"}


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
def create_intent_mapping(ontology_path: str = "configs/ontology.json") -> Dict[str, int]:
    """Intent-level mapping (21 classes: 13 fixed + 6 slotted + UNKNOWN + SILENCE)."""
    valid_labels, _ = load_ontology(Path(ontology_path))
    return {label: idx for idx, label in enumerate(sorted(valid_labels))}


def create_label_mapping(
    ontology_path: str = "configs/ontology.json", label_mode: str = "intent"
) -> Dict[str, int]:
    """Class mapping for the chosen label_mode.

    intent : 21 classes, identical to create_intent_mapping().
    leaf   : every non-slotted intent + one class per closed slot value
             (13 + 18 + 2 = 33 with the Option B ontology).
    """
    if label_mode == "intent":
        return create_intent_mapping(ontology_path)
    if label_mode != "leaf":
        raise ValueError(f"label_mode must be 'intent' or 'leaf', got {label_mode!r}")
    valid_labels, _ = load_ontology(Path(ontology_path))
    slot_values = load_slot_values(Path(ontology_path))
    leaves: List[str] = []
    for label in sorted(valid_labels):
        if label in slot_values:
            leaves.extend(f"{label}{LEAF_SEP}{v}" for v in slot_values[label])
        else:
            leaves.append(label)
    return {leaf: idx for idx, leaf in enumerate(leaves)}


def row_label(row: dict, label_mode: str) -> str:
    """Class key for one manifest row under label_mode.

    Slot values only apply to the 6 slotted intents; a stray slot on a
    fixed/UNKNOWN/SILENCE row (e.g. GSC digit "eight") must NOT produce a leaf key.
    """
    label = row.get("training_label", "")
    slot = (row.get("slot_value") or "").strip()
    if label_mode == "leaf" and slot and label in SLOTTED_INTENTS:
        return f"{label}{LEAF_SEP}{slot}"
    return label


def decode_label(class_key: str) -> Tuple[str, Optional[str]]:
    """'BRIGHTNESS|100 percent' -> ('BRIGHTNESS', '100 percent'); 'PAUSE' -> ('PAUSE', None)."""
    if LEAF_SEP in class_key:
        intent, slot = class_key.split(LEAF_SEP, 1)
        return intent, slot
    return class_key, None


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
class AudioProcessor:
    """Waveform -> normalized log-mel (1, n_mels, n_valid_frames), no padding."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 40,
        n_fft: int = 512,
        hop_length: int = 160,
        f_min: int = 50,
        f_max: int = 7600,
        top_db: float = 80.0,
        trim_db: Optional[float] = 40.0,
        trim_margin_ms: float = 100.0,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.trim_db = trim_db
        self.trim_margin = int(sample_rate * trim_margin_ms / 1000)
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft,
            hop_length=hop_length, f_min=f_min, f_max=f_max,
        )
        self.amp_to_db = T.AmplitudeToDB(stype="power", top_db=top_db)
        self._resamplers: Dict[int, T.Resample] = {}

    def to_mono_16k(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            if sr not in self._resamplers:
                self._resamplers[sr] = T.Resample(sr, self.sample_rate)
            waveform = self._resamplers[sr](waveform)
        return waveform

    def trim_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        """Energy trim relative to the clip's own peak frame (no-op if disabled)."""
        if self.trim_db is None:
            return waveform
        frame = int(0.025 * self.sample_rate)
        hop = frame // 2
        x = waveform[0]
        if x.numel() <= frame:
            return waveform
        frames = x.unfold(0, frame, hop)
        rms_db = 10.0 * torch.log10(frames.pow(2).mean(dim=1) + 1e-10)
        active = torch.nonzero(rms_db > rms_db.max() - self.trim_db).flatten()
        if active.numel() == 0:
            return waveform
        start = max(0, int(active[0]) * hop - self.trim_margin)
        end = min(x.numel(), int(active[-1]) * hop + frame + self.trim_margin)
        return waveform[:, start:end]

    def energy_crop(self, waveform: torch.Tensor, max_duration: float,
                    jitter: bool = False) -> torch.Tensor:
        '''Keep the ``max_duration`` window with the highest summed energy
        (25 ms frames, 12.5 ms hop, frame RMS). Applied AFTER trim; no-op if the
        clip is already <= max_duration. ``jitter`` (training only) shifts the
        chosen start by +/-0.25 s, clamped to the valid range.'''
        target = int(max_duration * self.sample_rate)
        n = waveform.shape[1]
        if n <= target:
            return waveform
        frame = int(0.025 * self.sample_rate)          # 25 ms
        hop = frame // 2                               # 12.5 ms
        x = waveform[0]
        frames = x.unfold(0, frame, hop)               # (n_frames, frame)
        rms = frames.pow(2).mean(dim=1).sqrt()         # per-frame RMS
        win_frames = target // hop
        if rms.shape[0] <= win_frames:
            return waveform
        cum = torch.cumsum(rms, dim=0)
        window_sums = cum[win_frames - 1:] - torch.cat(
            [torch.zeros(1, device=rms.device), cum[:-win_frames]])
        best_i = int(window_sums.argmax().item())
        start = best_i * hop
        if jitter:
            jit = int(0.25 * self.sample_rate)
            start += random.randint(-jit, jit)
        start = max(0, min(start, n - target))
        return waveform[:, start:start + target]

    def process(self, waveform: torch.Tensor) -> torch.Tensor:
        """(1, samples) -> (1, n_mels, frames), per-mel CMVN over these frames."""
        mel_db = self.amp_to_db(self.mel_spectrogram(waveform))
        mean = mel_db.mean(dim=-1, keepdim=True)
        std = mel_db.std(dim=-1, keepdim=True, unbiased=False)  # no NaN on 1-frame clips
        return (mel_db - mean) / (std + 1e-5)


def fit_frames(feat: torch.Tensor, n_frames: int, random_offset: bool) -> torch.Tensor:
    """Crop or zero-pad (post-CMVN, so 0 == mean) along time to n_frames."""
    t = feat.shape[-1]
    if t >= n_frames:
        start = random.randint(0, t - n_frames) if random_offset else 0
        return feat[..., start:start + n_frames]
    slack = n_frames - t
    left = random.randint(0, slack) if random_offset else 0
    return F.pad(feat, (left, slack - left))


class SpecAugment(torch.nn.Module):
    """SpecAugment: freq + time masking on the (…, freq, time) layout."""

    def __init__(self, freq_mask_param: int = 5, time_mask_param: int = 20,
                 num_masks: int = 2, prob: float = 0.5):
        super().__init__()
        self.num_masks = num_masks
        self.prob = prob
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        if np.random.rand() < self.prob:
            for _ in range(self.num_masks):
                mel_spec = self.freq_masking(mel_spec)
                mel_spec = self.time_masking(mel_spec)
        return mel_spec


def seed_worker(worker_id: int, base_seed: int = 0) -> None:
    """Per-worker RNG seeding + single intra-op thread (many runs share the node's CPUs)."""
    torch.set_num_threads(1)
    seed = base_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class WaveAugment:
    """Train-only waveform augmentation (PHASE2 FALLBACK addendum Step 3).

    Applied AFTER resample, BEFORE trim, in VCMDataset.__getitem__. Each effect
    fires independently with its configured probability. All random draws use
    numpy.random (seeded per-worker via seed_worker for reproducibility).
    """

    def __init__(self, cfg: Optional[dict] = None,
                 noise_bank: Optional[List[torch.Tensor]] = None,
                 sample_rate: int = 16000):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg)
        self.sample_rate = sample_rate
        self.noise_bank = noise_bank or []
        self.c_noise = dict(cfg.get("noise_mix", {}))
        self.c_reverb = dict(cfg.get("reverb", {}))
        self.c_speed = dict(cfg.get("speed_perturb", {}))
        self.c_gain = dict(cfg.get("gain", {}))
        self.c_band = dict(cfg.get("mic_bandlimit", {}))
        self._resamplers: Dict[float, T.Resample] = {}
        self.aug_stats = {"snr": [], "gain_db": [], "rt60": [], "speed_factor": []}

    @staticmethod
    def _p(d: dict, default: float) -> float:
        return float(d.get("p", default))

    def _noise_mix(self, wav: torch.Tensor) -> torch.Tensor:
        if not self.noise_bank or np.random.rand() >= self._p(self.c_noise, 0.6):
            return wav
        noise = self.noise_bank[int(np.random.randint(len(self.noise_bank)))]
        noise = noise.to(wav.device)
        n_len, w_len = noise.shape[-1], wav.shape[-1]
        if n_len >= w_len:
            off = int(np.random.randint(n_len - w_len + 1))
            noise = noise[..., off:off + w_len]
        else:
            reps = (w_len + n_len - 1) // n_len
            noise = noise.repeat(1, reps)[..., :w_len]
        snr = float(np.random.uniform(*self.c_noise.get("snr_db", [5, 25])))
        self.aug_stats["snr"].append(snr)
        p_sig = (wav ** 2).mean() + 1e-10
        p_noise = (noise ** 2).mean() + 1e-10
        scale = (p_sig / (p_noise * (10 ** (snr / 10)))) ** 0.5
        return wav + scale * noise

    def _reverb(self, wav: torch.Tensor) -> torch.Tensor:
        if np.random.rand() >= self._p(self.c_reverb, 0.3):
            return wav
        rt60 = float(np.random.uniform(*self.c_reverb.get("rt60_s", [0.15, 0.6])))
        self.aug_stats["rt60"].append(rt60)
        n = max(8, int(rt60 * self.sample_rate))
        t = torch.arange(n, device=wav.device).float() / self.sample_rate
        rir = torch.randn(n, device=wav.device) * torch.exp(-6.91 * t / rt60)
        rir[0] = 1.0  # direct path peak normalised
        rir = rir / (rir.abs().max() + 1e-8)
        wet = torchaudio.functional.fftconvolve(wav, rir.unsqueeze(0))[..., :wav.shape[-1]]
        return wet

    def _speed(self, wav: torch.Tensor) -> torch.Tensor:
        if np.random.rand() >= self._p(self.c_speed, 0.5):
            return wav
        factor = float(np.random.choice(self.c_speed.get("factors", [0.9, 1.0, 1.1])))
        self.aug_stats["speed_factor"].append(factor)
        if abs(factor - 1.0) < 1e-6:
            return wav
        new_sr = float(round(self.sample_rate * factor))
        if new_sr not in self._resamplers:
            self._resamplers[new_sr] = T.Resample(self.sample_rate, new_sr)
        return self._resamplers[new_sr](wav)

    def _gain(self, wav: torch.Tensor) -> torch.Tensor:
        if np.random.rand() >= self._p(self.c_gain, 0.5):
            return wav
        db = float(self.c_gain.get("db", 6.0))
        g_db = float(np.random.uniform(-db, db))
        self.aug_stats["gain_db"].append(g_db)
        g = 10 ** (g_db / 20.0)
        return torch.clamp(wav * g, -1.0, 1.0)

    def _bandlimit(self, wav: torch.Tensor) -> torch.Tensor:
        if np.random.rand() >= self._p(self.c_band, 0.2):
            return wav
        cutoff = float(np.random.uniform(*self.c_band.get("cutoff_hz", [3500, 7500])))
        return torchaudio.functional.lowpass_biquad(wav, self.sample_rate, cutoff)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return wav
        wav = self._noise_mix(wav)
        wav = self._reverb(wav)
        wav = self._speed(wav)
        wav = self._gain(wav)
        wav = self._bandlimit(wav)
        return wav


# --------------------------------------------------------------------------
# Shared waveform cache (GPU-throughput work, 2026-09-26)
# --------------------------------------------------------------------------
class WaveformCache:
    """Read-only, memory-mapped store of decoded mono 16 kHz float32 waveforms.

    Built once by scripts/build_waveform_cache.py. Every training process (and
    every DataLoader worker) memory-maps the same file, so the OS page cache
    holds ONE copy for all concurrent runs and nothing is decoded per epoch.
    Waveforms are stored exactly as ``AudioProcessor.to_mono_16k`` returns
    them, so cached and uncached paths give identical tensors (verified by
    ``build_waveform_cache.py --verify``).

    The memmap is opened lazily per process and dropped on pickling, so it is
    safe with both fork and spawn DataLoader start methods.
    """

    def __init__(self, cache_dir: str):
        import json
        self.cache_dir = Path(cache_dir)
        meta = json.loads((self.cache_dir / "index.json").read_text())
        self.sample_rate = int(meta["sample_rate"])
        self.total = int(meta["total_samples"])
        self.index: Dict[str, Tuple[int, int]] = {k: (int(v[0]), int(v[1])) for k, v in meta["index"].items()}
        self._mm = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None
        return state

    def __contains__(self, rel_path: str) -> bool:
        return rel_path in self.index

    def get(self, rel_path: str) -> torch.Tensor:
        if self._mm is None:
            self._mm = np.memmap(self.cache_dir / "waves.f32", dtype=np.float32, mode="r",
                                 shape=(self.total,))
        off, n = self.index[rel_path]
        return torch.from_numpy(np.array(self._mm[off:off + n])).unsqueeze(0)  # (1, n) copy


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class VCMDataset(Dataset):
    """PyTorch dataset from a canonical CSV manifest (filtered by source_split)."""

    def __init__(
        self,
        manifest_path: str,
        intent_to_label: Dict[str, int],
        audio_processor: AudioProcessor,
        audio_root: Optional[str] = None,
        augment: bool = False,
        max_duration: float = 2.5,
        split_value: Optional[str] = None,
        label_mode: str = "intent",
        spec_augment_cfg: Optional[dict] = None,
        wave_augment_cfg: Optional[dict] = None,
        corpus_filter: Optional[set] = None,
        waveform_cache: Optional[WaveformCache] = None,
        raw_output: bool = False,
        crop_mode: str = "start",
    ):
        self.manifest_path = Path(manifest_path)
        self.audio_root = Path(audio_root) if audio_root else Path.cwd()
        self.waveform_cache = waveform_cache
        self.raw_output = raw_output
        self.crop_mode = crop_mode
        self.max_duration = max_duration
        self.cache_misses = 0
        self.intent_to_label = intent_to_label
        self.audio_processor = audio_processor
        self.augment = augment
        self.label_mode = label_mode
        self.n_frames = int(max_duration * audio_processor.sample_rate / audio_processor.hop_length) + 1

        self.samples: List[dict] = []
        skipped_split = 0
        with open(self.manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if split_value is not None and row.get("source_split") != split_value:
                    skipped_split += 1
                    continue
                if corpus_filter is not None and row.get("corpus_id") not in corpus_filter:
                    continue
                key = row_label(row, label_mode)
                if key not in self.intent_to_label:
                    raise ValueError(
                        f"class {key!r} (label_mode={label_mode}) in {self.manifest_path.name} "
                        f"is not in the ontology-derived mapping ({len(self.intent_to_label)} "
                        f"classes). In leaf mode a slotted row needs a slot_value from "
                        f"ontology.json 'slot_values'."
                    )
                row["_class_key"] = key
                self.samples.append(row)

        if not self.samples:
            raise ValueError(
                f"No usable rows loaded from {self.manifest_path} (split_value={split_value!r}, "
                f"skipped {skipped_split} on split)."
            )
        # rows the shared cache does not cover (decoded from disk every epoch)
        self.uncached_rows = (
            sum(1 for r in self.samples if r["relative_path"] not in waveform_cache)
            if waveform_cache is not None else len(self.samples)
        )

        sa = dict(spec_augment_cfg or {})
        use_sa = augment and sa.pop("spec_augment", True)
        self.spec_augment = SpecAugment(**sa) if use_sa else None

        noise_bank = self._build_noise_bank() if (augment and wave_augment_cfg) else []
        self.noise_bank = noise_bank
        self.wave_augment = WaveAugment(
            wave_augment_cfg if augment else None,
            noise_bank, self.audio_processor.sample_rate,
        )

    def _load_16k(self, rel_path: str) -> torch.Tensor:
        """Mono 16 kHz waveform from the shared cache, or from disk on a miss."""
        if self.waveform_cache is not None and rel_path in self.waveform_cache:
            return self.waveform_cache.get(rel_path)
        self.cache_misses += 1
        wav, sr = torchaudio.load(str(self.audio_root / rel_path))
        return self.audio_processor.to_mono_16k(wav, sr)

    def _build_noise_bank(self) -> List[torch.Tensor]:
        """Train-split noise bank: gsc_noise + personal SILENCE (train only)."""
        bank: List[torch.Tensor] = []
        for row in self.samples:
            corpus = row.get("corpus_id")
            is_noise = corpus == "gsc_noise"
            is_silence = corpus == "personal_awi" and row.get("training_label") == "SILENCE"
            if not (is_noise or is_silence):
                continue
            try:
                bank.append(self._load_16k(row["relative_path"]))
            except Exception:
                continue
        return bank

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        waveform = self._load_16k(sample["relative_path"])
        if self.raw_output:
            return waveform, self.intent_to_label[sample["_class_key"]]
        if self.augment and self.wave_augment is not None:
            waveform = self.wave_augment(waveform)
        waveform = self.audio_processor.trim_silence(waveform)
        if self.crop_mode == "energy":
            waveform = self.audio_processor.energy_crop(
                waveform, self.max_duration, jitter=self.augment)

        feat = self.audio_processor.process(waveform)                 # (1, n_mels, t)
        feat = fit_frames(feat, self.n_frames, random_offset=self.augment)

        if self.spec_augment is not None:
            feat = self.spec_augment(feat)                            # natural (freq, time) layout

        feat = feat.transpose(1, 2)                                   # -> (1, time, n_mels)
        return feat, self.intent_to_label[sample["_class_key"]]


def compute_class_weights(
    samples: List[dict], intent_mapping: Dict[str, int], mode: str = "sqrt_inverse"
) -> torch.Tensor:
    """Class weights from split rows.

    mode: "none" (all ones), "inverse" (total / (n * count)),
    "sqrt_inverse" (square root of inverse, renormalized to mean 1 over
    present classes -- softer; avoids the FSC effect in Step 10 where
    UNKNOWN was down-weighted ~17x). Absent classes get 0.
    """
    counts: Dict[str, int] = {}
    for row in samples:
        key = row.get("_class_key", row["training_label"])
        counts[key] = counts.get(key, 0) + 1
    n = len(intent_mapping)
    total = sum(counts.values())
    weights = torch.zeros(n, dtype=torch.float32)
    for label, idx in intent_mapping.items():
        cnt = counts.get(label, 0)
        if cnt == 0:
            continue
        if mode == "none":
            weights[idx] = 1.0
        elif mode == "inverse":
            weights[idx] = total / (n * cnt)
        elif mode == "sqrt_inverse":
            weights[idx] = (total / cnt) ** 0.5
        else:
            raise ValueError(f"class weighting mode {mode!r} not in none|inverse|sqrt_inverse")
    present = weights > 0
    if mode == "sqrt_inverse" and present.any():
        weights[present] = weights[present] / weights[present].mean()
    return weights


def create_dataloaders(
    train_manifest: str,
    val_manifest: str,
    intent_mapping: Dict[str, int],
    audio_root: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    sample_rate: int = 16000,
    train_split: str = "train",
    val_split: str = "validation",
    data_cfg: Optional[dict] = None,
    label_mode: str = "intent",
    seed: Optional[int] = None,
    val_corpora: Optional[list] = None,
    gpu_pipeline: bool = False,
    crop_mode: str = "start",
    device: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/validation loaders. data_cfg is the `data:` block of train_config.yaml."""
    cfg = dict(data_cfg or {})
    processor = AudioProcessor(
        sample_rate=sample_rate,
        n_mels=int(cfg.get("n_mels", 40)),
        n_fft=int(cfg.get("n_fft", 512)),
        hop_length=int(cfg.get("hop_length", 160)),
        f_min=int(cfg.get("f_min", 50)),
        f_max=int(cfg.get("f_max", 7600)),
        top_db=float(cfg.get("top_db", 80.0)),
        trim_db=cfg.get("trim_db", 40.0),
    )
    max_duration = float(cfg.get("max_duration", 2.5))
    sa_cfg = cfg.get("augmentation", {})

    wave_cfg = cfg.get("wave_augmentation")
    cache_dir = cfg.get("waveform_cache")
    cache = WaveformCache(cache_dir) if cache_dir else None
    common = dict(intent_to_label=intent_mapping, audio_processor=processor,
                  audio_root=audio_root, max_duration=max_duration, label_mode=label_mode,
                  waveform_cache=cache, crop_mode=crop_mode)
    train_dataset = VCMDataset(manifest_path=train_manifest, augment=True,
                               split_value=train_split, spec_augment_cfg=sa_cfg,
                               wave_augment_cfg=wave_cfg, raw_output=gpu_pipeline, **common)
    val_dataset = VCMDataset(manifest_path=val_manifest, augment=False,
                             split_value=val_split,
                             corpus_filter=set(val_corpora) if val_corpora else None,
                             raw_output=gpu_pipeline, **common)

    if gpu_pipeline:
        from .vcm_gpu_pipeline import (GPUBatchProcessor, cache_val_features,
                                       make_gpu_collate, make_val_tensor_loader)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        n_frames = int(max_duration * processor.sample_rate / processor.hop_length) + 1
        train_processor = GPUBatchProcessor(processor, wave_cfg, sa_cfg, n_frames,
                                            train_dataset.noise_bank, dev)
        val_processor = GPUBatchProcessor(processor, None, sa_cfg, n_frames, [], dev)
        val_feats, val_labels = cache_val_features(val_dataset, val_processor, dev)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True,
                                  persistent_workers=num_workers > 0,
                                  collate_fn=make_gpu_collate(train_processor, augment=True, device=dev))
        val_loader = make_val_tensor_loader(val_feats, val_labels, batch_size, num_workers, seed)
        return train_loader, val_loader

    worker_init = lambda wid: seed_worker(wid, seed if seed is not None else 0)  # noqa: E731
    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     persistent_workers=num_workers > 0, worker_init_fn=worker_init)
    if num_workers > 0:
        loader_kw["prefetch_factor"] = int(cfg.get("prefetch_factor", 4))
    return (
        DataLoader(train_dataset, shuffle=True, **loader_kw),
        DataLoader(val_dataset, shuffle=False, **loader_kw),
    )
