"""Unified inference path shared by evaluate_vcm.py and the RPi demo.

predict_wav mirrors EXACTLY what the RPi will run:
    load WAV -> to_mono_16k -> trim_silence -> process (log-mel + CMVN)
    -> fit_frames(random_offset=False) -> model -> softmax -> (class_key, prob)

Model/pipeline configuration (2026-09-26):
    If ``model_config.json`` sits next to the checkpoint (written by
    train_vcm.py for every run from the CRNN-Attn work onward), it is the
    single source of truth for arch, model hyperparameters, label mode,
    n_mels, max_duration and the mel front-end. Constructor arguments are then
    ignored except ``device``. Without a sidecar (e.g. B2, trained before the
    sidecar existed) the constructor arguments are used, as before.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torchaudio

from .vcm_data_loader import AudioProcessor, create_label_mapping, fit_frames

ONTOLOGY = "configs/ontology.json"
logger = logging.getLogger(__name__)


def load_model_config(checkpoint: Union[str, Path]) -> Optional[dict]:
    sidecar = Path(checkpoint).with_name("model_config.json")
    return json.loads(sidecar.read_text()) if sidecar.is_file() else None


class VCMInferencer:
    def __init__(
        self,
        checkpoint: Union[str, Path],
        device: str = "cpu",
        arch: str = "tcresnet",
        width_mult: float = 1.0,
        label_mode: str = "leaf",
        n_mels: int = 40,
        max_duration: float = 3.0,
        sample_rate: int = 16000,
        hop_length: int = 160,
        crop_mode: str = "start",
    ):
        self.device = torch.device(device)
        cfg = load_model_config(checkpoint)
        if cfg is not None:
            model_cfg = dict(cfg["model_cfg"])
            label_mode = cfg["label_mode"]
            n_mels = int(cfg["n_mels"])
            max_duration = float(cfg["max_duration"])
            sample_rate = int(cfg["sample_rate"])
            hop_length = int(cfg["hop_length"])
            fe = dict(n_fft=int(cfg["n_fft"]), f_min=int(cfg["f_min"]), f_max=int(cfg["f_max"]),
                      top_db=float(cfg["top_db"]), trim_db=cfg["trim_db"])
            self.config_source = "sidecar"
        else:
            model_cfg = {"arch": arch, "width_mult": width_mult, "dropout_rate": 0.1}
            fe = dict(n_fft=512, f_min=50, f_max=7600, top_db=80.0, trim_db=40.0)
            self.config_source = "constructor"
        self.arch = model_cfg.get("arch", "tcresnet")
        self.label_mode = label_mode
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.hop_length = hop_length
        self.crop_mode = crop_mode

        from .vcm_models_v2 import build_model
        self.mapping = create_label_mapping(ONTOLOGY, label_mode)
        self.idx_to_key = {v: k for k, v in self.mapping.items()}
        n_classes = len(self.mapping)
        if cfg is not None and int(cfg["n_classes"]) != n_classes:
            raise ValueError(
                f"{checkpoint}: trained with {cfg['n_classes']} classes but the current ontology "
                f"gives {n_classes} ({label_mode}). The ontology changed since training."
            )

        self.processor = AudioProcessor(
            sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length, **fe,
        )
        self.n_frames = int(max_duration * sample_rate / hop_length) + 1

        self.model = build_model(model_cfg, n_classes=n_classes, n_mels=n_mels).to(self.device)
        ckpt = torch.load(str(checkpoint), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])  # strict: a mismatch fails loudly
        self.model.eval()
        logger.info(f"VCMInferencer: {self.arch} from {checkpoint} (config: {self.config_source})")

    @torch.no_grad()
    def predict_wav(self, path_or_array: Union[str, Path, torch.Tensor]):
        """Return (class_key, prob) for a WAV path or a (1, samples) waveform."""
        if isinstance(path_or_array, (str, Path)):
            wav, sr = torchaudio.load(str(path_or_array))
        else:
            wav, sr = path_or_array, self.sample_rate
        # feature extraction runs on CPU (as in training); only the feature moves to device
        wav = self.processor.to_mono_16k(wav, sr)
        wav = self.processor.trim_silence(wav)
        if self.crop_mode == "energy":
            wav = self.processor.energy_crop(wav, self.max_duration, jitter=False)
        feat = self.processor.process(wav)                    # (1, n_mels, t)
        feat = fit_frames(feat, self.n_frames, random_offset=False)
        feat = feat.transpose(1, 2).unsqueeze(0).to(self.device)  # (1, 1, time, n_mels)
        logits = self.model(feat)
        prob = torch.softmax(logits, dim=-1)[0]
        idx = int(prob.argmax().item())
        return self.idx_to_key[idx], float(prob[idx].item())

    def is_command(self, class_key: str) -> bool:
        return class_key not in ("UNKNOWN", "SILENCE")
