"""Optional wake chime played ON THE EDGE (Pi speaker).

The laptop UI always plays its own chime through the browser; this one is for
the Pi setup where the speaker is attached to the edge. Off by default.
The chime is synthesised (no asset licensing) and written to a temp WAV once.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

CHIME_SR = 22_050
CHIME_TONES_HZ = (880.0, 1318.5)   # A5 -> E6, a rising fifth
CHIME_TONE_S = 0.09
CHIME_GAIN = 0.18


def chime_samples(sr: int = CHIME_SR) -> np.ndarray:
    parts = []
    for f in CHIME_TONES_HZ:
        t = np.arange(int(CHIME_TONE_S * sr)) / sr
        env = np.minimum(1.0, t / 0.005) * np.exp(-t / 0.045)   # 5 ms attack, soft decay
        parts.append(np.sin(2 * np.pi * f * t) * env)
    return (CHIME_GAIN * np.concatenate(parts)).astype(np.float32)


def write_chime_wav(path: Path) -> Path:
    x = (chime_samples() * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(CHIME_SR)
        w.writeframes(x.tobytes())
    return path


class EdgeChime:
    def __init__(self, enabled: bool, player: str = "aplay"):
        self.enabled = enabled
        self.player = player
        self.path: Optional[Path] = None
        self.detail: Optional[str] = None
        if not enabled:
            return
        if shutil.which(player) is None:
            self.enabled, self.detail = False, f"{player} not installed"
            return
        self.path = write_chime_wav(Path(tempfile.gettempdir()) / "tinyvcm_chime.wav")

    def _argv(self) -> list[str]:
        name = Path(self.player).name
        if name == "aplay":
            return [self.player, "-q", str(self.path)]
        if name == "mpv":
            return [self.player, "--no-video", "--really-quiet", str(self.path)]
        return [self.player, str(self.path)]      # pw-play, paplay, ...

    def play(self) -> None:
        if not self.enabled or self.path is None:
            return
        try:
            subprocess.Popen(self._argv(),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            self.detail = str(exc)
            log.warning("edge chime failed: %s", exc)
