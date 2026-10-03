"""Music playback on the edge, including wake-word ducking.

The logical volume (what the user chose, 1..5) and the physical output level
(what the speaker is doing right now) are kept separate, so ducking can never
permanently change the user's volume.

Post-command rules (brainstormed plan, section 13):

=============  =============================================
intent         after the command window closes
=============  =============================================
PLAY_MUSIC     restore level, start/resume playback
PAUSE          pause (not audible), restore level silently
STOP           stop, restore level silently
NEXT           next track, restore level
VOLUME_UP      restore, then level + 1 (max 5)
VOLUME_DOWN    restore, then level - 1 (min 1)
anything else  restore level (incl. rejected commands)
=============  =============================================
"""
from __future__ import annotations

import json
import logging
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

log = logging.getLogger(__name__)

MEDIA_INTENTS = {"PLAY_MUSIC", "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN"}
AUDIO_EXTENSIONS = {".mp3", ".ogg", ".flac", ".wav", ".m4a"}


class Backend(Protocol):
    name: str
    available: bool
    detail: Optional[str]

    def load(self, path: Optional[Path]) -> None: ...
    def set_paused(self, paused: bool) -> None: ...
    def stop(self) -> None: ...
    def set_volume_percent(self, pct: int) -> None: ...


class SimulatedBackend:
    """No audio output: state only. Used on laptops/WSL during development."""

    name = "simulated"
    available = True
    detail = "no speaker: playback is simulated"

    def load(self, path: Optional[Path]) -> None: ...
    def set_paused(self, paused: bool) -> None: ...
    def stop(self) -> None: ...
    def set_volume_percent(self, pct: int) -> None: ...


class MpvBackend:
    """mpv controlled through its JSON IPC socket (Pi + Bluetooth speaker)."""

    name = "mpv"

    def __init__(self, socket_path: str = "/tmp/tinyvcm-mpv.sock"):
        self.socket_path = socket_path
        self.available = False
        self.detail: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        if shutil.which("mpv") is None:
            self.detail = "mpv not installed"
            return
        self._proc = subprocess.Popen(
            ["mpv", "--idle=yes", "--no-video", "--really-quiet", "--loop-file=inf",
             f"--input-ipc-server={socket_path}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(50):  # wait up to ~5 s for the socket
            if Path(socket_path).exists():
                self.available = True
                break
            time.sleep(0.1)
        if not self.available:
            self.detail = "mpv IPC socket did not appear"

    def _cmd(self, *args) -> None:
        if not self.available:
            return
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect(self.socket_path)
                s.sendall((json.dumps({"command": list(args)}) + "\n").encode())
        except OSError as exc:
            self.detail = f"mpv IPC error: {exc}"
            log.warning(self.detail)

    def load(self, path: Optional[Path]) -> None:
        if path is not None:
            self._cmd("loadfile", str(path), "replace")
            self._cmd("set_property", "pause", False)

    def set_paused(self, paused: bool) -> None:
        self._cmd("set_property", "pause", paused)

    def stop(self) -> None:
        self._cmd("stop")

    def set_volume_percent(self, pct: int) -> None:
        self._cmd("set_property", "volume", pct)


@dataclass
class MediaController:
    backend: Backend
    tracks: list[Path] = field(default_factory=list)
    volume_steps_pct: tuple[int, ...] = (20, 40, 60, 80, 100)
    duck_volume_pct: int = 10
    level: int = 3                 # logical volume, 1..len(volume_steps_pct)
    status: str = "stopped"        # playing | paused | stopped
    track_index: int = 0
    ducked: bool = False
    started_at_ms: Optional[float] = None

    @classmethod
    def from_folder(cls, backend: Backend, folder: Optional[str], **kw) -> "MediaController":
        tracks: list[Path] = []
        if folder and Path(folder).expanduser().is_dir():
            tracks = sorted(p for p in Path(folder).expanduser().iterdir()
                            if p.suffix.lower() in AUDIO_EXTENSIONS)
        mc = cls(backend=backend, tracks=tracks, **kw)
        mc.backend.set_volume_percent(mc.level_pct)
        return mc

    # ------------------------------------------------------------------
    @property
    def max_level(self) -> int:
        return len(self.volume_steps_pct)

    @property
    def level_pct(self) -> int:
        return self.volume_steps_pct[self.level - 1]

    def _track(self) -> Optional[Path]:
        return self.tracks[self.track_index] if self.tracks else None

    def _restore(self) -> None:
        self.ducked = False
        self.backend.set_volume_percent(self.level_pct)

    # ------------------------------------------------------------------
    def duck(self) -> bool:
        """Lower the output for the command window. Returns True if it ducked."""
        if self.status != "playing" or self.ducked:
            return False
        self.ducked = True
        self.backend.set_volume_percent(min(self.duck_volume_pct, self.level_pct))
        return True

    def play(self) -> None:
        if self.status == "paused":
            self.backend.set_paused(False)
        elif self.status != "playing":
            self.backend.load(self._track())
            self.started_at_ms = time.time() * 1000
        self.status = "playing"

    def pause(self) -> None:
        if self.status == "playing":
            self.backend.set_paused(True)
            self.status = "paused"

    def stop(self) -> None:
        self.backend.stop()
        self.status = "stopped"

    def next(self) -> None:
        if self.tracks:
            self.track_index = (self.track_index + 1) % len(self.tracks)
        else:
            self.track_index += 1
        self.backend.load(self._track())
        self.started_at_ms = time.time() * 1000
        self.status = "playing"

    def volume_up(self) -> None:
        self.level = min(self.max_level, self.level + 1)

    def volume_down(self) -> None:
        self.level = max(1, self.level - 1)

    def apply(self, intent: Optional[str]) -> bool:
        """Close the command window. ``intent`` is None for rejected commands.

        Returns True when the intent was a media command handled here.
        """
        handled = intent in MEDIA_INTENTS
        if intent == "PLAY_MUSIC":
            self.play()
        elif intent == "PAUSE":
            self.pause()
        elif intent == "STOP":
            self.stop()
        elif intent == "NEXT":
            self.next()
        elif intent == "VOLUME_UP":
            self.volume_up()
        elif intent == "VOLUME_DOWN":
            self.volume_down()
        self._restore()
        return handled

    def snapshot(self) -> dict:
        t = self._track()
        return {
            "status": self.status,
            "level": self.level,
            "max_level": self.max_level,
            "level_pct": self.level_pct,
            "ducked": self.ducked,
            "track_index": self.track_index,
            "track_count": len(self.tracks),
            "track_title": t.stem.replace("_", " ") if t else f"Demo track {self.track_index + 1}",
            "started_at_ms": self.started_at_ms,
            "backend": self.backend.name,
            "backend_ok": self.backend.available,
            "backend_detail": self.backend.detail,
        }


def build_media(cfg: dict) -> MediaController:
    backend: Backend
    if cfg.get("backend", "simulated") == "mpv":
        backend = MpvBackend(cfg.get("mpv_socket", "/tmp/tinyvcm-mpv.sock"))
        if not backend.available:
            log.warning("mpv unavailable (%s); falling back to simulated playback", backend.detail)
            sim = SimulatedBackend()
            sim.detail = f"mpv unavailable ({backend.detail}); simulated"
            backend = sim
    else:
        backend = SimulatedBackend()
    return MediaController.from_folder(
        backend, cfg.get("folder"),
        volume_steps_pct=tuple(cfg.get("volume_steps_pct", (20, 40, 60, 80, 100))),
        duck_volume_pct=int(cfg.get("duck_volume_pct", 10)),
        level=int(cfg.get("initial_level", 3)),
    )
