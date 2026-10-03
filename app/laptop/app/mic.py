"""Laptop microphone capture -> 16 kHz PCM16 20 ms frames -> edge link.

The device is opened at its NATIVE rate (often 44.1/48 kHz) and resampled in
software; asking the driver for 16 kHz is the classic silent failure (see
../openwakeword/WAKE_WORD_STATUS.md). Use native Windows Python: WSL has no
reliable microphone path.

    python -m laptop.app.mic      # list input devices and measure the default mic
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

import numpy as np

from vcm_common.protocol import AUDIO_FRAME_SAMPLES, AUDIO_SAMPLE_RATE_HZ

log = logging.getLogger(__name__)


class To16k:
    """Streaming resampler: device-rate int16 mono -> 16 kHz int16, 320-sample frames.

    Uses soxr's stateful stream, so block boundaries leave no artefacts.
    soxr holds back ~40 ms internally; that latency is negligible here.
    """

    def __init__(self, in_rate: int):
        self.in_rate = int(in_rate)
        self._stream = None
        if self.in_rate != AUDIO_SAMPLE_RATE_HZ:
            import soxr  # type: ignore

            self._stream = soxr.ResampleStream(self.in_rate, AUDIO_SAMPLE_RATE_HZ, 1,
                                               dtype="int16", quality="HQ")
        self._pending = np.zeros(0, dtype=np.int16)

    def push(self, block: np.ndarray) -> list[bytes]:
        x = block.reshape(-1).astype(np.int16, copy=False)
        if self._stream is not None:
            x = self._stream.resample_chunk(x)
        self._pending = np.concatenate([self._pending, x])
        n = self._pending.size // AUDIO_FRAME_SAMPLES
        frames = [self._pending[i * AUDIO_FRAME_SAMPLES:(i + 1) * AUDIO_FRAME_SAMPLES].tobytes()
                  for i in range(n)]
        self._pending = self._pending[n * AUDIO_FRAME_SAMPLES:]
        return frames


class MicStreamer:
    def __init__(self, enabled: bool, device, send: Callable[[bytes], Awaitable[None]]):
        self.enabled = enabled
        self.device = device
        self.send = send
        self.status = "disabled" if not enabled else "starting"
        self.detail: Optional[str] = None
        self.device_rate: Optional[int] = None
        self._stream = None

    async def start(self) -> None:
        if not self.enabled:
            return
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=400)

        try:
            import sounddevice as sd  # type: ignore

            info = sd.query_devices(self.device, "input")
            rate = int(info["default_samplerate"])
            conv = To16k(rate)

            def callback(indata, frames, time_info, status):  # PortAudio thread
                if status:
                    log.debug("mic status: %s", status)
                for fr in conv.push(indata[:, 0].copy()):
                    loop.call_soon_threadsafe(lambda f=fr: queue.full() or queue.put_nowait(f))

            self._stream = sd.InputStream(samplerate=rate, channels=1, dtype="int16",
                                          blocksize=int(rate * 0.02), device=self.device,
                                          callback=callback)
            self._stream.start()
            self.device_rate = rate
            self.status, self.detail = "streaming", f"{info['name']} @ {rate} Hz -> 16 kHz"
            log.info("microphone: %s", self.detail)
        except Exception as exc:  # noqa: BLE001
            self.status, self.detail = "error", str(exc)
            log.error("microphone unavailable: %s", exc)
            return

        async def pump() -> None:
            while True:
                await self.send(await queue.get())

        asyncio.create_task(pump())

    def snapshot(self) -> dict:
        return {"enabled": self.enabled, "status": self.status, "detail": self.detail,
                "device_rate": self.device_rate}


if __name__ == "__main__":
    import sounddevice as sd  # type: ignore

    print(sd.query_devices())
    dev = sd.default.device[0]
    rate = int(sd.query_devices(dev, "input")["default_samplerate"])
    print(f"\ndefault input: {dev} ({sd.query_devices(dev)['name']}, native {rate} Hz)")
    print("recording 2 s -- speak now ...")
    x = sd.rec(2 * rate, samplerate=rate, channels=1, dtype="int16")
    sd.wait()
    y = np.frombuffer(b"".join(To16k(rate).push(x[:, 0])), dtype="<i2")
    rms = float(np.sqrt(np.mean((y.astype(np.float64) / 32768.0) ** 2)))
    print(f"RMS level after resampling: {20 * np.log10(rms + 1e-12):.1f} dBFS "
          "(speech is typically -40 to -15; below -45 means wrong device)")
