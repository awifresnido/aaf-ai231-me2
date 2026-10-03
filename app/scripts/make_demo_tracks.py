"""Write 3 licence-free synthetic demo tracks (for testing play/next/volume/ducking).

    python scripts/make_demo_tracks.py [out_dir]      # default: media/

Each track is a 30 s loopable chord progression with a soft pulse, 44.1 kHz
stereo WAV, at about -18 dBFS so ducking and volume steps are clearly audible.
"""
import sys
import wave
from pathlib import Path

import numpy as np

SR = 44_100
SECONDS = 30
TRACKS = {  # name -> chord roots (Hz) and pulse tempo (BPM)
    "Demo_Track_1_Morning": ([261.6, 196.0, 220.0, 174.6], 96),
    "Demo_Track_2_Drive": ([293.7, 246.9, 196.0, 220.0], 118),
    "Demo_Track_3_Evening": ([220.0, 174.6, 261.6, 196.0], 84),
}


def track(roots, bpm):
    t = np.arange(SR * SECONDS) / SR
    bar_s = 4 * 60 / bpm
    y = np.zeros_like(t)
    for i, root in enumerate(roots * int(np.ceil(SECONDS / (bar_s * len(roots))))):
        a, b = i * bar_s, (i + 1) * bar_s
        m = (t >= a) & (t < b)
        tt = t[m] - a
        env = np.minimum(1, tt / 0.05) * np.minimum(1, (b - a - tt) / 0.05)
        for ratio, g in ((1, 0.5), (1.26, 0.3), (1.5, 0.3), (2, 0.15)):
            y[m] += g * env * np.sin(2 * np.pi * root * ratio * tt)
    beat = 60 / bpm
    pulse = np.exp(-((t % beat) / 0.06)) * np.sin(2 * np.pi * 110 * t)
    y = y + 0.4 * pulse
    y *= 0.125 / np.sqrt(np.mean(y ** 2))           # ~ -18 dBFS RMS
    left, right = y, np.roll(y, int(0.012 * SR))    # slight stereo width
    return (np.clip(np.stack([left, right], 1), -1, 1) * 32767).astype("<i2")


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "media"
    out.mkdir(parents=True, exist_ok=True)
    for name, (roots, bpm) in TRACKS.items():
        with wave.open(str(out / f"{name}.wav"), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(track(roots, bpm).tobytes())
        print(f"wrote {out / (name + '.wav')}")


if __name__ == "__main__":
    main()
