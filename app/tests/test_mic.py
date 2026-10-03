import numpy as np
import pytest

from laptop.app.mic import To16k


def test_passthrough_at_16k_frames_exactly():
    c = To16k(16000)
    frames = c.push(np.zeros(700, dtype=np.int16))
    assert len(frames) == 2 and all(len(f) == 640 for f in frames)
    assert len(c.push(np.zeros(260, dtype=np.int16))) == 1   # 60 carried + 260 = 320


@pytest.mark.parametrize("rate", [44100, 48000])
def test_resampled_tone_keeps_pitch_and_level(rate):
    pytest.importorskip("soxr")
    t = np.arange(rate) / rate
    x = (0.3 * 32767 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    c = To16k(rate)
    out = []
    for i in range(0, x.size, int(rate * 0.02)):     # 20 ms device blocks
        out += c.push(x[i:i + int(rate * 0.02)])
    y = np.frombuffer(b"".join(out), dtype="<i2").astype(np.float64)[1600:-1600]
    assert abs(y.size + 3200 - 16000) < 1000   # soxr holds back ~40 ms internally
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    peak_hz = np.argmax(spec) * 16000 / y.size
    assert abs(peak_hz - 440) < 5
    assert abs(np.sqrt(np.mean(y ** 2)) / 32767 - 0.3 / np.sqrt(2)) < 0.02
