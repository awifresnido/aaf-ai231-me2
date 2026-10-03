#!/usr/bin/env python3
"""Voice-command test bench for the three tiny-VCM candidates (AI231 ME2).

Mirrors test_hey_rhasspy.py in ../openwakeword, but for the COMMAND models:

  A2  -> intent mode, 21 classes  (checkpoints/A2_s0/best.pt)
  A7  -> leaf   mode, 33 classes  (checkpoints/A7_s0/best.pt)
  B2  -> leaf   mode, 33 classes  (checkpoints/B2_s0/best.pt)  <- deployed

Question: on MY voice, with MY microphone, do the candidates say the right
command? Everything is measured live; nothing is assumed.

Modes
-----
  --list                     enumerate audio devices
  --probe SEC                per-device level probe while you talk (find the mic)
  --selftest                 load all three models, predict on a real clip (no mic)
  --takes [N]                guided: say each command, score all three models
  --wav FILE [--expect KEY]  score one WAV (any rate/channels)
  --wav-dir DIR [--manifest CSV]   score a folder; with --manifest, grade against
                              the true labels and report per-model accuracy
  --live                     rolling 3 s window, live top-1 per model

Command selection (takes mode):
  --scope leaf|intent        leaf (default) = 31 phrases, intent = 19
  --commands "LIGHT_ON,VOLUME_UP"   explicit comma list (leaf or intent keys)
  --shuffle                  random order

Options
-------
  --models A2,A7,B2          which candidates (default all three)
  --tau 0.80                 deployed accept threshold
  --device N|substring       input device
  --take-seconds 4.0         capture window (matches the app's window_s)
  --topk 3                   show runner-up labels
  --json OUT                 machine-readable report

Run with the env that has torch:  ~/miniconda3/envs/ai231/bin/python
(needs sounddevice once:  pip install sounddevice)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "training"))  # make src/ importable

ONTOLOGY = ROOT / "configs" / "ontology.json"
PHRASES_CSV = ROOT / "configs" / "command_phrases.csv"
CHECKPOINTS = {
    "A2": (ROOT / "checkpoints" / "A2_s0" / "best.pt", "intent"),
    "A7": (ROOT / "checkpoints" / "A7_s0" / "best.pt", "leaf"),
    "B2": (ROOT / "checkpoints" / "B2_s0" / "best.pt", "leaf"),
}
SAMPLE_RATE = 16000
# Accept thresholds = the demo app's manifests (app/models/vcm/*/manifest.yaml):
# B2 uses the phase-2 tau; the phase-1 models had no tau selected, the app uses 0.70.
DEFAULT_TAU = {"A2": 0.70, "A7": 0.70, "B2": 0.80}
DEFAULT_WINDOW = 4.0
NON_COMMAND = {"UNKNOWN", "SILENCE"}
SLOTTED = set(json.loads(ONTOLOGY.read_text(encoding="utf-8"))["labels"]["slotted"])

C_OK, C_BAD, C_DIM, C_OFF = "\033[32m", "\033[31m", "\033[2m", "\033[0m"


# --------------------------------------------------------------------------- #
# audio helpers (faithful copy of the fixed Mic from ../openwakeword)
# --------------------------------------------------------------------------- #
def bar(score: float, width: int = 20) -> str:
    filled = int(round(max(0.0, min(1.0, score)) * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean((x.astype(np.float64) / 32768.0) ** 2)))
    return 20.0 * math.log10(rms + 1e-12)


def resample(x: np.ndarray, src_rate: int, dst_rate: int = SAMPLE_RATE) -> np.ndarray:
    if src_rate == dst_rate:
        return x.astype(np.int16, copy=False)
    g = math.gcd(int(src_rate), int(dst_rate))
    up, down = dst_rate // g, src_rate // g
    try:
        from scipy.signal import resample_poly

        y = resample_poly(x.astype(np.float32), up, down)
    except Exception:  # pragma: no cover
        n_out = int(round(len(x) * dst_rate / src_rate))
        y = np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x.astype(np.float64))
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)


class BlockResampler:
    def __init__(self, native_rate: int):
        self.native_rate = int(native_rate)
        self.block_ms = 80
        self.block_frames = int(round(self.native_rate * self.block_ms / 1000))

    def push(self, block: np.ndarray) -> np.ndarray:
        return resample(block, self.native_rate, SAMPLE_RATE)


def read_wav(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        sr, ch, sw, n = f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()
        raw = f.readframes(n)
    if sw != 2:
        raise ValueError(f"{path}: need 16-bit PCM WAV, found {sw * 8}-bit")
    x = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return x, sr


def write_wav(path: Path, x: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(x.astype(np.int16).tobytes())


def speech_stats(x: np.ndarray, win: int = 400, gap_s: float = 0.15):
    n = len(x) // win
    if n == 0:
        return None
    seg = x[: n * win].reshape(n, win).astype(np.float64) / 32768.0
    env = 20 * np.log10(np.sqrt((seg ** 2).mean(axis=1)) + 1e-12)
    floor = float(np.percentile(env, 20))
    thr = max(floor + 12, -58)
    idx = np.where(env > thr)[0]
    if idx.size == 0:
        return None
    step = win / SAMPLE_RATE
    merged = 1
    for a, b in zip(idx, idx[1:]):
        if (b - a) * step > gap_s:
            merged += 1
    speech_db = float(np.percentile(env[idx], 90))
    tail_db = float(env[-max(1, int(0.16 / step)):].mean())
    return idx[0] * step, (idx[-1] + 1) * step, merged, tail_db, speech_db, floor


def _sd():
    try:
        import sounddevice as sd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            f"{C_BAD}ERROR{C_OFF} sounddevice is not importable ({exc}).\n"
            "       install it into this env: pip install sounddevice\n"
            "       (Linux also needs libportaudio2: sudo apt install -y libportaudio2)"
        )
    return sd


def list_devices() -> None:
    sd = _sd()
    try:
        default_in, _ = sd.default.device
    except Exception:
        default_in = None
    print("input devices (use --device with the index or a name substring):\n")
    hostapis = sd.query_hostapis()
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        api = hostapis[d["hostapi"]]["name"] if d["hostapi"] < len(hostapis) else "?"
        mark = " <-- default" if i == default_in else ""
        print(f"  [{i}] {d['name']}  | api={api} | default rate={int(d['default_samplerate'])} "
              f"| in={d['max_input_channels']}{mark}")
    print(f"\ndefault input index: {default_in}")


def resolve_device(spec: Optional[str]):
    sd = _sd()
    if spec is None:
        return None, sd.query_devices(kind="input")
    if spec.isdigit():
        return int(spec), sd.query_devices(int(spec))
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and spec.lower() in d["name"].lower():
            return i, d
    raise SystemExit(f"{C_BAD}ERROR{C_OFF} no input device matches '{spec}' (run --list)")


class Mic:
    """Microphone captured at the hardware rate; software-resampled to 16 kHz.

    Callback stream -> bounded queue of (arrival_time, block). record() flushes
    everything older than a short pre-roll, so a take only contains FRESH audio
    arriving after the cue (a blocking stream would hand back stale buffered
    audio from the pause between takes and the take would sound cut off).
    """

    PREROLL_S = 0.3
    BACKLOG_S = 10.0

    def __init__(self, device_spec: Optional[str] = None, block_ms: int = 80, verbose: bool = True):
        import collections
        import threading

        sd = _sd()
        self.sd = sd
        try:
            idx, info = resolve_device(device_spec)
        except SystemExit:
            raise
        except Exception as exc:
            raise SystemExit(
                f"{C_BAD}ERROR{C_OFF} no usable input device ({exc}).\n"
                "       run --list to see what this Python can see.\n"
                "       WSL: the Windows mic needs WSLg forwarding; or record on Windows"
                " and score with --wav-dir."
            )
        self.device, self.info = idx, info
        self.rate = int(info["default_samplerate"])
        self.rs = BlockResampler(self.rate)
        self.block_frames = self.rs.block_frames
        self._q = collections.deque(maxlen=max(4, int(self.BACKLOG_S * 1000 / block_ms)))
        self._cv = threading.Condition()
        self.overflows = 0
        self.last_record_wall_s = 0.0
        if verbose:
            print(
                f"mic: [{idx if idx is not None else 'default'}] {info['name']}\n"
                f"     hardware rate {self.rate} Hz -> software resample to {SAMPLE_RATE} Hz "
                f"(blocks of {self.block_frames} samples = {block_ms} ms)"
            )

        def callback(indata, frames, time_info, status):  # PortAudio thread
            if status and status.input_overflow:
                self.overflows += 1
            with self._cv:
                self._q.append((time.monotonic(), indata[:, 0].copy()))
                self._cv.notify()

        try:
            self.stream = sd.InputStream(
                device=idx, channels=1, samplerate=self.rate, dtype="int16",
                blocksize=self.block_frames, callback=callback,
            )
            self.stream.start()
        except Exception as exc:
            raise SystemExit(
                f"{C_BAD}ERROR{C_OFF} could not open the microphone at {self.rate} Hz: {exc}\n"
                "       WSL: is the Windows mic forwarded (WSLg)? try --device."
            )

    def _next(self, timeout: float = 2.0) -> Tuple[float, np.ndarray]:
        with self._cv:
            if not self._q and not self._cv.wait_for(lambda: bool(self._q), timeout=timeout):
                raise SystemExit(
                    f"{C_BAD}ERROR{C_OFF} the microphone delivered no audio for {timeout:.0f}s "
                    "(device stalled or disconnected; WSL: check WSLg audio / --device)"
                )
            return self._q.popleft()

    def flush(self, keep_s: float = 0.0) -> None:
        cutoff = time.monotonic() - keep_s
        with self._cv:
            while self._q and self._q[0][0] < cutoff:
                self._q.popleft()

    def read(self) -> np.ndarray:
        return self._next()[1].astype(np.int16, copy=False)

    def record(self, seconds: float, meter: bool = False) -> np.ndarray:
        self.flush(keep_s=self.PREROLL_S)
        need = int(round((seconds + self.PREROLL_S) * self.rate))
        chunks: List[np.ndarray] = []
        got, t_start, last_meter = 0, time.monotonic(), 0.0
        while got < need:
            _, b = self._next(timeout=max(2.0, seconds))
            chunks.append(b)
            got += len(b)
            now = time.monotonic()
            if meter and now - last_meter > 0.1:
                last_meter = now
                lvl = dbfs(b)
                left = max(0.0, (need - got) / self.rate)
                print(f"\r        {C_OK}SPEAK NOW{C_OFF}  {left:3.1f}s left  mic {lvl:6.1f} dBFS "
                      f"{bar(max(0.0, (lvl + 60) / 50), 16)}   ", end="", flush=True)
        native = np.concatenate(chunks)
        native = native[-need:] if len(native) >= need else native
        self.last_record_wall_s = time.monotonic() - t_start
        return resample(native, self.rate)

    def close(self) -> None:
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


def mode_probe(seconds: float, json_out: Optional[Path]) -> int:
    sd = _sd()
    print(f"\n=== device probe: {seconds:.1f}s per input device ===")
    print(f"  {C_OK}>>> KEEP TALKING for the whole probe (count 1..20 out loud) <<<{C_OFF}\n")
    hostapis = sd.query_hostapis()
    rows: List[dict] = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        api = hostapis[d["hostapi"]]["name"] if d["hostapi"] < len(hostapis) else "?"
        rate = int(d["default_samplerate"])
        label = f"[{i}] {d['name']} (api={api}, {rate} Hz)"
        try:
            block = max(256, int(rate * 0.05))
            levels: List[float] = []
            with sd.InputStream(device=i, channels=1, samplerate=rate, dtype="int16", blocksize=block) as st:
                end = time.time() + seconds
                while time.time() < end:
                    data, _ = st.read(block)
                    levels.append(dbfs(data[:, 0]))
            mean, peak = float(np.mean(levels)), float(np.max(levels))
            if mean > -45:
                verdict, colour = "LIVE - this one hears you", C_OK
            elif mean > -58:
                verdict, colour = "faint - too far, or low gain", C_BAD
            else:
                verdict, colour = "silent (not a microphone / muted)", C_BAD
            print(f"  {label:<50.50s} mean {mean:6.1f}  peak {peak:6.1f} dBFS  {colour}{verdict}{C_OFF}")
            rows.append({"index": i, "name": d["name"], "api": api, "rate": rate,
                         "mean_dbfs": round(mean, 1), "peak_dbfs": round(peak, 1)})
        except Exception as exc:
            print(f"  {label:<50.50s} {C_DIM}cannot open: {exc}{C_OFF}")
    live = [r for r in rows if r["mean_dbfs"] > -45]
    print()
    if live:
        best = max(live, key=lambda r: r["mean_dbfs"])
        print(f"  {C_OK}best device: [{best['index']}] {best['name']}{C_OFF}")
        print(f"  use it:  --device {best['index']}")
    else:
        print(f"  {C_BAD}no device heard you.{C_OFF}")
        print("  Windows: Settings > System > Sound > Input - pick the mic, watch its level bar move.")
    if json_out:
        json_out.write_text(json.dumps({"mode": "probe", "seconds": seconds, "devices": rows}, indent=2))
        print(f"  wrote {json_out}")
    return 0


# --------------------------------------------------------------------------- #
# commands + models
# --------------------------------------------------------------------------- #
@dataclass
class Command:
    leaf_key: str          # e.g. "ALARM|6:00 AM" (leaf-mode expected), or "LIGHT_ON"
    intent_key: str        # e.g. "ALARM"
    phrase: str            # what to say, e.g. "alarm 6 am"
    slotted: bool = False

    def label(self) -> str:
        return self.leaf_key if self.slotted else self.intent_key


def _phrases() -> Dict[str, str]:
    d: Dict[str, str] = {}
    for r in csv.DictReader(PHRASES_CSV.open(encoding="utf-8")):
        key, phrase = r["class_key"].strip(), r["phrase"].strip()
        if key not in d or len(phrase) < len(d[key]):
            d[key] = phrase
    return d


def build_commands() -> Tuple[List[Command], List[Command]]:
    """Return (leaf_commands 31, intent_commands 19)."""
    ontology = json.loads(ONTOLOGY.read_text(encoding="utf-8"))
    fixed = sorted(ontology["labels"]["fixed"])
    slotted = sorted(ontology["labels"]["slotted"])
    slot_values = ontology["slot_values"]
    phr = _phrases()
    leaves: List[Command] = []
    for f in fixed:
        leaves.append(Command(leaf_key=f, intent_key=f, phrase=phr.get(f, f.lower().replace("_", " "))))
    for s in slotted:
        for v in slot_values[s]:
            key = f"{s}|{v}"
            phrase = phr.get(key) or phr.get(s) or s.lower().replace("_", " ")
            leaves.append(Command(leaf_key=key, intent_key=s, phrase=phrase, slotted=True))
    seen: Dict[str, Command] = {}
    for c in leaves:
        if c.intent_key not in seen or len(c.phrase) < len(seen[c.intent_key].phrase):
            seen[c.intent_key] = Command(leaf_key=c.intent_key, intent_key=c.intent_key,
                                         phrase=c.phrase, slotted=False)
    intents = [seen[k] for k in sorted(seen)]
    return leaves, intents


def load_candidates(which: Sequence[str]):
    from src.vcm_infer import VCMInferencer

    out: Dict[str, VCMInferencer] = {}
    for name in which:
        path, mode = CHECKPOINTS[name]
        if not path.is_file():
            raise SystemExit(f"{C_BAD}ERROR{C_OFF} checkpoint missing: {path}\n"
                             "       (checkpoints/ is git-ignored; re-sync from the DGX - see MODELS.md)")
        inf = VCMInferencer(str(path), arch="tcresnet", width_mult=1.0, label_mode=mode,
                            n_mels=40, max_duration=3.0, sample_rate=SAMPLE_RATE)
        out[name] = inf
    return out


def predict(inf, x16: np.ndarray) -> Tuple[str, float]:
    import torch

    t = torch.from_numpy(x16.astype(np.float32) / 32768.0).unsqueeze(0)
    return inf.predict_wav(t)


def predict_topk(inf, x16: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
    """(key, prob) sorted desc via the exact predict_wav feature path, then top-k."""
    import torch

    from src.vcm_data_loader import fit_frames

    t = torch.from_numpy(x16.astype(np.float32) / 32768.0).unsqueeze(0)
    wav = inf.processor.to_mono_16k(t, inf.sample_rate)
    wav = inf.processor.trim_silence(wav)
    feat = inf.processor.process(wav)
    feat = fit_frames(feat, inf.n_frames, random_offset=False)
    feat = feat.transpose(1, 2).unsqueeze(0).to(inf.device)
    logits = inf.model(feat)
    prob = torch.softmax(logits, dim=-1)[0]
    top = torch.topk(prob, min(k, prob.numel()))
    return [(inf.idx_to_key[int(i)], float(p)) for i, p in zip(top.indices, top.values)]


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def mode_selftest(candidates: Dict, json_out: Optional[Path]) -> int:
    print("\n=== selftest: load all candidates, predict on a real clip ===")
    ok = True
    sample = None
    for p in sorted((ROOT / "data" / "personal" / "raw").glob("**/*.wav")):
        sample = p
        break
    x16 = read_wav(sample)[0] if sample else np.zeros(SAMPLE_RATE, np.int16)
    for name, inf in candidates.items():
        mode = CHECKPOINTS[name][1]
        key, prob = predict(inf, x16)
        n_params = sum(p.numel() for p in inf.model.parameters())
        print(f"  {name} [{mode}, {len(inf.mapping)} classes, {n_params:,} params] -> "
              f"{key} ({prob:.3f})  on {sample.name if sample else 'silence'}")
        ok &= key in inf.idx_to_key.values()
    status = f"{C_OK}ALL LOADED{C_OFF}" if ok else f"{C_BAD}FAIL{C_OFF}"
    print(f"\n  selftest: {status}")
    if json_out:
        json_out.write_text(json.dumps({"mode": "selftest", "ok": bool(ok),
                                        "candidates": {n: CHECKPOINTS[n][1] for n in candidates}}, indent=2))
    return 0 if ok else 1


def tau_str(taus: Dict[str, float]) -> str:
    return ", ".join(f"{n} {t:.2f}" for n, t in taus.items())


def judge(mode: str, cmd: Command, key: str, prob: float, tau: float) -> Tuple[str, bool, str]:
    """Grade one prediction the way the demo app would act on it.

    outcome: 'correct' = the app executes the intended action
             'wrong'   = the app executes a DIFFERENT action (the costly error)
             'reject'  = the app does nothing (low confidence, UNKNOWN/SILENCE, or an
                         intent-mode model naming a slotted intent, which has no slot value)
    intent_ok: top-1 intent matches, regardless of tau/slot (model-quality view)
    """
    intent = key.split("|", 1)[0]
    intent_ok = intent == cmd.intent_key
    if intent in NON_COMMAND:
        return "reject", intent_ok, f"heard {intent}"
    if prob < tau:
        return "reject", intent_ok, f"conf {prob:.2f} < tau {tau:.2f}"
    if mode == "intent" and intent in SLOTTED:
        return "reject", intent_ok, "no slot value -> app cannot execute"
    expected = cmd.leaf_key if (cmd.slotted and mode == "leaf") else cmd.intent_key
    if key == expected:
        return "correct", intent_ok, ""
    if intent_ok:
        return "wrong", True, f"right intent, wrong slot -> {key}"
    return "wrong", False, f"-> {key}"


def _safe(key: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in key).strip("_")


def mode_takes(candidates: Dict, commands: List[Command], taus: Dict[str, float], take_s: float,
               device: Optional[str], save_dir: Optional[Path], json_out: Optional[Path]) -> int:
    mic = Mic(device)
    print(f"\n=== guided command test: {len(commands)} command(s), tau {tau_str(taus)} ===")
    print("  say each command naturally when SPEAK NOW appears (the meter shows your level)")
    print("  a take with no speech is retaken; type s + Enter at the prompt to skip a command\n")
    results: List[dict] = []
    manifest_rows: List[dict] = []
    try:
        for i, cmd in enumerate(commands, 1):
            tag = f"  [{i}/{len(commands)}] {C_DIM}say:{C_OFF} \"{cmd.phrase}\""
            if cmd.slotted:
                tag += f"   {C_DIM}(= {cmd.leaf_key}){C_OFF}"
            while True:
                if _await_enter(f"{tag}  -- Enter to start").strip().lower() == "s":
                    x = None
                    break
                for c in ("3", "2", "1"):
                    print(f"\r        {c} ...", end="", flush=True)
                    time.sleep(0.5)
                x = mic.record(take_s, meter=True)
                print("\r        done" + " " * 70)
                if mic.last_record_wall_s < 0.8 * take_s:
                    print(f"      {C_BAD}WARN: recording took {mic.last_record_wall_s:.1f}s instead of "
                          f"{take_s:.1f}s - buffered audio; retaking{C_OFF}")
                    continue
                st = speech_stats(x)
                if st is None:
                    print(f"      {C_BAD}NO SPEECH - you missed the cue (not a model result); "
                          f"retaking{C_OFF}")
                    continue
                s, e, _nb, tail, sp, fl = st
                cut = e >= len(x) / SAMPLE_RATE - 0.15 and tail > fl + 12
                print(f"      {C_DIM}speech {s:.2f}-{e:.2f}s, level {sp:.1f} dB{C_OFF}"
                      + (f"  {C_BAD}[still speaking at the cut - say it sooner, or "
                         f"--take-seconds {take_s + 1:.0f}]{C_OFF}" if cut else ""))
                break
            if x is None:
                print(f"      {C_DIM}skipped{C_OFF}")
                continue
            row = {"command": cmd.label(), "phrase": cmd.phrase, "models": {}}
            if save_dir is not None:
                wav = save_dir / f"{i:02d}_{_safe(cmd.label())}.wav"
                write_wav(wav, x)
                row["wav"] = str(wav)
                label, _, slot = cmd.label().partition("|")
                manifest_rows.append({
                    "relative_path": str(wav.relative_to(ROOT)), "training_label": label,
                    "leaf_label": label, "slot_value": slot, "source_split": "live_test",
                    "corpus_id": "bench_takes", "phrase": cmd.phrase,
                })
            for name, inf in candidates.items():
                mode = CHECKPOINTS[name][1]
                key, prob = predict(inf, x)
                outcome, intent_ok, detail = judge(mode, cmd, key, prob, taus[name])
                col = C_OK if outcome == "correct" else C_BAD
                mark = {"correct": "OK  ", "wrong": "MISS", "reject": "REJ "}[outcome]
                print(f"      {name:3s} {col}{mark}{C_OFF}  {key:<28s} {prob:.3f}  {C_DIM}{detail}{C_OFF}")
                row["models"][name] = {"pred": key, "prob": round(prob, 3), "outcome": outcome,
                                       "intent_ok": intent_ok, "tau": taus[name]}
            results.append(row)
    except KeyboardInterrupt:
        print(f"\n  {C_DIM}stopped early - summarising the takes so far{C_OFF}")
    finally:
        mic.close()
    _summarize(results, candidates, taus)
    if save_dir is not None and manifest_rows:
        with (save_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(manifest_rows[0]))
            w.writeheader()
            w.writerows(manifest_rows)
        rel = save_dir.relative_to(ROOT)
        print(f"  saved {len(manifest_rows)} take(s) + manifest.csv to {rel}/")
        print(f"  re-score later:  ./run_commands.sh wav {rel} --manifest {rel}/manifest.csv")
        if json_out is None:
            json_out = save_dir / "report.json"
    if json_out:
        json_out.write_text(json.dumps({"mode": "takes", "taus": taus, "results": results}, indent=2))
        print(f"  wrote {json_out}")
    return 0


def _summarize(results: List[dict], candidates: Dict, taus: Dict[str, float]) -> None:
    print()
    if not results:
        print("  no scored takes")
        return
    for name in candidates:
        rows = [r["models"][name] for r in results if name in r["models"]]
        tot = len(rows)
        if tot == 0:
            continue
        correct = sum(m["outcome"] == "correct" for m in rows)
        wrong = sum(m["outcome"] == "wrong" for m in rows)
        rej = sum(m["outcome"] == "reject" for m in rows)
        intent_ok = sum(bool(m["intent_ok"]) for m in rows)
        acc = correct / tot
        if acc == 1.0:
            verdict = f"{C_OK}USABLE{C_OFF}"
        elif acc >= 0.7:
            verdict = f"{C_OK}MOSTLY USABLE{C_OFF}"
        else:
            verdict = f"{C_BAD}NOT RELIABLE{C_OFF}"
        print(f"  {name} (tau {taus[name]:.2f}): right action {correct}/{tot} ({acc:.0%}), "
              f"WRONG action {wrong}, no action {rej} | top-1 intent {intent_ok}/{tot} -> {verdict}")
    print("\n  right action = the app would do what you said; WRONG action = it would do something")
    print("  else (the costly error); no action = below tau, UNKNOWN/SILENCE, or no slot value.")


def _load_manifest_labels(path: Path) -> Dict[str, Tuple[str, str]]:
    """basename -> (leaf_key, intent_key)."""
    out: Dict[str, Tuple[str, str]] = {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        rel = r.get("relative_path", "")
        label = r.get("leaf_label", "") or r.get("training_label", "")
        slot = (r.get("slot_value") or "").strip()
        leaf = f"{label}|{slot}" if slot else label
        out[Path(rel).name] = (leaf, label)
    return out


def mode_wav(candidates: Dict, paths: Sequence[Path], expect: Optional[str], taus: Dict[str, float],
             topk_n: int, json_out: Optional[Path]) -> int:
    print(f"\n=== scoring {len(paths)} WAV file(s), tau {tau_str(taus)} ===")
    results = []
    for p in sorted(paths):
        x, sr = read_wav(p)
        if sr != SAMPLE_RATE:
            x = resample(x, sr)
        print(f"\n  {p.name}  ({len(x) / SAMPLE_RATE:.2f}s, {dbfs(x):.0f} dB)")
        rec = {"file": p.name, "models": {}}
        for name, inf in candidates.items():
            top = predict_topk(inf, x, topk_n)
            key, prob = top[0]
            line = f"    {name:3s} {key:<28s} {prob:.3f}"
            if len(top) > 1:
                line += "   " + C_DIM + "  ".join(f"{k}={p_:.2f}" for k, p_ in top[1:]) + C_OFF
            if expect:
                col = C_OK if key == expect and prob >= taus[name] else C_BAD
                line += f"   {col}expect {expect}{C_OFF}"
            print(line)
            rec["models"][name] = [{"key": k, "prob": round(p_, 3)} for k, p_ in top]
        results.append(rec)
    if json_out:
        json_out.write_text(json.dumps({"mode": "wav", "taus": taus, "results": results}, indent=2))
        print(f"\n  wrote {json_out}")
    return 0


def mode_wav_dir(candidates: Dict, dir_: Path, manifest: Optional[Path], taus: Dict[str, float],
                 topk_n: int, json_out: Optional[Path]) -> int:
    paths = sorted(dir_.rglob("*.wav"))
    if not paths:
        raise SystemExit(f"{C_BAD}ERROR{C_OFF} no .wav files in {dir_}")
    truth = _load_manifest_labels(manifest) if manifest else {}
    print(f"\n=== scoring {len(paths)} WAV(s){f', grading vs {manifest.name}' if manifest else ''}, "
          f"tau {tau_str(taus)} ===")
    results: List[dict] = []
    for p in paths:
        x, sr = read_wav(p)
        if sr != SAMPLE_RATE:
            x = resample(x, sr)
        true_leaf, true_intent = truth.get(p.name, (None, None))
        cmd = (Command(leaf_key=true_leaf, intent_key=true_intent, phrase="",
                       slotted="|" in true_leaf) if true_leaf else None)
        rec = {"file": p.name, "expected": true_leaf, "models": {}}
        for name, inf in candidates.items():
            key, prob = predict(inf, x)
            m = {"pred": key, "prob": round(prob, 3)}
            if cmd is not None:
                outcome, intent_ok, _ = judge(CHECKPOINTS[name][1], cmd, key, prob, taus[name])
                m.update(outcome=outcome, intent_ok=intent_ok, tau=taus[name])
            rec["models"][name] = m
        results.append(rec)
    graded = [r for r in results if r["expected"]]
    if graded:
        _summarize(graded, candidates, taus)
    else:
        print("  (no manifest labels matched these files - pass --manifest to grade)")
    if json_out:
        json_out.write_text(json.dumps({"mode": "wav-dir", "taus": taus, "results": results}, indent=2))
        print(f"\n  wrote {json_out}")
    return 0


def mode_live(candidates: Dict, taus: Dict[str, float], device: Optional[str], seconds: Optional[float],
              json_out: Optional[Path]) -> int:
    mic = Mic(device)
    print(f"\n=== live command classification (rolling 3.0s window, tau {tau_str(taus)}) ===")
    print("  say a command, watch the top-1 per model; Ctrl-C to stop\n")
    buf = np.zeros(0, np.int16)
    window = int(3.0 * SAMPLE_RATE)
    mic.flush()
    t0 = time.time()
    last_pred = 0.0
    try:
        while True:
            if seconds is not None and time.time() - t0 >= seconds:
                break
            block = mic.read()
            buf = np.concatenate([buf, mic.rs.push(block)])[-window:]
            now = time.time()
            if now - last_pred < 0.35 or len(buf) < window:
                continue
            last_pred = now
            lvl = dbfs(buf)
            parts = []
            for name, inf in candidates.items():
                key, prob = predict(inf, buf)
                col = C_OK if (prob >= taus[name] and key not in NON_COMMAND) else C_DIM
                parts.append(f"{name}:{col}{key}{C_OFF}={prob:.2f}")
            print(f"\r  t={now - t0:5.1f}s  mic {lvl:5.1f} dBFS  " + "   ".join(parts) + "   ",
                  end="", flush=True)
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        mic.close()
        if json_out:
            json_out.write_text(json.dumps({"mode": "live", "taus": taus, "window_s": 3.0}, indent=2))
    return 0


def _await_enter(prompt: str) -> str:
    if sys.stdin.isatty():
        try:
            return input(prompt)
        except EOFError:
            pass
    print(prompt.rstrip() + "  (auto-continuing)")
    return ""


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="tiny-VCM command test bench",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="A2,A7,B2", help="candidates (default A2,A7,B2)")
    p.add_argument("--tau", type=float, default=None,
                   help="one threshold for all models (default: per model, as in the app: "
                        "B2 0.80, A7 0.70, A2 0.70)")
    p.add_argument("--device", default=None)
    p.add_argument("--take-seconds", type=float, default=DEFAULT_WINDOW)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--json", dest="json_out", type=Path, default=None)

    p.add_argument("--list", action="store_true", help="list input devices")
    p.add_argument("--probe", type=float, default=None, metavar="SEC")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--takes", type=int, nargs="?", const=0, default=None, metavar="N",
                   help="guided test; N omitted = all commands in scope")
    p.add_argument("--scope", choices=["leaf", "intent"], default="leaf")
    p.add_argument("--commands", default=None, help="comma-separated subset (leaf or intent keys)")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--save-takes", action="store_true",
                   help="save each take + manifest.csv + report.json under takes/<timestamp>/")
    p.add_argument("--wav", type=Path, nargs="+", default=None)
    p.add_argument("--wav-dir", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None, help="CSV manifest to grade --wav-dir")
    p.add_argument("--expect", default=None, help="expected class key for --wav")
    p.add_argument("--live", action="store_true")
    p.add_argument("--seconds", type=float, default=None, help="stop --live after N seconds")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    if a.list:
        list_devices()
        return 0
    if a.probe is not None:
        return mode_probe(a.probe, a.json_out)

    which = [s.strip() for s in a.models.split(",") if s.strip()]
    json_out = a.json_out if (a.json_out is None or a.json_out.is_absolute()) else Path.cwd() / a.json_out
    candidates = load_candidates(which)
    taus = {n: (a.tau if a.tau is not None else DEFAULT_TAU.get(n, 0.70)) for n in which}

    if a.selftest:
        return mode_selftest(candidates, json_out)
    if a.wav_dir is not None:
        return mode_wav_dir(candidates, a.wav_dir, a.manifest, taus, a.topk, json_out)
    if a.wav:
        return mode_wav(candidates, a.wav, a.expect, taus, a.topk, json_out)
    if a.live:
        return mode_live(candidates, taus, a.device, a.seconds, json_out)

    # takes mode (explicitly requested via --takes or --commands)
    if a.takes is None and a.commands is None:
        print(__doc__)
        print("pick a mode: --selftest / --takes / --wav / --wav-dir / --live / --list / --probe")
        return 0

    leaves, intents = build_commands()
    if a.commands:
        keys = {k.strip() for k in a.commands.split(",") if k.strip()}
        pool = [c for c in leaves if c.leaf_key in keys or c.intent_key in keys]
        found = {c.leaf_key for c in pool} | {c.intent_key for c in pool}
        missing = keys - found
        if missing:
            raise SystemExit(f"{C_BAD}ERROR{C_OFF} unknown command(s): {', '.join(sorted(missing))}")
    elif a.scope == "intent":
        pool = intents
    else:
        pool = leaves
    if a.shuffle:
        random.shuffle(pool)
    if a.takes:
        pool = pool[:a.takes]
    if not pool:
        raise SystemExit(f"{C_BAD}ERROR{C_OFF} nothing to say (empty command set)")
    save_dir = None
    if a.save_takes:
        from datetime import datetime

        save_dir = ROOT / "takes" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        print(f"  takes will be saved to {save_dir.relative_to(ROOT)}/")
    return mode_takes(candidates, pool, taus, a.take_seconds, a.device, save_dir, json_out)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130)
