# Deployment — Pi 4 edge architecture

## Architecture

```
microphone ──► wake word ──► VCM (ONNX, 1 thread) ──► intent + slot
                     ▲
              VAD plugin (disabled by default)
```

- **Model:** single-file ONNX (opset 17), `exports/me2_gold/E1f_s1/model.onnx`
  (Focus, CRNN-Attn, τ = 0.95); secondary `B2f_s0`.
- **Runtime:** onnxruntime 1.30.0, `intra_op_num_threads = 1`.
- **Wake model:** `hey_rhasspy`, threshold 0.5.
- **VAD endpointing:** Silero/energy plugin, wired and tested, **`enabled: false`**
  (the §5 sweep found no hangover at ≤ 1 % truncation, so it stays off).
- **Benchmark mode:** music + chime off; gated by `edge_service.benchmark_mode`.

## Class benchmark compliance

The edge emits the benchmark's log-line shape
(`{"event":"wake"}` + `{"intent","slot","infer_ms","audio_ms"}`), which parses
with the benchmark's own `LineParser`.

`scripts/bench_pi.sh` reproduces live run `20261003-005002` (full, seed 15206,
E1f_s1) through the pinned `bench/vcm-benchmark` submodule:

```bash
./scripts/bench_pi.sh <user@host> --model exports/me2_gold/E1f_s1/model.onnx \
    --seed 15206 --run-id 20261003-005002
```

The committed run's `metrics.json` is under
`results/me2_gold/live/20261003-005002/`.

## Live run `20261003-005002` (full, 218 inputs, E1f_s1)

| Metric | Value |
|---|---|
| intent accuracy | 80.7 % [75–86 %] |
| command accuracy | 80.7 % |
| false accept | 6.2 % (1/16) |
| false reject | 20.4 % |
| false wake | 0.0 % (0/16) |
| slot exact | 100 % |
| wake detect | 92.1 % |
| latency p50 / p95 / p99 | 2.38 / 3.18 / 3.46 s |
| inference p95 | 38.9 ms |
| real-time factor | 0.008 (p95 0.010) |
| CPU temp max / RAM peak | 58.4 °C / 466.8 MB |

## Pi setup

- OS: Raspberry Pi OS (64-bit), Python 3.11
- onnxruntime 1.30.0, 1 thread
- wake model `hey_rhasspy` threshold 0.5, benchmark mode on
- `./reproduce.sh --pi <user@host>` deploys the edge service and runs the
  benchmark end-to-end
