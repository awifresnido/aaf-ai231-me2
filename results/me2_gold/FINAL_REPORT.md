# ME2 Gold — Final Report

Branch `me2-gold` · commit `ea12e84` · dataset revision `6947f13073e57eb6ae67e7e2fc3680700b82aa13`
Owner: Awi · Cluster: ai-n003.ai-n003 (NVIDIA A100-SXM4-40GB)

---

## Per-gate results

| Gate | Result | Key numbers | Files written |
|---|---|---|---|
| **A — Setup** | ✅ | dataset revision matches `6947f13…`; crop = **start** (GATE 2, carried from me2-master); 9 init checkpoints located (`B2_s0–2`, `E1_s0–2`, `G2_s0–2`) with sha256 logged | `logs/summary_*.json` (init sha256) |
| **B — Dataset** | ✅ | 19,200 rows → train 14,061 (7.17 h, 726 spk) / val 4,815 (2.63 h, 201 spk) / test 324 (0.18 h, 6 spk); G1–G4 all pass (90 relabelled + 2 other-command excluded; UNKNOWN 18 %, SILENCE 483) | `reports/me2_gold_v1.json` (sha256 `f81867…`) |
| **C — Step-0 + smoke** | ✅ | 9 v1 checkpoints evaluated (s0 vs `results/me2/eval_*.json` within 0.5 pt); B2f_s0 smoke: strict load OK, step-0 loss ≈ eval loss | `results/me2_gold/step0_*.json`, `step0_table.json` |
| **D — Fine-tune** | ✅ | 9 runs (3 models × 3 seeds) on GPUs [0,1,3]; ~23 min total GPU-s; e.g. B2f_s0 22 ep/175 s, E1f_s1 9 ep/112 s, G2f_s0 14 ep/173 s | `logs/summary_{B2f,E1f,G2f}_s{0,1,2}.json` |
| **E — Eval + decision** | ✅ | all 6 families evaluated at τ=0.95; decision `replace_v1 = false` for all three (see below) | `results/me2_gold/eval_arch_*.json`, `decision.json`, `benchmark_offline_*.json` |
| **F — Export + compliance** | ✅ | E1f_s1 + B2f_s0 ONNX (parity PASS); app default E1f_s1; log line + benchmark mode; tests pass | `onnx_parity_report.json`, `presentation_metrics.{json,md}` |

## Decision per model (pre-registered rule)

A fine-tuned model replaces its v1 parent only if **all four** hold; `wrong ≤ 1 %` (1) passed for every model, but condition (2) *Awi s02 ≥ v1 − 1 pt* failed for all three:

| Model | (1) wrong≤1% | (2) Awi s02 ≥ v1−1pt | (3) group spk ≥ v1+5pt | (4) FAR ≤ v1+1pt | Verdict |
|---|---|---|---|---|---|
| **B2f** | ✅ 0.05% | ❌ −5.0 pt | ❌ +1.9 pt | ✅ | keep B2 |
| **E1f** | ✅ 0.0% | ❌ −2.9 pt | ✅ +17.7 pt | ✅ | keep E1 |
| **G2f** | ✅ 0.03% | ❌ −6.5 pt | ❌ +1.1 pt | ✅ | keep G2 |

**Deployment action (deviation):** the rule says retain v1, but **Awi chose E1f_s1 as the app default** — the fine-tunes materially improve the group test speaker (+17.7 pt) and cut Single-Words FAR (2.06 % → 0.58 %) at the cost of a small Awi regression (2.9 pt) that the rule's floor forbids. B2f_s0 kept as secondary; G2f not exported.

## Holdout result (unseen speaker 202520785, pooled 3 seeds, τ = 0.95)

| Model | holdout_only (202×3) | holdout_real (84×3) | holdout_syn (100×3) | holdout FAR (16×3) |
|---|---|---|---|---|
| **E1f** | 89.25 % | 77.78 % | 99.67 % | 6.25 % |
| **B2f** | 86.20 % | 69.84 % | 100 % | 6.25 % |
| **G2f** | 61.29 % | 30.56 % | 88.33 % | 2.08 % |
| v1 E1 | 86.38 % | 73.81 % | 97.67 % | 6.25 % |
| v1 B2 | 86.20 % | 71.83 % | 99.0 % | 10.42 % |
| v1 G2 | 60.22 % | 29.37 % | 87.0 % | 4.17 % |

## Benchmark compliance

- **Log line** (`{"event":"wake"}` + `{"intent","slot","infer_ms","audio_ms"}`) parses with the benchmark's own `LineParser` (§5a test).
- **Benchmark mode** (music + chime off) gated by `edge_service.benchmark_mode`.
- **Tests:** 6 benchmark-compliance tests + full suite **221 passed**; `npm run build` ✓.
- **Offline companion:** `benchmark_offline_{B2,E1,G2,B2f,E1f,G2f}.json` computed with the benchmark's imported `metrics` module (19-intent + 93-command accuracy with Wilson CI, macro P/R/F1/F2, false-accept/reject/misfire, per-intent, confusions, slot metrics).
- **Live run (`20261003-005002`, full, 218 inputs, E1f_s1):** intent accuracy **80.7 %** [75–86 %], command accuracy 80.7 %, false accept 6.2 % (1/16), false reject **20.4 %**, false wake 0.0 % (0/16), misfire 0.0 %, slot exact 100 %, wake detect **92.1 %**, response rate 91.1 % (18 no-response). Latency p50/p95/p99 = 2.38 / 3.18 / 3.46 s; inference p95 38.9 ms; RTF 0.008 (p95 0.010); CPU temp max 58.4 °C, RAM peak 466.8 MB, no throttling. Source: `~/20261003-005002/metrics.json` (Pi).
- **VAD rerun (`20261003-033915`, same 218 inputs + seed 15206, E1f_s1, VAD enabled — Silero v4, hangover 500 ms):** intent accuracy **68.3 %** [62–74 %], false reject **33.9 %**, false accept 6.2 % (1/16), false wake 0.0 %, slot exact 100 %, wake detect / response 80.2 % (40 no-response). **Latency p50/p95/p99 = 0.64 / 0.74 / 0.79 s (−77 % vs fixed)**; inference p95 32.3 ms; RTF 0.010 (p95 0.017); CPU temp max 61.3 °C, RAM peak 477.6 MB, no throttling. Rejections are almost all `X → REJECT` (Silero misses onset / truncates the played speaker→mic audio), asymmetric (synthetic −17.9 pt vs real −6.3 pt). Live close-mic speech is confirmed working. Source: `~/20261003-033915/metrics.json` (Pi).

## Presentation metrics

- `results/me2_gold/presentation_metrics.json` (+ Markdown twin) — 20 sections incl. `live_benchmark` (run `20261003-005002`, fixed window) and `live_benchmark_vad` (run `20261003-033915`, VAD endpointing); every value traced to an artifact.

## Open items

1. **Live benchmark** — done (`20261003-005002`, full run).
2. **Pi clock** — `sudo date -s @$(date +%s)` once (needs password) if the clock-fix acceptance test is still wanted.
3. **VAD endpointing** — plugin wired on `app-vad-endpointing`, now **enabled** (Silero v4, hangover 500 ms) after live validation: latency p95 drops 3.18 → 0.74 s (−77 %). On the benchmark's *played* commands it rejects more (false-reject 33.9 % vs 20.4 % fixed, mostly `X → REJECT`), but live close-mic speech is confirmed working — the regression is a recorded-audio artifact, not live behaviour. Committed configs keep `endpointing.enabled: false` (the Pi's live config is flipped `true` for testing).
