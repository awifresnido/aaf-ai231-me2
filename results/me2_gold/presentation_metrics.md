# ME2 Gold — presentation metrics

Every value traced to an artifact. Missing values are `null` with explicit reasons (never estimated).

- **Input:** 16 kHz mono 16-bit PCM WAV
- **Features:** log-mel 40 x 301 (3.0 s; n_fft=512, hop=160/10 ms, f_min=50, f_max=7600), per-mel CMVN
- **Crop mode:** `start` (GATE 2, carried from me2-master: energy crop helped quiet group recordings 201435283 but the pre-registered rule required BOTH ≥ 5 pt; kept 'start')

| Model | Role | Architecture | Params | ONNX | fp32 | Opset | Parity |
|---|---|---|---|---|---|---|---|
| **E1f_s1** | focus | CRNN-Attn | 69,234 | 279.2 KB | 0.27 MB | 17 | PASS |
| **B2f_s0** | sweep | TC-ResNet | 67,793 | 268.3 KB | 0.26 MB | 17 | PASS |
| G2f_s0 | benchmark | DS-CNN | 68,129 | *not exported* | — | — | — |

- **Head:** 33 classes (13 fixed intents + 6 slotted intents x 3 slot values (18 leaves) + UNKNOWN + SILENCE), label mode `leaf`

| Split | Clips | Hours | Unique speakers |
|---|---|---|---|
| train | 14,061 | 7.171 | 726 |
| validation | 4,815 | 2.626 | 201 |
| test | 324 | 0.179 | 6 |

- **Holdout:** test split = gold holdout 202 (84 real + 100 synthetic + 16 out-of-scope + 2 paraphrase) + Awi s03 122
- **Personal Awi audio:** 1,220 clips (train 976 = s01 repeat x4, val 122 = s02, test 122 = s03); sha256-verified copies, 0 augmented
- **Exclusions:** 90 relabelled group recordings (OUT_OF_SCOPE) + 2 other-command rows; all `supplemental_synth` excluded
- **Gates:** G1 True | G2 coverage pass | G3 UNKNOWN 18% / SILENCE 483 | G4 disjoint pass

- **Cluster:** ai-n003.ai-n003 — NVIDIA A100-SXM4-40GB
- **GPUs used:** [0, 1, 3] (9 runs: 3 GPUs x 3 concurrent)
- **Stack:** PyTorch 2.14.0+cu130 + CUDA 13.0
- **Objective:** weighted cross-entropy (nn.CrossEntropyLoss with sqrt-inverse class weights over 33 classes; no label smoothing)
- **Optimiser:** Adam, lr=0.0005, weight_decay=0.0001, betas=[0.9, 0.999], plateau schedule (patience 8)

| Run | Epochs | Steps | Best epoch | train loss | val loss |
|---|---|---|---|---|---|
| B2f_s0 | 22 | 4,840 | 14 | 0.3303 | 0.2184 |
| E1f_s1 | 9 | 1,980 | 1 | 0.2995 | 0.1511 |
| G2f_s0 | 14 | 3,080 | 6 | 0.8835 | 0.8910 |

- **Wall clock:** 112–175 s/run; 23.0 min total GPU-seconds across 9 runs; seeds [0, 1, 2]

- **Operating τ = 0.95** for all three models; per-seed wrong on the τ set ≤ 0.08% (all seeds pass condition 1). Unlike me2-master, no fine-tune collapsed.

me2-gold and v1 rates pooled over 3 seeds at τ=0.95 (from `eval_arch_*`).

| Slice (N) | B2f | v1 B2 | E1f | v1 E1 | G2f | v1 G2 |
|---|---|---|---|---|---|---|
| **validation_real** — group test spk 201435283 (537) | 63.3% | 61.5% | **65.7%** | 48.0% | 11.0% | 9.9% |
| **awi_s02** — Awi s02 (366) | 88.9% | **93.9%** | 91.4% | **94.3%** | 56.6% | **63.1%** |
| **awi_s03** — Awi s03 (366) | 92.1% | **95.7%** | 92.8% | **93.9%** | 69.2% | **70.6%** |
| **holdout_real** — unseen spk 202520785 (252) | 69.8% | 71.8% | **77.8%** | 73.8% | 30.6% | 29.4% |
| **holdout_syn** — holdout synthetic (300) | 100% | 99.0% | **99.7%** | 97.7% | 88.3% | 87.0% |
| **holdout_only** — full gold holdout (606) | 86.2% | 86.2% | **89.2%** | 86.4% | 61.3% | 60.2% |

| Negative slice (FAR) | B2f | v1 B2 | E1f | v1 E1 | G2f | v1 G2 |
|---|---|---|---|---|---|---|
| **near_miss** (off-schema, 158) | 21.5% | 18.6% | **15.0%** | 18.8% | 13.3% | 13.5% |
| **single_words_clean** (GSC, 10,730) | 1.18% | 4.22% | **0.58%** | 2.06% | 0.26% | 2.09% |
| **synthetic_neg_test** (250) | 2.53% | 4.53% | **2.27%** | 3.20% | 1.73% | 2.13% |

- **Rule:** a fine-tuned model replaces its v1 parent only if ALL four conditions hold (see JSON `decision_rule.rule`).
- **Verdicts:** all three `replace_v1 = false` — condition (2) "Awi s02 ≥ v1−1pt" fails for every model (B2f −5.0pt, E1f −2.9pt, G2f −6.5pt). Condition (3) passes only for E1f (+17.7pt on the group test speaker).
- **Deployment action:** the rule says **retain v1**, but **Awi chose E1f_s1 as the app default** (documented deviation) — the fine-tunes improve the group speaker and cut false-accepts at the cost of a small Awi regression the rule forbids.

- **Inference p95 (offline, deploy.json):** E1f_s1 1.29 ms model + 2.91 ms features = **4.2 ms**; B2f_s0 0.35 + 1.49 = **1.84 ms**
- **Runtime / threads:** onnxruntime 1.30.0, 1 thread
- **Live response latency p50/p95/p99:** 2.38 / 3.18 / 3.46 s · **inference p95** 38.9 ms · **RTF** mean 0.008 / p95 0.010 (live run `20261003-005002`)
- **Wake word:** openWakeWord `hey_rhasspy` (threshold 0.5, patience 2, debounce 1.5 s); **wake detect rate 92.1 %** (live)

## Live benchmark (`vcm-benchmark` full run `20261003-005002`)

218 inputs (202 with wake word + 16 false-wake) on the Pi, model **E1f_s1**, wake word `hey_rhasspy`, seed 15206.

| metric | value |
|---|---|
| intent accuracy (19) / command (93) | **80.7 %** [75–86 %] / 80.7 % |
| false accept (OOS fired) | 6.2 % (1/16) [1–28 %] |
| false reject (command ignored) | 20.4 % |
| false wake (no wake word) | 0.0 % (0/16) |
| misfire | 0.0 % |
| slot exact | 100.0 % (n=91) |
| wake detect rate / response rate | 92.1 % / 91.1 % (18 no-response) |
| latency p50 / p95 / p99 | 2.38 / 3.18 / 3.46 s |
| inference (Pi) mean / p95 | 31.4 / 38.9 ms |
| RTF mean / p95 | 0.008 / 0.010 |
| CPU temp max / RAM (process) peak | 58.4 °C / 466.8 MB |

Live intent accuracy is ~8.5 pt below the offline holdout (89.2 %) and false-reject is 20.4 % vs ~11 % offline — room noise + speaker distance + the fixed 4 s window. Response latency p95 (3.18 s) is dominated by that fixed window; see the VAD-endpointed rerun below (p95 0.74 s, −77 %).

## Live benchmark — VAD endpointing (`20261003-033915`)

Same 218 inputs + seed 15206 as `005002`, model **E1f_s1**, VAD endpointing **enabled** (Silero v4, hangover 500 ms, max_window 4 s).

| metric | value | vs fixed |
|---|---|---|
| intent accuracy (19) / command (93) | **68.3 %** [62–74 %] / 68.3 % | −12.4 pt |
| false accept (OOS fired) | 6.2 % (1/16) | 0 |
| false reject (command ignored) | **33.9 %** | +13.5 pt |
| false wake (no wake word) | 0.0 % (0/16) | 0 |
| misfire / slot exact | 0.0 % / 100.0 % (n=79) | — |
| wake detect / response rate | 80.2 % / 80.2 % (40 no-response) | −11.9 pt |
| latency p50 / p95 / p99 | **0.64 / 0.74 / 0.79 s** | −77 % |
| inference (Pi) mean / p95 | 21.8 / 32.3 ms | −6.6 ms |
| RTF mean / p95 | 0.010 / 0.017 | ↑ (smaller audio_ms) |
| CPU temp max / RAM peak | 61.3 °C / 477.6 MB | +2.9 °C |

VAD cuts latency **77 %** (p95 3.18 → 0.74 s) but on the benchmark's *played* commands the false-reject rate jumps to 33.9 % (40 no-response): the rejections are almost all `X → REJECT`, i.e. Silero misses onset or truncates the speaker→mic audio. Asymmetric — synthetic voice −17.9 pt vs real voice −6.3 pt. Live close-mic speech is confirmed working, so this is a recorded-audio artifact, not live behaviour.

- **Git:** `ea12e84` on `me2-gold`
- **Manifest:** `data/manifests/me2_gold_v1.csv` (sha256 `f81867392d20e3a23cbbcd2b77559c3b357e943f10e3095b65f7aa1b6171cdb2`)
- **Dataset revision:** `6947f13073e57eb6ae67e7e2fc3680700b82aa13`
- **Train:** `python scripts/launch_parallel.py --runs configs/runs_me2_gold.txt --gpus 0,1,3 --per-gpu 3 --waveform-cache data/cache/me2_gold_v1_wave16k`
- **Eval:** `python scripts/me2_gold_eval.py`
- **Export:** `python scripts/export_me2_gold.py`
