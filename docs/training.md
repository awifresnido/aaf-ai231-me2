# Training

## Pipeline

```
v1 pretraining (provenance) ──► fine-tune on ME2 Gold ──► eval + release gate ──► ONNX export
```

### v1 (provenance)

Three architectures were pretrained on the "Blend v1" composite (synthetic
Option-B TTS + FSC + GSC + SLURP + personal recordings), 3 seeds each:

- **B2** = TC-ResNet ("Sweep"), 67,793 params
- **E1** = CRNN-Attn ("Focus"), 69,234 params
- **G2** = DS-CNN ("Benchmark"), 68,129 params

These are the *starting* checkpoints for the fine-tune and the *baseline* for
the checklist's "comparable-size baseline" comparison.

### ME2 Gold fine-tune

9 runs (3 architectures × 3 seeds) starting from the matching v1 checkpoint
(strict load), on `me2_gold_v1` at revision
`6947f13073e57eb6ae67e7e2fc3680700b82aa13`. Every run:

```
--phase 2 --label-mode leaf --class-weighting sqrt_inverse \
--select-metric command_macro_f1 \
--train-manifest data/manifests/me2_gold_v1.csv \
--val-manifest data/manifests/me2_gold_v1.csv \
--max-duration 3.0 --crop-mode start \
--select-corpora gold_real personal_awi \
--batch-size 64 --grad-accum 1 --amp \
--dataset-name me2_gold_v1 --dataset-revision 6947f13073e57eb6ae67e7e2fc3680700b82aa13 \
--init-checkpoint checkpoints/<v1>/best.pt --seed {0,1,2} --run-name <family>_s{0,1,2}
```

Full argument list: `configs/runs_me2_gold.txt`. Launched with
`launch_parallel.py --runs configs/runs_me2_gold.txt --gpus 0,1,3 --per-gpu 1`.

## Cluster

| Field | Value |
|---|---|
| Node | `ai-n003` |
| GPU | NVIDIA A100-SXM4-40GB |
| GPUs used | 0, 1, 3 |
| Seeds | 0, 1, 2 |
| Per-run wall-clock | ~2–3 min (22 ep B2f_s0 175 s; 9 ep E1f_s1 112 s; 14 ep G2f_s0 173 s) |
| Total GPU-seconds | ~23 min across 9 runs |

Wall-clock and seeds are also recorded in `results/me2_gold/logs/summary_*.json`
(and `history_*.json` for the per-epoch curves).

## Audio front-end

40 log-mel bins, 512-point FFT, 10 ms hop, 3.0 s max duration (start-crop).
Waveform augmentation (noise mix, reverb, speed perturbation, gain, mic
band-limit) at train time. See `configs/train_config.yaml`.
