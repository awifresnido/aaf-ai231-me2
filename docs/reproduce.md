# Reproduction

## One command

```bash
git clone --recursive https://github.com/awifresnido/aaf-ai231-me2.git
cd aaf-ai231-me2
./reproduce.sh                                    # setup → data → weights → eval → export → app
./reproduce.sh --dry-run                          # print every command and URL, run nothing
```

## Stages

| Stage | Does | Verifies |
|---|---|---|
| `setup` | creates `.venv` from `env/` lock files; checks Python / ffmpeg / Node | versions printed |
| `data` | loads ME2 Gold from the DGX shared cache via `/data/ai231/load.py` | split sizes printed; revision `6947f13…` |
| `weights` | downloads release assets (18 checkpoints + 2 ONNX + personal audio) | `sha256sum -c results/me2_gold/checksums.sha256` |
| `manifest` | `training/scripts/adapt_me2_gold.py --extract` → `me2_gold_v1.csv` | manifest sha256 `f81867…` |
| `train` | `launch_parallel.py --runs configs/runs_me2_gold.txt --gpus <gpus>` | per-run summary JSON |
| `eval` | `me2_gold_eval.py --infer … --decide` | equals committed `eval_*.json` |
| `export` | single-file ONNX (opset 17) + PyTorch parity | parity PASS |
| `app` | laptop backend + React UI in mock-edge mode | health endpoint at `localhost:8080` |
| `bench` | `--pi`: deploy edge + run `bench/vcm-benchmark` | writes `live/<run-id>/metrics.json` |

## Retraining (DGX)

```bash
./reproduce.sh --train dgx --gpus 0,1,3
```

Reproduces the 9 fine-tune runs. Expected deltas vs the committed results are
small (GPU non-determinism); the tolerance is stated in the release notes.

## Tolerances & troubleshooting

- **eval** on released weights must equal the committed `eval_*.json` **exactly**;
  retrained weights are compared within the stated tolerance.
- **manifest** sha256 depends on the personal audio: with `--no-personal`, the
  result is expected to differ **on the Awi slices only**.
- **On the DGX** the dataset is read from `/data/ai231` (shared cache). Outside
  the DGX, download from Hugging Face:
  `hf download airimonda/ai231-me2-voice-commands --revision 6947f13073e57eb6ae67e7e2fc3680700b82aa13`.

## DOI

The dataset has a Hugging Face DOI: [10.57967/hf/10723](https://doi.org/10.57967/hf/10723).
