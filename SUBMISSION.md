# SUBMISSION.md — slide-2 table (AI231 ME2)

Filled from the repository's own files.

| Field | Value |
|---|---|
| GitHub repository | [awifresnido/aaf-ai231-me2](https://github.com/awifresnido/aaf-ai231-me2) — public, MIT |
| Dataset location | Hugging Face `airimonda/ai231-me2-voice-commands` @ `6947f13073e57eb6ae67e7e2fc3680700b82aa13` · DOI [10.57967/hf/10723](https://doi.org/10.57967/hf/10723) · access: per-source (FSC non-commercial; group audio linked, never re-hosted); on the DGX read via `/data/ai231/load.py` |
| A100 cluster | `ai-n003` · GPUs used `0,1,3` · total & per-run wall-clock (see `results/me2_gold/logs/summary_*.json`) · seeds `0, 1, 2` |
| Model weights | [`v0`](https://github.com/awifresnido/aaf-ai231-me2/releases/tag/v0) · CC BY-NC 4.0 · sha256 in `results/me2_gold/checksums.sha256` |

## Checklist evidence

| # | Item | Evidence | Verify with |
|---|---|---|---|
| 1 | Repo public, one-command reproduction | `./reproduce.sh`, `docs/reproduce.md` | `./reproduce.sh --dry-run` |
| 2 | Dataset licensed + citable | `docs/data.md`, `CITATION.cff`, `configs/data.yaml` | `cat configs/data.yaml` |
| 3 | Training logs + final checkpoint | `results/me2_gold/logs/`, GitHub Release + `checksums.sha256` | `sha256sum -c results/me2_gold/checksums.sha256` |
| 4 | Pi latency reproduced | `scripts/bench_pi.sh`, `results/me2_gold/live/20261003-005002/metrics.json` | `./scripts/bench_pi.sh --help` |
| 5 | Held-out test, unseen speakers | `docs/evaluation.md#holdout`, `results/me2_gold/eval_*.json` | `python -c "import json; print(json.load(open('results/me2_gold/decision.json'))['E1f']['numbers']['ft_holdout_only_correct'])"` |
| 6 | Baseline of comparable size | DS-CNN vs CRNN-Attn vs TC-ResNet (README) | `docs/figures/make_figures.py --check` |
