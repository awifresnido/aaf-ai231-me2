#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

# Public source pool (FSC + GSC) -> canonical Option B manifests
if [[ ! -f data/manifests/fsc.csv ]]; then
  python3 scripts/adapt_fsc.py
fi
if [[ ! -f data/manifests/gsc_v2.csv ]]; then
  python3 scripts/adapt_gsc.py
fi
if [[ ! -f data/manifests/gsc_noise.csv ]]; then
  python3 scripts/slice_gsc_noise.py
fi

# Option B synthetic set: clean clips only by default (see source registry)
if [[ -d data/external/mark_optionb/MEX2/OptionB ]]; then
  python3 scripts/adapt_mark_optionb.py
fi

merge_inputs=()
for manifest in fsc gsc_v2 gsc_noise mark_optionb; do
  [[ -f "data/manifests/${manifest}.csv" ]] && merge_inputs+=("data/manifests/${manifest}.csv")
done

python3 scripts/merge_manifests.py "${merge_inputs[@]}" \
  --output data/manifests/public_base_v0.csv

python3 scripts/report_manifest.py \
  data/manifests/public_base_v0.csv \
  --output reports/public_base_v0.md

# Compact dry-run subset
python3 scripts/make_dryrun_manifest.py
python3 scripts/validate_manifest.py \
  data/manifests/dryrun_public_v0.csv \
  --project-root . --check-speaker-split
python3 scripts/report_manifest.py \
  data/manifests/dryrun_public_v0.csv \
  --output reports/dryrun_public_v0.md

# Personal recordings (present only after the recorder has been run and uploaded)
if compgen -G 'data/personal/raw/*/*/recordings.csv' >/dev/null; then
  split_args=()
  if [[ -f configs/personal_session_splits.csv ]]; then
    split_args=(--session-splits configs/personal_session_splits.csv)
  fi
  python3 scripts/adapt_personal.py \
    --root data/personal/raw \
    --augmented-root data/processed/personal_awi_augmented \
    "${split_args[@]}" \
    --output data/manifests/personal_awi.csv

  python3 scripts/merge_manifests.py \
    data/manifests/dryrun_public_v0.csv \
    data/manifests/personal_awi.csv \
    --output data/manifests/dryrun_composite_v0.csv

  python3 scripts/validate_manifest.py \
    data/manifests/dryrun_composite_v0.csv \
    --project-root . --check-speaker-split

  python3 scripts/report_manifest.py \
    data/manifests/dryrun_composite_v0.csv \
    --output reports/dryrun_composite_v0.md
fi

# Class pool, once the shared Drive folder is synced
if [[ -d data/external/class_pool_raw ]]; then
  python3 scripts/adapt_class_pool.py \
    --raw-dir data/external/class_pool_raw \
    --output data/manifests/class_pool.csv
fi
