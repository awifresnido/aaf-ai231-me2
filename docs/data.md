# Data — sources, licence, access, splits

The repository contains **no audio**. Every clip is either read from the shared
DGX cache or downloaded from its source at build time. The primary dataset is
the class "ME2 Gold" set, published on Hugging Face as
`airimonda/ai231-me2-voice-commands`.

## Primary dataset — ME2 Gold

| Field | Value |
|---|---|
| Hugging Face repo | `airimonda/ai231-me2-voice-commands` |
| Revision | `6947f13073e57eb6ae67e7e2fc3680700b82aa13` |
| Configs | `default` (train / test / holdout / numerals), `supplemental_synth`, `synthetic_negatives` |
| Access (DGX) | read via `/data/ai231/load.py` (owner-provided loader; `cache_dir=/data/ai231`) |
| DOI | pending — see `CITATION.cff` / `.zenodo.json` |

`load.py` (the dataset owner's loader) uses the `datasets` library to open the
four splits and the two auxiliary configs from the shared cache:

```python
from datasets import load_dataset
ds = load_dataset("airimonda/ai231-me2-voice-commands", cache_dir="/data/ai231")
train, test, holdout = ds["train"], ds["test"], ds["holdout"]
numerals = ds["numerals"]
negs = load_dataset("airimonda/ai231-me2-voice-commands", "synthetic_negatives", cache_dir="/data/ai231")
```

## Per-source licensing

| Source | Licence / terms | Role |
|---|---|---|
| ME2 Gold (class) | dataset-card terms; group recordings are **linked, never re-hosted** | train / validation / holdout / negatives |
| Fluent Speech Commands (FSC) | FSC Public License — **non-commercial academic use** | v1 pretraining provenance |
| Google Speech Commands v0.02 | CC-BY-4.0 | clean Single-Words FAR slice |
| SLURP | CC-BY-4.0 text / CC-BY-NC-4.0 audio | v1 pretraining provenance |
| Awi's own recordings (s01/s02/s03) | published by the author as a release asset | train / validation / final check |

The fine-tuned weights are released under CC BY-NC 4.0 because training data
includes FSC (non-commercial). See `LICENSE-WEIGHTS.md`.

## Splits

The composite manifest `data/manifests/me2_gold_v1.csv` is built by
`training/scripts/adapt_me2_gold.py`:

- **train** = gold train (after exclusions) + Awi s01 ×4 + negatives
- **validation** = gold test + `synthetic_negatives` test + Awi s02
- **final test / holdout** = gold holdout + Awi s03

Exclusions (applied by the adapter): 90 relabelled group recordings
(`out-of-scope`), off-schema slot values (→ near-miss slice), `supplemental_synth`
in full, and the `listen_other_command.csv` rows.

Manifest sha256 (with personal audio included): `f81867392d20e3a23cbbcd2b77559c3b357e943f10e3095b65f7aa1b6171cdb2`.

## DOI route (owner decision)

- Derived artefacts (manifest, splits, exclusion list, checksums — no audio) are
  prepared for a Zenodo deposit via `.zenodo.json`.
- Whether the HF dataset itself receives a DOI is for the dataset owner; a
  short request message is drafted in `docs/reproduce.md#doi`.
