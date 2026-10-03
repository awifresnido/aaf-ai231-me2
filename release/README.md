# release/ — GitHub Release assets

This directory is **git-ignored** (`.gitignore` keeps only this README). The
actual release assets are attached to the GitHub Release tagged `v0` and are
downloaded by `reproduce.sh` (stage `weights`), then verified against
`results/me2_gold/checksums.sha256`.

## Assets for `v0`

| Asset | Destination after download |
|---|---|
| `B2_s0` … `G2_s2` (9 v1 checkpoints) | `checkpoints/<id>/best.pt` |
| `B2f_s0` … `G2f_s2` (9 fine-tuned checkpoints) | `checkpoints/<id>/best.pt` |
| `E1f_s1_onnx` | `exports/me2_gold/E1f_s1/model.onnx` |
| `B2f_s0_onnx` | `exports/me2_gold/B2f_s0/model.onnx` |
| `personal_awi.zip` | `data/personal/raw/202453069/` (Awi's s01/s02/s03) |

## Creating the release (Awi, after review)

```bash
# on the DGX, from the training repo (read-only source of truth):
sha256sum checkpoints/*/best.pt exports/me2_gold/*/model.onnx > results/me2_gold/checksums.sha256
# zip the personal recordings:
zip -r personal_awi.zip data/personal/raw/202453069
# attach all of the above (plus personal_awi.zip) to the GitHub Release "v0"
```

The `checksums.sha256` file is committed to git so `reproduce.sh` can verify
every downloaded asset.
