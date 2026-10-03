# Model packages

Each model the app can load is a folder `models/vcm/<id>/` containing a
`manifest.yaml` (committed). The edge scans this folder at start-up and when
you press **Rescan models folder** in the Engineering view.

## Registered now (see ../tiny-vcm/MODELS.md)

| id | phase | labels | threshold | role |
|---|---|---|---|---|
| `B2_s0` | 2 | leaf (33) | 0.80 (tau from phase 2) | final model, active by default |
| `A7_s0` | 1 | leaf (33) | 0.70 (app default) | TTS-only baseline, default challenger |
| `A2_s0` | 1 | intent (21) | 0.70 (app default) | TTS-only, no slots: slotted commands are never executed; compare it on intent accuracy |

In development the manifests reference the weights in place
(`../../../../tiny-vcm/checkpoints/<id>/best.pt`), so nothing is copied.
`pytest tests/test_torch_engine.py` test-loads every registered checkpoint.

## Adding a checkpoint

    python scripts/register_model.py --checkpoint ../tiny-vcm/checkpoints/B2_s1/best.pt \
      --id B2_s1 --name "Sweep, seed 1" --phase phase2 \
      --data "composite_v1 + wave aug" --min-confidence 0.80 --no-copy

Drop `--no-copy` when packaging for the Pi: the weights are then copied into
the folder as `model.pt` (gitignored) and the folder can be rsynced as is.
The class list is derived with tiny-vcm's own `create_label_mapping`; the
leaf order is ontology slot-value order (`20, 60, 100 percent`), not the
alphabetical listing shown in MODELS.md.
