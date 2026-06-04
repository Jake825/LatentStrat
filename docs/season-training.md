# Season Training

Season training is the supported supervised Set Transformer workflow.

## Build Features

```powershell
latentstrat season build-features `
  --season 2026 `
  --output data/features/season/features_2026.parquet `
  --sidecar-output-dir data/sidecars/v58_2026 `
  --event-metadata-output data/features/season/events_2026.parquet
```

The reusable Parquet file is the training boundary. Optional scouting rows are merged while the
table is built; training reads only the Parquet file.

## Train

```powershell
latentstrat season train data/features/season/features_2026.parquet `
  --prior-checkpoint artifacts/pretraining/prior/prior_run/checkpoint.pt `
  --rankings-sidecar data/sidecars/v58_2026/rankings_2026.parquet `
  --selections-sidecar data/sidecars/v58_2026/selections_2026.parquet `
  --playoffs-sidecar data/sidecars/v58_2026/playoffs_2026.parquet `
  --output artifacts/season/features_run
```

The run writes `checkpoint.pt`, CSV diagnostics, PNG plots, and `manifest.json`. TensorBoard writes
to `runs/` by default; use `--no-tensorboard` for quiet automation.

## Validate

```powershell
latentstrat season validate `
  --features data/features/season/features_2026.parquet `
  --prior-checkpoint artifacts/pretraining/prior/prior_run/checkpoint.pt `
  --output artifacts/walk-forward/v58_walk_forward_2026
```

Walk-forward validation is the primary temporal evaluation path. It trains only on canonical weeks
known before each held-out week and writes paired fold-match predictions for comparison.

## Deferred Integrations

The historical match-breakdown artifacts remain offline-only. Runtime score auxiliary attachment is
reserved for a later promotion experiment:

```text
z_red  -> ScoreEmbeddingPredictor -> frozen_red_16d
z_blue -> ScoreEmbeddingPredictor -> frozen_blue_16d
```
