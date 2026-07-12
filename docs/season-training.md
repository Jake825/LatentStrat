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

## CPU Training Runtime

Season training defaults to CPU, AdamW, gradient clipping at `1.0`, and an optimizer-step cosine schedule ending at `1e-5`. The same runtime drives walk-forward folds. Use `--gradient-accumulation-steps` to increase the effective batch size, or select `--scheduler none|cosine|one-cycle` explicitly.

Every CLI run writes an epoch-boundary resume checkpoint at `resume/season/latest.ckpt`. Continue an interrupted run with the same data and mathematical configuration:

```powershell
latentstrat season train data/features/season/features_2026.parquet `
  --resume-checkpoint artifacts/season/features_run/resume/season/latest.ckpt `
  --output artifacts/season/features_run
```

`--resume-checkpoint` restores model, optimizer, scheduler, RNG, loader, history, and early-stopping state. It cannot be combined with `--checkpoint` or `--prior-checkpoint`, which remain weights-only warm starts. Walk-forward folds resume automatically from their fold-local checkpoints under the selected output directory.

## Validate

```powershell
latentstrat season validate `
  --features data/features/season/features_2026.parquet `
  --prior-checkpoint artifacts/pretraining/prior/prior_run/checkpoint.pt `
  --output artifacts/walk-forward/v58_walk_forward_2026
```

Walk-forward validation is the primary temporal evaluation path. It trains only on canonical weeks
known before each held-out week and writes paired fold-match predictions for comparison.

## Static 2026 Reference

The static study is a separate schema-`7` path; it does not change the schema-`6` production
default. It removes `Z_event`, compares additive, teammate-set, and full-match architectures, and
uses nested selection/refit folds. TensorBoard records per-task train/validation loss and activity,
score and probability metrics, ten-bin ECE, optimizer-group rates, clipping and runtime, parameter
utilization, and bounded `Z_base`/shared-module histograms. See the
[reference-study contract](reference-2026-static-study.md).

The [static architecture reliability study](reference-2026-static-reliability.md) is a separate
smoke-first command. It freezes optimization on development week 4, then uses a fixed three-seed
weeks 6/8/10 matrix. Its TensorBoard root is `runs/reference-2026-static-reliability/`; a two-epoch
timing preflight refuses the full grid when even a conservative estimate exceeds the CPU budget.

## Deferred Integrations

The historical match-breakdown artifacts remain offline-only. Runtime score auxiliary attachment is
reserved for a later promotion experiment:

```text
z_red  -> ScoreEmbeddingPredictor -> frozen_red_16d
z_blue -> ScoreEmbeddingPredictor -> frozen_blue_16d
```
