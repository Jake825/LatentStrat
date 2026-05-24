---
tags:
  - latentstrat
  - cli-reference
aliases:
  - "LatentStrat CLI Reference"
  - "Command Reference"
related:
  - "[[rebuild-from-scratch]]"
  - "[[training-and-validation]]"
  - "[[current-state]]"
---

# CLI Reference

This page explains the current `latentstrat` command surface. For the exact
runtime help, run:

```bash
latentstrat --help
```

Examples use current V5.6.4/V5.8 paths where possible. Older artifact names in
local folders are historical.

## Primary Pipeline Commands

### `build-prior-features`

Builds the transductive prior feature table for team numbers `0..max_team_number`.
It pulls TBA team history, Statbotics normalized EPA trajectory, and OpenAI
narrative embeddings.

```bash
latentstrat build-prior-features \
  --target-season 2026 \
  --max-team-number 12500 \
  --output data/prior_features_v563_2026.parquet
```

Use cached embeddings whenever possible. Rebuild when narrative logic, EPA
extraction, culture targets, or team cap changes.

### `train-prior`

Trains the sacrificial prior distiller and writes a stripped checkpoint
containing only `embedding_table`, metadata, and history.

```bash
latentstrat train-prior \
  --features data/prior_features_v563_2026.parquet \
  --output artifacts/prior_v564_latent16 \
  --epochs 1000 \
  --latent-dim 16 \
  --max-team-number 12500 \
  --tensorboard \
  --tensorboard-run-name v564_prior_latent16_2026
```

V5.6.4 is a training-loss update over the V5.6.3 prior feature schema, so this
command can legitimately use `prior_features_v563_2026.parquet`.

### `inspect-prior`

Writes prior latent-space diagnostics such as PCA plots, norm histograms,
nearest neighbors, sanity checks, and copied training history.

```bash
latentstrat inspect-prior \
  --checkpoint artifacts/prior_v564_latent16/checkpoint.pt \
  --features data/prior_features_v563_2026.parquet \
  --output artifacts/prior_v564_latent16/inspection
```

### `build-features`

Builds the match-grain Parquet feature table from TBA and optional scouting
data. With `--sidecar-output-dir`, it also writes rankings, selections, and
playoffs sidecars.

```bash
latentstrat build-features \
  --season 2026 \
  --output data/features_v58_2026.parquet \
  --sidecar-output-dir data/v58_sidecars_2026
```

For a single event:

```bash
latentstrat build-features \
  --event-key 2026ilch \
  --output data/features_2026ilch.parquet
```

### `train-features`

Trains the Set Transformer from a feature Parquet file. It can load a stripped
prior checkpoint into `Z_base` and optional sidecars for heterogeneous training.

```bash
latentstrat train-features data/features_v58_2026.parquet \
  --output artifacts/v58_full_season_common_metrics_100 \
  --epochs 100 \
  --mini-batch-size 256 \
  --prior-checkpoint artifacts/prior_v564_latent16/checkpoint.pt \
  --rankings-sidecar data/v58_sidecars_2026/rankings_2026.parquet \
  --selections-sidecar data/v58_sidecars_2026/selections_2026.parquet \
  --playoffs-sidecar data/v58_sidecars_2026/playoffs_2026.parquet \
  --no-early-stopping \
  --restore-best \
  --tensorboard
```

### `validate-walk-forward`

Runs V5.8 temporal validation. Each fold trains on weeks `<= N`, validates on
week `N + 1`, and resets from the prior checkpoint before the next fold.

```bash
latentstrat validate-walk-forward \
  --features data/features_v58_2026.parquet \
  --prior-checkpoint artifacts/prior_v564_latent16/checkpoint.pt \
  --rankings-sidecar data/v58_sidecars_2026/rankings_2026.parquet \
  --selections-sidecar data/v58_sidecars_2026/selections_2026.parquet \
  --playoffs-sidecar data/v58_sidecars_2026/playoffs_2026.parquet \
  --output artifacts/v58_walk_forward_v564_latent16_50ep_2026 \
  --epochs 50 \
  --mini-batch-size 256 \
  --latent-dim 16 \
  --no-save-fold-checkpoints \
  --tensorboard \
  --tensorboard-run-name v58_walk_forward_v564_latent16_50ep_2026
```

## Experiment And Diagnostic Commands

### `run-prior-grid`

Runs the latent-dimension elbow experiment over the production prior distiller.

```bash
latentstrat run-prior-grid \
  --features data/prior_features_v563_2026.parquet \
  --output artifacts/prior_elbow_grid \
  --epochs 500 \
  --latent-dims 2,4,8,16,32,64,128,256
```

### `inspect-embeddings`

Runs a small training/inspection path and writes embedding diagnostics. Use it
for exploratory embedding tables, PCA, neighbor checks, attention, and zero-out
diagnostics. For the prior-only coordinate map, prefer `inspect-prior`.

```bash
latentstrat inspect-embeddings --event-key 2026ilch
```

### `build-evidence-packet`

Runs evidence experiments and writes a scoreboard/diagnostic packet. This is
where shuffled controls, null controls, baseline comparisons, and embedding
diagnostics are grouped for review.

```bash
latentstrat build-evidence-packet --event-key 2026ilch
```

### `full-season-offline`

Convenience command that imports a season, trains the model, and writes core
artifacts. Use it for rough offline checks. For repeatable research runs, prefer
explicit `build-features`, `train-features`, and `validate-walk-forward`
commands.

```bash
latentstrat full-season-offline --season 2026
```

## Venue And Consolidation Commands

### `train-features --venue-mode`

Venue mode fine-tunes selected parts of a checkpoint for a specific event. It is
intended for live-event adaptation experiments, not season-level validation.

```bash
latentstrat train-features data/features_2026ilch.parquet \
  --venue-mode \
  --event-key 2026ilch \
  --checkpoint artifacts/features_run/v5_checkpoint.pt \
  --output artifacts/venue_2026ilch
```

### `consolidate-event`

Folds event deltas back into durable base embeddings using the offline
consolidation gate.

```bash
latentstrat consolidate-event \
  artifacts/features_run/v5_checkpoint.pt \
  --event-key 2026ilch \
  --embedding-db data/team_embeddings.db \
  --delta-weeks 1.0
```

Consolidation is an offline step. It is not part of the main V5.8
walk-forward metric path.

## Setup, Smoke, And Cache Commands

### `api-smoke`

Checks TBA API connectivity.

```bash
latentstrat api-smoke
```

### `smoke-test`

Runs a single-event smoke training path. Use this after setup changes.

```bash
latentstrat smoke-test --event-key 2026ilch
```

### `init-scouting-db`

Creates the local SQLite scouting database schema.

```bash
latentstrat init-scouting-db --path data/scouting.db
```

### `clear-cache`

Removes Python provider caches. Use this when cached TBA or Statbotics payloads
are stale or when debugging provider behavior.

```bash
latentstrat clear-cache
```

## TensorBoard

Training commands should support local TensorBoard logging:

```bash
tensorboard --logdir=runs
```

Use `--no-tensorboard` for quiet batch or test runs when the command exposes
that option.
