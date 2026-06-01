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

This page explains the current `latentstrat` command surface. For the exact runtime help, run:

```bash
latentstrat --help
```

Examples use current V5.6.4/V5.8 paths where possible. Older artifact names in local folders are historical.

## Default Local Storage Layout

When paths are omitted, CLI defaults use the canonical generated-file layout: `data/cache/` for TBA, Statbotics, and OpenAI SQLite caches; `data/scouting/scouting.db` for scouting; `data/embeddings/latentstrat_embeddings.sqlite` for durable consolidated embeddings; `data/features/prior/`, `data/features/season/`, and `data/features/event/` for Parquet feature tables; `data/sidecars/` for relational sidecars; grouped `artifacts/` subdirectories for model outputs; and `runs/` for TensorBoard. Explicit older paths still work when provided.

## Primary Pipeline Commands

### `build-prior-features`

Builds the transductive prior feature table for team numbers `0..max_team_number`. It pulls TBA team history, Statbotics normalized EPA trajectory, and OpenAI narrative embeddings.

```bash
latentstrat build-prior-features \
  --target-season 2026 \
  --max-team-number 12500 \
  --output data/features/prior/prior_features_v563_2026.parquet
```

Use cached embeddings whenever possible. Rebuild when narrative logic, EPA extraction, culture targets, or team cap changes.

### `train-prior`

Trains the sacrificial prior distiller and writes a stripped checkpoint containing only `embedding_table`, metadata, and history.

```bash
latentstrat train-prior \
  --features data/features/prior/prior_features_v563_2026.parquet \
  --output artifacts/prior/prior_v564_latent16 \
  --epochs 1000 \
  --latent-dim 16 \
  --max-team-number 12500 \
  --tensorboard \
  --tensorboard-run-name v564_prior_latent16_2026
```

V5.6.4 is a training-loss update over the V5.6.3 prior feature schema, so this command can legitimately use `prior_features_v563_2026.parquet`.

### `inspect-prior`

Writes prior latent-space diagnostics such as PCA plots, t-SNE team-galaxy views, latent/stat correlation heatmaps, norm histograms, nearest neighbors, sanity checks, and copied training history.

```bash
latentstrat inspect-prior \
  --checkpoint artifacts/prior/prior_v564_latent16/checkpoint.pt \
  --features data/features/prior/prior_features_v563_2026.parquet \
  --output artifacts/prior/prior_v564_latent16/inspection
```

### `build-features`

Builds the match-grain Parquet feature table from TBA and optional scouting data. With `--sidecar-output-dir`, it also writes rankings, selections, and playoffs sidecars.

```bash
latentstrat build-features \
  --season 2026 \
  --output data/features/season/features_v58_2026.parquet \
  --sidecar-output-dir data/sidecars/v58_2026
```

For a single event:

```bash
latentstrat build-features \
  --event-key 2026ilch \
  --output data/features/event/features_2026ilch.parquet
```

### `train-features`

Trains the Set Transformer from a feature Parquet file. It can load a stripped prior checkpoint into `Z_base` and optional sidecars for heterogeneous training. Feature runs also write static diagnostic visuals for PMA attention, zero-out sensitivity, task precision, embedding drift, event deltas, and selection value when the needed prior or sidecar inputs are available.

```bash
latentstrat train-features data/features/season/features_v58_2026.parquet \
  --output artifacts/season/v58_full_season_common_metrics_100 \
  --epochs 100 \
  --mini-batch-size 256 \
  --prior-checkpoint artifacts/prior/prior_v564_latent16/checkpoint.pt \
  --rankings-sidecar data/sidecars/v58_2026/rankings_2026.parquet \
  --selections-sidecar data/sidecars/v58_2026/selections_2026.parquet \
  --playoffs-sidecar data/sidecars/v58_2026/playoffs_2026.parquet \
  --no-early-stopping \
  --restore-best \
  --tensorboard
```

For frozen-prior transformer warmup, keep `Z_base` and `Z_event` fixed while training the Set Transformer and heads:

```bash
latentstrat train-features data/features/season/features_v58_2026.parquet \
  --output artifacts/season/frozen_prior_warmup \
  --prior-checkpoint artifacts/prior/prior_v564_latent16/checkpoint.pt \
  --split-policy stratified-event-comp \
  --freeze-team-embeddings \
  --tensorboard
```

`--split-policy` accepts `chronological-holdout`, `week-held-out`, `event-held-out`, or `stratified-event-comp`. `--freeze-team-embeddings` requires `--prior-checkpoint` or `--checkpoint` and cannot be combined with `--venue-mode`.

### `validate-walk-forward`

Runs V5.8 temporal validation. Each fold trains on weeks `<= N`, validates on week `N + 1`, and resets from the prior checkpoint before the next fold.

```bash
latentstrat validate-walk-forward \
  --features data/features/season/features_v58_2026.parquet \
  --prior-checkpoint artifacts/prior/prior_v564_latent16/checkpoint.pt \
  --rankings-sidecar data/sidecars/v58_2026/rankings_2026.parquet \
  --selections-sidecar data/sidecars/v58_2026/selections_2026.parquet \
  --playoffs-sidecar data/sidecars/v58_2026/playoffs_2026.parquet \
  --output artifacts/walk-forward/v58_walk_forward_v564_latent16_50ep_2026 \
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
  --features data/features/prior/prior_features_v563_2026.parquet \
  --output artifacts/prior-grid/prior_elbow_grid \
  --epochs 500 \
  --latent-dims 2,4,8,16,32,64,128,256
```

### `inspect-embeddings`

Runs a small training/inspection path and writes embedding diagnostics. Use it for exploratory embedding tables, PCA, neighbor checks, attention, and zero-out diagnostics. For the prior-only coordinate map, prefer `inspect-prior`.

```bash
latentstrat inspect-embeddings --event-key 2026ilch
```

### `build-evidence-packet`

Runs evidence experiments and writes a scoreboard/diagnostic packet. This is where shuffled controls, null controls, baseline comparisons, and embedding diagnostics are grouped for review.

```bash
latentstrat build-evidence-packet --event-key 2026ilch
```

### `full-season-offline`

Convenience command that imports a season, trains the model, and writes core artifacts. Use it for rough offline checks. For repeatable research runs, prefer explicit `build-features`, `train-features`, and `validate-walk-forward` commands.

```bash
latentstrat full-season-offline --season 2026
```

## Venue And Consolidation Commands

### `train-features --venue-mode`

Venue mode fine-tunes selected parts of a checkpoint for a specific event. It is intended for live-event adaptation experiments, not season-level validation.

```bash
latentstrat train-features data/features/event/features_2026ilch.parquet \
  --venue-mode \
  --event-key 2026ilch \
  --checkpoint artifacts/season/features_run/v5_checkpoint.pt \
  --output artifacts/season/venue_2026ilch
```

### `consolidate-event`

Folds event deltas back into durable base embeddings using the offline consolidation gate.

```bash
latentstrat consolidate-event \
  artifacts/season/features_run/v5_checkpoint.pt \
  --event-key 2026ilch \
  --embedding-db data/embeddings/latentstrat_embeddings.sqlite \
  --delta-weeks 1.0
```

Consolidation is an offline step. It is not part of the main V5.8 walk-forward metric path.

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
latentstrat init-scouting-db --path data/scouting/scouting.db
```

### `clear-cache`

Removes Python provider caches under `data/cache/` and legacy transition caches such as `tba_cache.sqlite` and `statbotics_offline_cache/`. It does not remove feature tables, scouting databases, artifacts, or TensorBoard runs. Use this when cached TBA, Statbotics, or OpenAI embedding payloads are stale or when debugging provider behavior.

```bash
latentstrat clear-cache
```

## Local Output Organizer

The helper script `scripts/organize_local_outputs.ps1` can dry-run moves from legacy generated paths into the canonical storage layout. It never deletes files and skips existing destinations.

```powershell
scripts/organize_local_outputs.ps1
scripts/organize_local_outputs.ps1 -Apply
```

## TensorBoard

Training commands should support local TensorBoard logging:

```bash
tensorboard --logdir=runs
```

Use `--no-tensorboard` for quiet batch or test runs when the command exposes that option.
