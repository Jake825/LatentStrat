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

Examples use the V5.6.4 prior, tagged V5.8 baseline, and active V6-Lite paths where possible. Older artifact names in local folders are historical.

## Default Local Storage Layout

When paths are omitted, CLI defaults use the canonical generated-file layout: `data/cache/` for TBA, Statbotics, and OpenAI SQLite caches; `data/world_model/match_breakdowns.sqlite` for the durable raw match-breakdown corpus; `data/scouting/scouting.db` for scouting; `data/embeddings/latentstrat_embeddings.sqlite` for durable consolidated embeddings; `data/features/prior/`, `data/features/season/`, `data/features/event/`, and `data/features/world_model/` for Parquet feature tables; `data/sidecars/` for relational sidecars; grouped `artifacts/` subdirectories for model outputs; and `runs/` for TensorBoard. Explicit older paths still work when provided.

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
  --sidecar-output-dir data/sidecars/v58_2026 \
  --event-metadata-output data/features/season/events_2026.parquet
```

For a single event:

```bash
latentstrat build-features \
  --event-key 2026ilch \
  --output data/features/event/features_2026ilch.parquet
```

### `train-features`

Trains the V6 Set Transformer from a feature Parquet file. It can load a stripped prior checkpoint into `Z_base`, optional sidecars for heterogeneous training, and a frozen target bundle with `--world-model-bundle`. Omitting the bundle is the explicit supervised-only V6 ablation. Feature runs write `v6_checkpoint.pt`.

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
Resuming an active frozen-target V6 checkpoint also requires its matching `--world-model-bundle`;
bundle widths must match the checkpoint architecture.

### `build-world-model`

Builds award, rank, and pick V6-Lite target-space scaffolds offline. Historical score archetypes
use the dedicated match-breakdown commands below. Score configuration is reserved but fails
clearly if enabled before per-alliance runtime attachment exists.

### `sync-match-breakdowns`

Synchronizes raw historical TBA event and match payloads into durable SQLite storage. It defaults
to event types `0..5`; use `--include-foc` or `--include-remote` only for explicit experiments.

```powershell
latentstrat sync-match-breakdowns `
  --start-season 2015 `
  --end-season 2026 `
  --corpus-db data/world_model/match_breakdowns.sqlite
```

Use `--refresh` to send conditional requests with stored ETags.

### `train-match-breakdown-encoder`

Trains season-specific encoders and decoders around one shared 16D bottleneck and writes the
offline historical score artifact:

```powershell
latentstrat train-match-breakdown-encoder `
  --start-season 2015 `
  --end-season 2026 `
  --epochs 50 `
  --seasons-per-step 4 `
  --rows-per-season 64 `
  --learning-rate 0.001 `
  --seed 2026
```

### `inspect-match-breakdown-encoder`

Writes a standalone static inspection report for the frozen historical alliance embeddings. This
does not retrain the encoder or modify its bundle:

```powershell
latentstrat inspect-match-breakdown-encoder `
  --artifact-dir artifacts/world_model/match_breakdown/v1_2015_2026 `
  --output artifacts/world_model/match_breakdown/v1_2015_2026/inspection `
  --seed 2026 `
  --tsne-max-rows 11000 `
  --network-node-limit 400 `
  --neighbors-per-node 2
```

The report contains deterministic PCA and t-SNE views, parallel-coordinate archetype profiles, and
a PCA-positioned cross-season neighbor network. Projection plots are interpretation aids, not
standalone promotion evidence.

### `validate-walk-forward`

Runs fold-safe temporal validation. Each fold trains on weeks `<= N`, validates on week `N + 1`, and resets from the prior checkpoint. The historical score artifact is not attached to this path yet.

```bash
latentstrat validate-walk-forward \
  --features data/features/season/features_v58_2026.parquet \
  --prior-checkpoint artifacts/prior/prior_v564_latent16/checkpoint.pt \
  --rankings-sidecar data/sidecars/v58_2026/rankings_2026.parquet \
  --selections-sidecar data/sidecars/v58_2026/selections_2026.parquet \
  --playoffs-sidecar data/sidecars/v58_2026/playoffs_2026.parquet \
  --output artifacts/walk-forward/v58_walk_forward_2026 \
  --epochs 50 \
  --mini-batch-size 256 \
  --latent-dim 16 \
  --no-save-fold-checkpoints \
  --tensorboard \
  --tensorboard-run-name v58_walk_forward_2026
```

### `compare-world-model`

Runs deterministic paired bootstrap non-inferiority checks over fold-match prediction exports:

```bash
latentstrat compare-world-model \
  --baseline artifacts/baselines/v5.8/walk-forward/walk_forward_predictions.parquet \
  --candidate artifacts/walk-forward/v6_score_attached/walk_forward_predictions.parquet \
  --output artifacts/walk-forward/v6_score_attached/promotion_vs_v58.csv
```

### `write-baseline-manifest`

Writes the tracked SHA-256 manifest for the ignored V5.8 archive:

```bash
latentstrat write-baseline-manifest artifacts/baselines/v5.8 \
  --output baselines/v5.8-baseline.json \
  --tag v5.8-baseline
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
  --checkpoint artifacts/season/features_run/v6_checkpoint.pt \
  --output artifacts/season/venue_2026ilch
```

### `consolidate-event`

Folds event deltas back into durable base embeddings using the offline consolidation gate.

```bash
latentstrat consolidate-event \
  artifacts/season/features_run/v6_checkpoint.pt \
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

Removes Python provider caches under `data/cache/` and legacy transition caches such as `tba_cache.sqlite` and `statbotics_offline_cache/`. It does not remove the durable `data/world_model/match_breakdowns.sqlite` corpus, feature tables, scouting databases, artifacts, or TensorBoard runs. Use this when cached TBA, Statbotics, or OpenAI embedding payloads are stale or when debugging provider behavior.

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
