---
tags:
  - latentstrat
  - rebuild-guide
aliases:
  - "Rebuild Guide"
  - "Rebuild LatentStrat"
related:
  - "[[cli-reference]]"
  - "[[current-state]]"
  - "[[schemas-and-artifacts-reference]]"
  - "[[training-and-validation]]"
---

# Rebuild LatentStrat From Scratch

This page is the practical rebuild checklist. It assumes a Windows PowerShell environment, but the commands are standard CLI commands where possible.

If you are rebuilding an older experiment state, first check the [Changelog](changelog.md) for the nearest Git commit and the [Experiment Ledger](experiment-ledger.md) for the artifact-era context.

## 1. Install

From the repo root:

```powershell
pip install -e ".[dev]"
```

Run a quick import and CLI check:

```powershell
latentstrat --help
latentstrat smoke-test
```

## 2. Configure Secrets

Create `.env` from the example:

```powershell
Copy-Item .env.example .env
notepad .env
```

Expected variables:

```text
TBA_API_KEY=your_tba_key_here
OPENAI_API_KEY=your_openai_key_here
```

Do not paste real keys into docs, screenshots, commits, or transcripts.

## Local Storage Layout

New default paths keep generated outputs grouped by purpose. Provider caches live under `data/cache/`, the durable raw match-breakdown corpus lives under `data/world_model/`, scouting lives under `data/scouting/scouting.db`, durable consolidated embeddings live under `data/embeddings/latentstrat_embeddings.sqlite`, feature tables live under `data/features/`, sidecars live under `data/sidecars/`, artifacts live under grouped `artifacts/` subdirectories, and TensorBoard runs live under `runs/`. Explicit legacy paths still work when passed on the CLI, but new rebuilds should use the canonical layout.

## 3. Build The Prior Feature Table

The prior feature table is the offline training set for Day Zero team identities.

```powershell
latentstrat build-prior-features `
  --target-season 2026 `
  --max-team-number 12500 `
  --output data/features/prior/prior_features_v564_2026.parquet
```

Expected output:

- One row per team number `0..12500`.
- OpenAI narrative vectors.
- Statbotics normalized EPA trajectory columns.
- Raw cultural target columns.
- Cached OpenAI embeddings under `data/cache/openai_embeddings.sqlite`.

If the command fails with zero observed normalized EPA values, inspect the Statbotics extraction path and rebuild. V5.6.4 training expects usable normalized EPA trajectory columns.

## 4. Train The Prior

```powershell
latentstrat train-prior `
  --features data/features/prior/prior_features_v564_2026.parquet `
  --output artifacts/prior/prior_v564_latent16 `
  --epochs 1000 `
  --latent-dim 16 `
  --max-team-number 12500 `
  --tensorboard `
  --tensorboard-run-name v564_prior_latent16_2026
```

View training:

```powershell
tensorboard --logdir=runs
```

Expected output:

```text
artifacts/prior/prior_v564_latent16/checkpoint.pt
```

The checkpoint should contain a stripped embedding table, not the sacrificial decoder.

## 5. Inspect The Prior

```powershell
latentstrat inspect-prior `
  --checkpoint artifacts/prior/prior_v564_latent16/checkpoint.pt `
  --features data/features/prior/prior_features_v564_2026.parquet `
  --output artifacts/prior/prior_v564_latent16/inspection
```

Review:

- `prior_sanity_checks.csv`
- `prior_nearest_neighbors.csv`
- `prior_latent_table.csv`
- PCA plots and norm histograms

## 6. Build Season Features And Sidecars

```powershell
latentstrat build-features `
  --season 2026 `
  --output data/features/season/features_v58_2026.parquet `
  --sidecar-output-dir data/sidecars/v58_2026 `
  --event-metadata-output data/features/season/events_2026.parquet
```

Expected outputs:

```text
data/features/season/features_v58_2026.parquet
data/sidecars/v58_2026/rankings_2026.parquet
data/sidecars/v58_2026/selections_2026.parquet
data/sidecars/v58_2026/playoffs_2026.parquet
data/features/season/events_2026.parquet
```

The feature table should include canonical `event_week` values for walk-forward validation.
Event metadata remains available for later fold-local V6 target integrations.

## 7. Build The V6-Lite Historical Score Artifact

```powershell
latentstrat sync-match-breakdowns `
  --start-season 2015 `
  --end-season 2026

latentstrat train-match-breakdown-encoder `
  --start-season 2015 `
  --end-season 2026 `
  --epochs 50 `
  --seasons-per-step 4 `
  --rows-per-season 64 `
  --learning-rate 0.001 `
  --seed 2026
```

Review `artifacts/world_model/match_breakdown/v1_2015_2026/validation_report.json`. It records grouped reconstruction losses, effective rank, nearest-neighbor sanity rows, and linear probes. The corresponding `bundle.json` records `promotion_eligible=false`: this all-years artifact is offline-only and is not attached to Set Transformer training yet.

## 8. Train A Full-Season Model

Use this when you want a conventional train/validation split:

```powershell
latentstrat train-features data/features/season/features_v58_2026.parquet `
  --output artifacts/season/v58_full_season_run `
  --epochs 100 `
  --mini-batch-size 256 `
  --prior-checkpoint artifacts/prior/prior_v564_latent16/checkpoint.pt `
  --rankings-sidecar data/sidecars/v58_2026/rankings_2026.parquet `
  --selections-sidecar data/sidecars/v58_2026/selections_2026.parquet `
  --playoffs-sidecar data/sidecars/v58_2026/playoffs_2026.parquet `
  --no-early-stopping `
  --restore-best `
  --tensorboard `
  --tensorboard-run-name v58_full_season_run
```

Expected artifacts:

- `feature_history.csv`
- `feature_common_metrics.csv`
- task-specific metric CSVs
- attention and zero-out diagnostics
- `v6_checkpoint.pt`

## 9. Run Walk-Forward Validation

Use this as the main season validation path:

```powershell
latentstrat validate-walk-forward `
  --features data/features/season/features_v58_2026.parquet `
  --prior-checkpoint artifacts/prior/prior_v564_latent16/checkpoint.pt `
  --rankings-sidecar data/sidecars/v58_2026/rankings_2026.parquet `
  --selections-sidecar data/sidecars/v58_2026/selections_2026.parquet `
  --playoffs-sidecar data/sidecars/v58_2026/playoffs_2026.parquet `
  --output artifacts/walk-forward/v58_walk_forward_2026 `
  --epochs 50 `
  --mini-batch-size 256 `
  --latent-dim 16 `
  --no-save-fold-checkpoints `
  --tensorboard `
  --tensorboard-run-name v58_walk_forward_v564_latent16_50ep_2026
```

Expected artifacts:

- `walk_forward_metrics.csv`
- `walk_forward_history.csv`
- `walk_forward_predictions.parquet`
- `config.json`
- TensorBoard summary and fold runs under `runs/`

After a future milestone attaches per-alliance score targets, compare its walk-forward predictions against the archived V5.8 export:

```powershell
latentstrat compare-world-model `
  --baseline artifacts/baselines/v5.8/walk-forward/walk_forward_predictions.parquet `
  --candidate artifacts/walk-forward/v6_score_attached/walk_forward_predictions.parquet `
  --output artifacts/walk-forward/v6_score_attached/promotion_vs_v58.csv
```

## 10. Validate Documentation And Repo State

For documentation-only changes:

```powershell
git diff --check
```

For code changes:

```powershell
ruff check .
pytest -q
```

Before staging:

```powershell
git status --short
```

Do not stage:

- `.env`
- `runs/`
- generated `data/` files unless explicitly intended
- `.obsidian/workspace.json` local workspace churn

## Common Problems

### OpenAI Calls Are Slow

The first prior build can take time because uncached narratives need embeddings. Reruns should mostly hit the SQLite embedding cache.

### Prior Training Rejects Feature Table

Current prior training expects V5.6.4 columns:

- normalized EPA trajectory targets and observed masks
- raw culture targets
- OpenAI narrative vectors

Rebuild prior features if the table is V5.6.1 or older.

### Team Number Out Of Bounds

If a team number is above the prior checkpoint cap, rebuild the prior with a larger `--max-team-number`.

### Walk-Forward Has No Folds

Check that the feature table has finite canonical `event_week` values and at least two distinct weeks.

### Metrics Look Better But Calibration Gets Worse

Check Brier score and log loss, not only accuracy. A model can pick winners more often while being too confident when wrong.
