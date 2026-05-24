---
tags:
  - latentstrat
  - feature-pipeline
  - metrics
aliases:
  - "Schemas And Artifacts Reference"
  - "Artifact Schema Reference"
related:
  - "[[feature-pipeline]]"
  - "[[metrics-and-artifacts]]"
  - "[[data-sources]]"
  - "[[rebuild-from-scratch]]"
  - "[[scouting-data-layer]]"
  - "[[scouting-data-ingestion]]"
---

# Schemas And Artifacts Reference

This page documents the generated data contracts and output files used by the current V5.6.4/V5.8 pipeline. It is a practical reference, not a replacement for source-level validation in tests.

## Prior Feature Table

`train-prior` expects one row per prior team number. Current local V5.6.4 training uses:

```text
data/prior_features_v563_2026.parquet
```

Required groups:

| Group | Columns |
|---|---|
| Identity | `team_number`, `team_key`, `archetype`, `target_season` |
| Narrative | `narrative`, `narrative_hash`, `embedding_model`, `llm_dim` |
| OpenAI target | `openai_narrative_vector` |
| Normalized EPA trajectory | `norm_epa_t_minus_4` through `norm_epa_t_minus_1` |
| EPA observation masks | `norm_epa_observed_t_minus_4` through `norm_epa_observed_t_minus_1` |
| Culture targets | `raw_rookie_year_delta`, `raw_seasons_played`, `raw_total_award_count`, `raw_blue_banner_count`, `raw_championship_appearance_count`, `raw_championship_win_count`, `raw_technical_award_count` |

V5.6.4 is a loss/model update, not a new feature schema. It uses the V5.6.3 feature table schema with feature-summed OpenAI loss during training.

Local row counts:

| File | Rows | Notes |
|---|---:|---|
| `data/prior_features_v56_2026.parquet` | `20001` | historical 20,000-row experiment table |
| `data/prior_features_v561_2026.parquet` | `12501` | text plus single EPA target generation |
| `data/prior_features_v562_2026.parquet` | `12501` | cultural target attempt with broken EPA masks |
| `data/prior_features_v563_2026.parquet` | `12501` | current V5.6.4 prior-training input |

## Match Feature Table

`train-features` and `validate-walk-forward` consume a match-grain Parquet file. The TBA match table remains the spine.

Required groups:

| Group | Columns |
|---|---|
| Match identity | `season`, `event_key`, `match_key`, `comp_level`, `set_number`, `match_number` |
| Time and ordering | `raw_event_week`, `event_week`, `time`, `actual_time`, `predicted_time`, `sort_ordinal` |
| Team slots | `red_team_1_key` through `red_team_3_key`, `blue_team_1_key` through `blue_team_3_key` |
| Continuous targets | red/blue auto and teleop point targets configured by `LatentStratOptions.cont_targets` |
| Win target | `red_win` |
| Endgame targets | per-slot normalized endgame labels |
| Award targets | per-slot judged-award auxiliary labels |
| V5.7 score-breakdown targets | atomic counts, committed fouls, bonus binaries, and special binaries |

Current local files:

| File | Rows | Columns | Notes |
|---|---:|---:|---|
| `data/features_v57_2026.parquet` | `18195` | `165` | V5.7 full-season feature table |
| `data/features_v58_2026.parquet` | `18195` | `166` | V5.8 canonical week table |

The exact number of columns can change as optional scouting or target columns are added. Required schema checks should focus on named column groups, not raw column count.

## Sidecar Tables

Sidecars are optional auxiliary labels. They do not change the match-row grain. In walk-forward validation, training sidecars are filtered to `event_week <= train_max_week`; validation sidecars are filtered to `event_week == val_week`.

| Sidecar | Key columns | Current local rows |
|---|---|---:|
| `rankings_2026.parquet` | `season`, `event_key`, `team_key`, `qual_rank`, `matches_played`, ranking stats, `raw_event_week`, `event_week` | `8102` |
| `selections_2026.parquet` | `season`, `event_key`, `alliance_number`, `captain_team_key`, `pick_team_key`, `pick_order`, `passed_over_team_key`, `raw_event_week`, `event_week` | `3597` |
| `playoffs_2026.parquet` | `season`, `event_key`, `alliance_number`, `team_1_key`, `team_2_key`, `team_3_key`, `playoff_finish_order`, `status`, `raw_event_week`, `event_week` | `1666` |

Declines are intentionally not modeled because TBA does not reliably populate them in event data.

## Prior Artifacts

`train-prior` writes a directory or checkpoint path. Directory output writes:

```text
checkpoint.pt
```

The current stripped checkpoint contains:

- `embedding_table`
- `max_team_number`
- `target_season`
- `prior_opts`
- `source_feature_path`
- `feature_metadata`
- `history`

It does not contain decoder, head, or log-var weights.

`inspect-prior` writes:

| File | Meaning |
|---|---|
| `prior_latent_table.csv` | one row per team vector with PCA columns and feature metadata when supplied |
| `prior_nearest_neighbors.csv` | cosine neighbors for each team |
| `prior_sanity_checks.csv` | finite, norm, collapse, and dominant-dimension checks |
| `prior_training_history.csv` | copied training history |
| PNG plots | PCA, norm histogram, EPA/culture colorings where available |

## Feature Training Artifacts

`train-features` writes:

| File | Meaning |
|---|---|
| `v5_checkpoint.pt` | trained Set Transformer checkpoint |
| `feature_history.csv` | epoch-level train/validation losses and task diagnostics |
| `feature_common_metrics.csv` | common match metrics by split |
| `feature_binary_metrics.csv` | Brier/log-loss style binary metrics |
| `feature_continuous_metrics.csv` | continuous target RMSE/MAE |
| `feature_endgame_metrics.csv` | ordinal endgame metrics |
| `feature_award_metrics.csv` | judged-award auxiliary metrics |
| `feature_set_attention.csv` | PMA attention diagnostics |
| `feature_zero_out_diagnostics.csv` | sensitivity to slot zero-out diagnostics |
| `feature_*_sidecar.csv` | indexed sidecar copies when sidecars are supplied |

`feature_history.csv` is a training-stability file. Use `feature_common_metrics.csv` or walk-forward metrics for model-quality claims.

## Walk-Forward Artifacts

`validate-walk-forward` writes:

| File | Meaning |
|---|---|
| `walk_forward_metrics.csv` | one row per fold plus `AVERAGE` |
| `walk_forward_history.csv` | fold and epoch training history |
| TensorBoard event files | summary and fold-level training curves |
| optional fold checkpoints | only when `--save-fold-checkpoints` is enabled |

The `AVERAGE` row uses row-weighted validation aggregates for common metrics. Do not average fold means by hand when validation fold sizes differ.

## Artifact Naming Convention

Use versioned artifact directories for experiments:

```text
artifacts/prior_v564_latent16/
artifacts/v58_walk_forward_v564_latent16_50ep_2026/
```

Do not overwrite old experiment folders when comparing model changes. The experiment ledger depends on stable paths.
