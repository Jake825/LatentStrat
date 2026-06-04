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

This page documents the generated data contracts and output files used by the V5.6.4 prior,
tagged V5.8 baseline, and active V6-Lite pipeline. It is a practical reference, not a replacement
for source-level validation in tests.

## Prior Feature Table

`train-prior` expects one row per prior team number. Current local V5.6.4 training uses:

```text
data/features/prior/prior_features_v563_2026.parquet
```

Older local artifacts may still exist at root-level `data/prior_features_*.parquet` paths. Those are historical explicit outputs; new defaults write under `data/features/prior/`.

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
| `data/prior_features_v563_2026.parquet` | `12501` | current local V5.6.4 prior-training input; new rebuilds should use `data/features/prior/` |

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
| `data/features_v57_2026.parquet` | `18195` | `165` | V5.7 full-season feature table, historical path |
| `data/features_v58_2026.parquet` | `18195` | `166` | V5.8 canonical week table, historical path; new rebuilds should use `data/features/season/` |

The exact number of columns can change as optional scouting or target columns are added. Required schema checks should focus on named column groups, not raw column count.

## Sidecar Tables

Sidecars are optional auxiliary labels. They do not change the match-row grain. In walk-forward validation, training sidecars are filtered to `event_week <= train_max_week`; validation sidecars are filtered to `event_week == val_week`.

| Sidecar | Key columns | Current local rows |
|---|---|---:|
| `rankings_2026.parquet` | `season`, `event_key`, `team_key`, `qual_rank`, `matches_played`, ranking stats, `raw_event_week`, `event_week` | `8102` |
| `selections_2026.parquet` | `season`, `event_key`, `alliance_number`, `captain_team_key`, `pick_team_key`, `pick_order`, `passed_over_team_key`, `raw_event_week`, `event_week` | `3597` |
| `playoffs_2026.parquet` | `season`, `event_key`, `alliance_number`, `team_1_key`, `team_2_key`, `team_3_key`, `playoff_finish_order`, `status`, `raw_event_week`, `event_week` | `1666` |

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
| `prior_tsne_coordinates.csv` | t-SNE `x/y` coordinates plus team metadata |
| `prior_latent_stat_correlations.csv` | latent dimension, stat name, Pearson correlation, and finite row count |
| PNG plots | PCA, t-SNE, latent/stat heatmap, norm histogram, EPA/culture colorings where available |

## Feature Training Artifacts

`train-features` writes:

| File | Meaning |
|---|---|
| `v6_checkpoint.pt` | strict-schema V6 Set Transformer checkpoint, including supervised-only ablations |
| `feature_history.csv` | epoch-level train/validation losses and task diagnostics |
| `feature_common_metrics.csv` | common match metrics by split |
| `feature_binary_metrics.csv` | Brier/log-loss style binary metrics |
| `feature_continuous_metrics.csv` | continuous target RMSE/MAE |
| `feature_calibration.csv` | binary probability calibration bins |
| `feature_availability_slices.csv` | validation metrics for unseen and low-data team slices |
| `feature_endgame_metrics.csv` | ordinal endgame metrics |
| `feature_award_metrics.csv` | judged-award auxiliary metrics |
| `feature_set_attention.csv` | PMA attention diagnostics |
| `feature_zero_out_diagnostics.csv` | sensitivity to slot zero-out diagnostics |
| `feature_team_attention_summary.csv` | team-level PMA attention averages and max-attention rates |
| `feature_team_zero_out_sensitivity.csv` | team-level validation zero-out RMSE deltas |
| `feature_embedding_drift.csv` | prior-to-final PCA coordinates and drift norms when a prior checkpoint is supplied |
| `feature_event_delta_summary.csv` | `Z_event` L2 norms by team-event row |
| `feature_selection_value.csv` | TeamValueHead scores joined to selection sidecar draft order |
| `feature_training_loss.png` | train and validation loss plot |
| `feature_task_losses.png` | raw task-loss plot |
| `feature_task_precision_evolution.png` | homoscedastic task precision curves |
| `feature_win_calibration.png` | binary calibration plot |
| `feature_attention_entropy.png` | PMA attention entropy plot |
| `feature_zero_out_delta_rmse.png` | zero-out sensitivity plot |
| `feature_pma_carry_support.png` | prior EPA or embedding norm against average PMA attention |
| `feature_zero_out_war.png` | prior EPA or embedding norm against team zero-out delta RMSE |
| `feature_embedding_drift_quiver.png` | PCA arrow plot from prior vector to final team-event vector |
| `feature_event_delta_by_week.png` | event-delta magnitude boxplot by canonical week |
| `feature_team_value_vs_draft_pick.png` | selection-sidecar draft pick number against TeamValueHead score |
| `feature_*_sidecar.csv` | indexed sidecar copies when sidecars are supplied |

`feature_history.csv` is a training-stability file. Use `feature_common_metrics.csv` or walk-forward metrics for model-quality claims.

## Walk-Forward Artifacts

`validate-walk-forward` writes:

| File | Meaning |
|---|---|
| `walk_forward_metrics.csv` | one row per fold plus `AVERAGE` |
| `walk_forward_history.csv` | fold and epoch training history |
| `walk_forward_predictions.parquet` | fold-match paired comparison rows |
| `config.json` | workflow config, Git commit, and SHA-256 source hashes |
| TensorBoard event files | summary and fold-level training curves |
| optional fold checkpoints | only when `--save-fold-checkpoints` is enabled |

The `AVERAGE` row uses row-weighted validation aggregates for common metrics. Do not average fold means by hand when validation fold sizes differ.

## V6-Lite Frozen Target Bundles

The V6-Lite V1 historical match-breakdown artifact writes:

| File | Meaning |
|---|---|
| `eval_model.pt` | holdout-only season-specific encoders/decoders and shared bottleneck used for diagnostics |
| `model.pt` | all-data retrain used for exported embeddings |
| `embeddings.parquet` | one frozen 16D latent per played alliance |
| `schema.json` | typed season-specific vector schemas |
| `union_schema_audit.json` | cross-season presence report; never used as a padded training tensor |
| `bundle.json` | source hashes, Git SHA, seasons, event filter, widths, provenance, and `promotion_eligible=false` |
| `validation_report.json` | grouped reconstruction loss, effective rank, nearest neighbors, and score probes |
| `training_history.csv` | eval and all-data epoch histories |

The canonical directory is:

```text
artifacts/pretraining/match-breakdown/v1_2015_2026/
```

The match-breakdown encoder intentionally writes no normalizer file. It uses raw values plus observed masks and keeps per-season vector widths separate.

The explicit V2 ablation writes `artifacts/pretraining/match-breakdown/v2_2015_2026/` without
replacing V1. It adds `rule_audit.json` and `resolved_config.json`. The rule audit records separate
eval-training and all-data resolution scopes, per-season schema hashes, rule-file hashes, enabled
rules, disabled hypotheses, failure counts, example row IDs, and residual summaries. V2
checkpoints include a diagnostic-only scalar quality head; `embeddings.parquet` remains 16D.

`inspect-match-breakdown-encoder` writes a standalone `inspection/` directory without changing
`bundle.json`:

| File | Meaning |
|---|---|
| `production_pca_coordinates.csv` | every production embedding in the production PCA frame |
| `holdout_pca_coordinates.csv` | eval-model holdout vectors orthogonally aligned for display in the production PCA frame |
| `pca_explained_variance.csv` and `.png` | component and cumulative production PCA variance |
| `production_pca_multiview.png`, `holdout_pca_multiview.png` | fixed-coordinate season, percentile, and top-component views |
| `production_tsne_coordinates.csv`, `holdout_tsne_coordinates.csv` | deterministic season-and-score-decile sampled t-SNE coordinates |
| `production_tsne_multiview.png`, `holdout_tsne_multiview.png` | local-neighborhood views; do not compare their absolute coordinates |
| `latent_archetype_profiles.csv`, `latent_parallel_coordinates.png` | raw and standardized production latent profiles by archetype slice |
| `cross_season_neighbor_nodes.csv`, `cross_season_neighbor_edges.csv`, `cross_season_neighbor_network.png` | PCA-positioned cross-season Euclidean neighbor network |
| `component_metric_sources.json` | one prioritized safe component aggregate per season and audited missing seasons |
| `inspection_manifest.json` | report input hashes, generation options, generated-file hashes, and geometry provenance |
| optional `baseline_neighbor_drift.csv`, `baseline_score_sorting_drift.csv` | keyed neighbor-overlap and score-sorting drift diagnostics for explicit V1/V2 comparison |

Projection assets are diagnostic interpretation aids. They do not change the artifact bundle and
do not replace leakage-safe fold metrics.

Award, rank, and pick target-space scaffolds remain under `artifacts/experimental/frozen-targets/`. The umbrella `bundle.json` records enabled spaces and source files. `train-features --world-model-bundle` fails if an enabled integrated space is absent. Score embedding configuration remains disabled until a follow-up attaches per-alliance targets to season training.

Large V5.8 comparison artifacts stay ignored under `artifacts/baselines/v5.8/`. The tracked
`baselines/v5.8-baseline.json` file records their SHA-256 hashes, summary metrics, tag, commit SHA,
and regeneration command.

## Artifact Naming Convention

New defaults group artifacts by run type. Use versioned artifact directories for experiments:

```text
artifacts/prior/prior_v564_latent16/
artifacts/walk-forward/v58_walk_forward_v564_latent16_50ep_2026/
```

Do not overwrite old experiment folders when comparing model changes. The experiment ledger depends on stable paths.

## Canonical Generated Storage

The canonical generated-file layout is:

```text
data/cache/tba.sqlite
data/cache/statbotics.sqlite
data/cache/openai_embeddings.sqlite
data/pretraining/match-breakdown/corpus.sqlite
data/scouting/scouting.db
data/embeddings/latentstrat_embeddings.sqlite
data/features/prior/
data/features/season/
data/features/event/
data/features/pretraining/match-breakdown/
data/sidecars/
artifacts/prior/
artifacts/prior-grid/
artifacts/season/
artifacts/walk-forward/
artifacts/experimental/frozen-targets/
artifacts/baselines/
artifacts/smoke/
artifacts/evidence/
artifacts/inspection/
runs/
```

Parquet feature tables remain file artifacts. Provider caches, scouting rows, durable embedding stores, and the raw match-breakdown corpus use SQLite. The local organizer script `scripts/organize_local_outputs.ps1` can dry-run moves from legacy paths into this layout.
