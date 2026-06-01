---
tags:
  - latentstrat
  - metrics
  - model-evaluation
aliases:
  - "Metrics And Artifacts"
  - "Artifact Guide"
related:
  - "[[model-evaluation]]"
  - "[[experiment-ledger]]"
  - "[[current-state]]"
  - "[[embedding-inspection]]"
---

# Metrics And Artifacts

LatentStrat produces several kinds of outputs. Some are model-quality metrics. Some are training diagnostics. Some are interpretation aids.

The safest rule is: never trust one number by itself.

For the exact metric formulas and model-output tensor shapes behind these reports, see the [Model Architecture Reference](model-architecture-reference.md). For run-by-run local evidence, see the [Experiment Ledger](experiment-ledger.md).

## Common Match Metrics

V5.8 reports common FRC-friendly next-match metrics.

| Metric | Meaning | Lower or higher |
|---|---|---|
| Phase score MSE | Error on predicted `auto + teleop` score for both alliances | Lower |
| Total score MSE | Error on FMS total score approximation, including opponent foul points when available | Lower |
| Match accuracy | Whether `p_red_win >= 0.5` matched the red-win label | Higher |
| Brier score | Mean squared error of win probability | Lower |
| Log loss | Penalizes confident wrong predictions | Lower |

LatentStrat reports red win probability:

```text
p_red_win = sigmoid(red_win_logit)
```

Blue win probability is:

```text
1 - p_red_win
```

## Why Brier And Log Loss Matter

Accuracy treats a 51 percent prediction and a 99 percent prediction the same if the predicted team wins. Brier score and log loss care about confidence.

For scouting and strategy, confidence matters. A picklist recommendation should not only say which alliance is favored; it should also signal how certain the model is.

## Prior Inspection Artifacts

`inspect-prior` writes a directory like:

```text
artifacts/prior_v564_latent16/inspection/
```

Important files:

- `prior_latent_table.csv`: one row per team embedding.
- `prior_nearest_neighbors.csv`: cosine neighbors in latent space.
- `prior_sanity_checks.csv`: basic checks on the embedding table.
- `prior_training_history.csv`: copied training curve.
- `prior_tsne_coordinates.csv`: deterministic t-SNE coordinates for the Day Zero team galaxy.
- `prior_latent_stat_correlations.csv`: Pearson correlations between latent dimensions and available prior stats.
- PCA, t-SNE, correlation-heatmap, and histogram images.

Use these to inspect whether the Day Zero prior has sensible structure. For example, if every future team is almost identical to every other future team, the prior may not be using narrative information strongly enough.

## Feature Training Artifacts

`train-features` writes a directory such as:

```text
artifacts/v58_full_season_common_metrics_100/
```

Important files:

- `feature_history.csv`: train/validation loss by epoch.
- `feature_common_metrics.csv`: common next-match metrics by split.
- `feature_binary_metrics.csv`: Brier and log loss for binary outputs.
- `feature_continuous_metrics.csv`: RMSE and MAE for continuous outputs.
- `feature_calibration.csv`: calibration bins for binary probabilities.
- `feature_availability_slices.csv`: validation slices for unseen and low-data teams.
- `feature_endgame_metrics.csv`: ordinal endgame metrics.
- `feature_award_metrics.csv`: award auxiliary-label metrics.
- `feature_set_attention.csv`: attention diagnostics.
- `feature_zero_out_diagnostics.csv`: prediction response to removing slots.
- `feature_team_attention_summary.csv`: team-level PMA attention aggregates.
- `feature_team_zero_out_sensitivity.csv`: per-team validation sensitivity when a visible slot is zeroed.
- `feature_embedding_drift.csv`: prior-to-final vector displacement when a prior checkpoint is supplied.
- `feature_event_delta_summary.csv`: `Z_event` L2 magnitudes by team-event row.
- `feature_selection_value.csv`: TeamValueHead scores for actual selection-sidecar picks.
- PNG plots for training loss, raw task losses, task precision, win calibration, attention entropy, carry/support, zero-out WAR, embedding drift, event deltas, and selection value when inputs are available.

`feature_history.csv` is a training diagnostic. It tells you whether training was stable, not whether the model is strategically useful by itself.

Attention, zero-out, t-SNE, and drift visuals are diagnostic evidence. Use them to form scouting questions, then check calibration, common metrics, sidecar labels, and validation slices before making quality claims.

## Walk-Forward Artifacts

`validate-walk-forward` writes:

- `walk_forward_metrics.csv`
- `walk_forward_history.csv`
- `walk_forward_predictions.parquet`
- `config.json` with source SHA-256 hashes and the exact comparison configuration
- TensorBoard fold runs.

`walk_forward_metrics.csv` contains one row per fold plus an `AVERAGE` row. The common metrics in the `AVERAGE` row are row-weighted season aggregates, not simple unweighted means of fold means.

This is the most important validation file for season-level claims because it simulates training on earlier weeks and predicting later weeks.

`walk_forward_predictions.parquet` is keyed by fold and match. V6 promotion uses
`compare-world-model` to bootstrap paired prediction differences within folds with seed `2026`.

## V6-Lite World-Model Artifacts

The V6-Lite V1 historical match-breakdown artifact lives under
`artifacts/world_model/match_breakdown/v1_2015_2026/` and writes:

| File | Meaning |
|---|---|
| `eval_model.pt` | holdout-only model used for diagnostics |
| `model.pt` | all-data retrain used for frozen embedding export |
| `embeddings.parquet` | one 16D target per played alliance |
| `schema.json` | separate typed schema for each eligible season |
| `union_schema_audit.json` | field-presence audit only; training never pads to this union |
| `bundle.json` | source hashes, Git SHA, event filter, widths, provenance, and audit counts |
| `validation_report.json` | grouped reconstruction losses, effective rank, neighbors, and probes |
| `training_history.csv` | eval and production retrain curves |

The historical encoder uses raw values and masks without normalizer files. It defaults to normal
official TBA event types `0..5`; FOC `6` and remote `7` are opt-in. The artifact records
`promotion_eligible=false` because all-years training is not leakage-safe walk-forward evidence.

`inspect-match-breakdown-encoder` writes a separate `inspection/` directory beside the bundle:

| Output | Meaning |
|---|---|
| `production_pca_multiview.png` and `holdout_pca_multiview.png` | one fixed PCA frame colored by season and within-season score/component percentiles |
| `production_tsne_multiview.png` and `holdout_tsne_multiview.png` | deterministic sampled local-neighborhood views; absolute coordinates are not comparable across the two plots |
| `pca_explained_variance.csv` and `.png` | production-space PCA variance summary |
| `latent_parallel_coordinates.png` and `latent_archetype_profiles.csv` | raw and standardized production latent profiles for five deterministic archetype slices |
| `cross_season_neighbor_nodes.csv`, `cross_season_neighbor_edges.csv`, and `.png` | extreme production rows with nearest different-season neighbors under Euclidean latent distance |
| `component_metric_sources.json` | selected safe aggregate field per season and missing-component seasons |
| `inspection_manifest.json` | input hashes, options, generated-file hashes, and production-versus-holdout provenance |

The holdout vectors come from `eval_model.pt` and are orthogonally aligned only for display in the
production PCA frame. These projection plots can reveal calendar-year separation, score gradients,
and cross-season analogies. They are interpretation aids, not standalone promotion evidence.

Award prototypes retain normalized raw 256D OpenAI vectors without PCA. Score attachment to the
Set Transformer remains deferred until per-alliance runtime integration is implemented.

Promotion comparisons require the 95% bootstrap confidence-interval upper bound to stay within
`+0.002` Brier, `+0.01` log loss, and `+1%` total-score MSE versus both tagged V5.8 and the prior
accepted V6 phase.

## Local Experiment Snapshot

The current local artifact set includes these observed `AVERAGE` rows:

| Run | Phase score MSE | Total score MSE | Accuracy | Brier | Log loss | Matches |
|---|---:|---:|---:|---:|---:|---:|
| V5.6.1 16D, 5-epoch walk-forward | 9869.29 | 10021.99 | 0.7067 | 0.1951 | 0.5716 | 18164 |
| V5.6.1 8D, 5-epoch walk-forward | 12601.13 | 12792.16 | 0.6594 | 0.2190 | 0.6269 | 18164 |
| V5.6.2 16D, 5-epoch walk-forward | 10760.56 | 10950.33 | 0.6604 | 0.2091 | 0.6037 | 18164 |
| V5.6.4 16D, 50-epoch walk-forward | 8953.20 | 9055.87 | 0.7192 | 0.2025 | 0.6796 | 18164 |
| Frozen `v5.8-baseline` replay, V5.6.4 16D, 50 epochs | 8829.77 | 8971.60 | 0.7164 | 0.2090 | 0.7227 | 18164 |

Interpretation:

- The 8D prior underperformed 16D in the local comparison.
- The V5.6.4 50-epoch run improved score MSE and accuracy.
- The same run worsened log loss and Brier relative to the earlier V5.6.1 16D run, so calibration needs more investigation.

These are local experiment results, not permanent claims about the model.

V6 promotion uses the frozen `v5.8-baseline` replay row and its paired prediction export. The
older V5.6.4 row remains useful historical evidence but is not the hash-manifest comparison input.

## Reading TensorBoard

TensorBoard is for training dynamics:

```bash
tensorboard --logdir=runs
```

Useful curves:

- Train loss versus validation loss.
- Learning rate.
- Homoscedastic log variances.
- Precision weights.
- OpenAI, EPA, culture, match-spine, and sidecar losses.

If training loss keeps improving while validation gets worse, the model is overfitting or the loss balancer is overconfident.

## What Counts As An Improvement

A model change is stronger evidence if it:

- Improves walk-forward metrics.
- Improves Brier or log loss, not just accuracy.
- Improves score MSE without destroying calibration.
- Still beats simple baselines and controls.
- Does not rely on future information.
- Looks reasonable in artifact slices and diagnostics.

Embedding plots and attention tables are useful evidence, but they are not proof by themselves.

## Related

- [Model evaluation](model-evaluation.md): calibration, baselines, controls, evidence packets, and review discipline.
- [Experiment ledger](experiment-ledger.md): run-by-run local evidence and artifact paths.
- [Current state](current-state.md): promoted artifacts and current caveats.
- [Model architecture reference](model-architecture-reference.md): metric formulas and loss equations.
- [Embedding inspection](embedding-inspection.md): latent-space diagnostics and nearest-neighbor interpretation.
