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
- PCA and histogram images.

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
- `feature_endgame_metrics.csv`: ordinal endgame metrics.
- `feature_award_metrics.csv`: award auxiliary-label metrics.
- `feature_set_attention.csv`: attention diagnostics.
- `feature_zero_out_diagnostics.csv`: prediction response to removing slots.

`feature_history.csv` is a training diagnostic. It tells you whether training was stable, not whether the model is strategically useful by itself.

## Walk-Forward Artifacts

`validate-walk-forward` writes:

- `walk_forward_metrics.csv`
- `walk_forward_history.csv`
- TensorBoard fold runs.

`walk_forward_metrics.csv` contains one row per fold plus an `AVERAGE` row. The common metrics in the `AVERAGE` row are row-weighted season aggregates, not simple unweighted means of fold means.

This is the most important validation file for season-level claims because it simulates training on earlier weeks and predicting later weeks.

## Local Experiment Snapshot

The current local artifact set includes these observed `AVERAGE` rows:

| Run | Phase score MSE | Total score MSE | Accuracy | Brier | Log loss | Matches |
|---|---:|---:|---:|---:|---:|---:|
| V5.6.1 16D, 5-epoch walk-forward | 9869.29 | 10021.99 | 0.7067 | 0.1951 | 0.5716 | 18164 |
| V5.6.1 8D, 5-epoch walk-forward | 12601.13 | 12792.16 | 0.6594 | 0.2190 | 0.6269 | 18164 |
| V5.6.2 16D, 5-epoch walk-forward | 10760.56 | 10950.33 | 0.6604 | 0.2091 | 0.6037 | 18164 |
| V5.6.4 16D, 50-epoch walk-forward | 8953.20 | 9055.87 | 0.7192 | 0.2025 | 0.6796 | 18164 |

Interpretation:

- The 8D prior underperformed 16D in the local comparison.
- The V5.6.4 50-epoch run improved score MSE and accuracy.
- The same run worsened log loss and Brier relative to the earlier V5.6.1 16D run, so calibration needs more investigation.

These are local experiment results, not permanent claims about the model.

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
