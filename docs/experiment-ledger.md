---
tags:
  - latentstrat
  - experiment-ledger
  - project-history
aliases:
  - "Experiment Ledger"
  - "Run Ledger"
related:
  - "[[project-history]]"
  - "[[changelog]]"
  - "[[metrics-and-artifacts]]"
  - "[[current-state]]"
---

# Experiment Ledger

This page records local LatentStrat experiments as engineering evidence. It is
not a paper, leaderboard, or universal claim. Use it to understand why the
current design moved in a particular direction.

For the narrative version of these decisions, see [Project History](project-history.md).
For the Git and semantic-version timeline, see [Changelog](changelog.md).

## Prior Runs

| Run | Artifact | Input | Epochs | Purpose | Local result | Design lesson |
|---|---|---|---:|---|---|---|
| V5.6 text-only 16D | `artifacts/prior_v56_latent16/` | `data/prior_features_v56_2026.parquet` | `1000` | Replace V5.5 text autoencoder with direct team-number dictionary | Produced `[20001, 16]` checkpoint-style latent table | Direct coordinate maps are simpler than runtime text encoders |
| V5.6.1 16D | `artifacts/prior_v561_latent16/` | `data/prior_features_v561_2026.parquet` | `1000` | Add single EPA target beside OpenAI target | Walk-forward comparison later gave better results than 8D and V5.6.2 | EPA-like quantitative strength signal helps the prior |
| V5.6.1 8D | `artifacts/prior_v561_latent8/` | `data/prior_features_v561_2026.parquet` | `1000` | Test smaller bottleneck | Walk-forward accuracy `0.6594`, worse than V5.6.1 16D `0.7067` | Default returned to 16D |
| V5.6.2 16D | `artifacts/prior_v562_latent16/` | `data/prior_features_v562_2026.parquet` | `1000` | Add raw cultural targets and 4-year normalized EPA trajectory | `norm_epa_mse` stayed `0.0`; EPA masks were effectively empty | Feature-family sanity checks must fail when a target family has no observations |
| V5.6.3 16D | `artifacts/prior_v563_latent16/` | `data/prior_features_v563_2026.parquet` | `1000` | Fix Statbotics normalized EPA extraction and grouped EPA trajectory loss | EPA contributed nonzero training loss; near-duplicate clustering remained for future/sibling rows | Correct EPA path and grouped trajectory loss were necessary but not sufficient |
| V5.6.4 16D | `artifacts/prior_v564_latent16/` | `data/prior_features_v563_2026.parquet` | `1000` | Change OpenAI loss from per-element mean to feature-summed vector distance | OpenAI loss had restored scale; prior inspection still showed dense future/sibling/ghost-token neighbors | Loss reduction is a first-class design choice in multi-task pretraining |

## Prior Inspection Evidence

The V5.6.4 inspection summary is:

```text
artifacts/prior_v564_latent16/inspection/v564_vs_v563_prior_summary.csv
```

Important local findings:

- V5.6.4 future rows had high norm concentration: mean norm about `1.91` with
  standard deviation about `0.015`.
- V5.6.4 future/sibling/ghost-token rows still had near-duplicate cosine
  neighbors at a high rate: `near_duplicate_neighbor_rate_0_999` about `0.997`.
- Anchor rows were less collapsed than synthetic future/sibling rows.

Interpretation: feature-summed OpenAI loss improved the training path but did
not fully solve synthetic-row clustering. Future work should inspect narrative
diversity, synthetic-row targets, and whether future/sibling rows need distinct
structural targets.

## Season And Walk-Forward Runs

| Run | Artifact | Input prior | Epochs/folds | Purpose | Local result | Design lesson |
|---|---|---|---|---|---|---|
| V5.7 full season | `artifacts/v57_full_season/` | V5.6.1-era prior | `19` epochs | First full V5.7 match-spine and sidecar run | Validation loss ended around `10.97` | Pipeline could train with sidecars, but longer stability was unproven |
| V5.7 long 100 | `artifacts/v57_full_season_long_100/` | V5.6.1-era prior | `100` epochs | Stress-test long heterogeneous training | Validation loss exploded to about `78.12` while train loss kept dropping | Homoscedastic overconfidence and sidecar overfit needed explicit controls |
| V5.7.2 stable 100 | `artifacts/v57_full_season_stable_100/` | V5.6.1-era prior | `100` epochs | Add log-var clamp, best restoration, cosine LR, and AdamW decay | Final validation loss still high, but LR and stability diagnostics were recorded | Long runs need best-validation restoration, not final-epoch saving |
| V5.8 common metrics 100 | `artifacts/v58_full_season_common_metrics_100/` | V5.6.1-era prior | `100` epochs | Add community-facing metrics to feature training | Wrote `feature_common_metrics.csv` | Score MSE, accuracy, Brier, and log loss are easier to compare than total homoscedastic loss |
| V5.8 walk-forward 16D | `artifacts/v58_walk_forward_tensorboard_2026/` | V5.6.1 16D | `10` folds, `5` epochs/fold | First TensorBoard walk-forward reference | Accuracy `0.7067`, Brier `0.1951`, log loss `0.5716` | Temporal validation became the primary model-quality path |
| V5.8 walk-forward 8D | `artifacts/v58_walk_forward_latent8_2026/` | V5.6.1 8D | `10` folds, `5` epochs/fold | Test 8D prior in season validation | Accuracy `0.6594`, score MSE worse than 16D | 8D was too tight for this local setup |
| V5.8 walk-forward V5.6.2 | `artifacts/v58_walk_forward_v562_latent16_2026/` | V5.6.2 16D | `10` folds, `5` epochs/fold | Test cultural-prior attempt | Accuracy `0.6604`, worse than V5.6.1 16D | EPA-mask bug likely hurt the prior despite richer targets |
| V5.8 walk-forward V5.6.4 50ep | `artifacts/v58_walk_forward_v564_latent16_50ep_2026/` | V5.6.4 16D | `10` folds, `50` epochs/fold | Long folded validation with latest prior | Best local score MSE and accuracy, but weaker Brier/log loss | Calibration is now the main follow-up |

## Current Walk-Forward Comparison

| Run | Phase score MSE | Total score MSE | Accuracy | Brier | Log loss | Matches |
|---|---:|---:|---:|---:|---:|---:|
| V5.6.1 16D, 5 epochs | `9869.29` | `10021.99` | `0.7067` | `0.1951` | `0.5716` | `18164` |
| V5.6.1 8D, 5 epochs | `12601.13` | `12792.16` | `0.6594` | `0.2190` | `0.6269` | `18164` |
| V5.6.2 16D, 5 epochs | `10760.56` | `10950.33` | `0.6604` | `0.2091` | `0.6037` | `18164` |
| V5.6.4 16D, 50 epochs | `8953.20` | `9055.87` | `0.7192` | `0.2025` | `0.6796` | `18164` |

Reading:

- V5.6.4 plus longer folded training gave the best local score MSE and match
  accuracy.
- The same run did not give the best Brier score or log loss.
- Future improvements should treat calibration as a first-class acceptance
  criterion, not a secondary chart.

## Open Follow-Ups

- Add in-repo Statbotics EPA and pRidge baselines only after explicitly
  designing leakage-safe source timing.
- Investigate calibration methods for walk-forward predictions.
- Review why synthetic future/sibling rows remain near-duplicate in prior space.
- Add clearer slice reports by week, event type, team archetype, and data
  availability.
