# Evaluation And Artifacts

LatentStrat separates generated evidence from tracked source files. Generated outputs are ignored
locally, while tracked manifests preserve promoted baseline hashes where required.

## Metrics

Use walk-forward paired predictions for season-model promotion:

- Total score MSE and phase score MSE.
- Winner accuracy.
- Brier score.
- Log loss.
- Fold-level paired bootstrap comparisons.

Calibration matters independently from winner accuracy. A model can rank winners more accurately
while producing worse probabilities.

## Statbotics Comparison

Build a leakage-safe baseline from Statbotics pre-match `pred` fields:

```powershell
latentstrat season build-statbotics-baseline `
  --season 2026 `
  --output data/baselines/statbotics/statbotics_predictions_2026.parquet
```

Compare it with an existing LatentStrat prediction artifact without loading or training a model:

```powershell
latentstrat artifacts evaluate-predictions `
  --candidate artifacts/baselines/v5.8/walk-forward/walk_forward_predictions.parquet `
  --statbotics data/baselines/statbotics/statbotics_predictions_2026.parquet `
  --output artifacts/evaluation/v58-vs-statbotics
```

The evaluator uses exact `match_key` intersections and reports unmatched coverage without imputation. Tied matches remain in score metrics and are excluded from winner accuracy, Brier score, log loss, and calibration. Score MAE/MSE/RMSE, score-differential metrics, probability metrics, ten-bin calibration, and observation-weighted weekly and aggregate results are written alongside paired predictions.

Paired uncertainty uses an event-cluster bootstrap. Deltas are LatentStrat minus Statbotics; negative deltas mean lower loss. The interval reflects variation across the events represented in the artifact, not uncertainty about all possible FRC seasons.

The 2026 static reference extends this contract to additive, teammate-set, full-match, rolling
pRidge, mean, ridge, and optional Statbotics rows. Its long-form metric table declares whether each metric
is lower- or higher-is-better, preserves per-week and per-event evidence, and keeps official
ranking/selection/playoff/award probes in `auxiliary_metrics.csv`. Interaction support is evaluated
from event-cluster paired bootstrap probabilities by test week. A smoke or interrupted matrix is
explicitly incomplete and cannot display an architectural conclusion.

The follow-up static reliability contract treats weeks 6 and 8 as formal evidence and week 10 as
a championship stress test. Raw probabilities are primary. Sequential calibration may use only
earlier held-out predictions and is reported separately. Interaction deltas use a hierarchical
bootstrap that resamples seeds and event clusters. Labels remain provisional until all three seeds
and the metric-only Statbotics finalizer complete. See the
[reliability-study contract](reference-2026-static-reliability.md).

## Manifest Contract

New artifact directories write `manifest.json` containing:

- Workflow and artifact version.
- UTC generation timestamp and Git SHA.
- Resolved configuration.
- Source hashes.
- Output hashes and byte sizes.
- Promotion eligibility and migration provenance.

The manifest excludes itself from its output hash table.

## Paths

```text
artifacts/pretraining/prior/
artifacts/pretraining/prior-grid/
artifacts/pretraining/match-breakdown/
artifacts/season/
artifacts/walk-forward/
artifacts/experimental/frozen-targets/
artifacts/archive/
```

## Interpretation

Embedding PCA, t-SNE, parallel-coordinate, and cross-season neighbor plots are diagnostics. They can
reveal gradients, clustering, and suspicious calendar-year separation. They are not standalone
promotion evidence.

The V5.8 tagged replay remains the season-model comparison boundary:

```text
baselines/v5.8-baseline.json
artifacts/baselines/v5.8/
```

The replay is reproducible historical development evidence, not unbiased promotion evidence: its
fold checkpoints selected the best epoch using the same held-out weeks represented in the reported
predictions. The static reference now implements the nested temporal protocol for its deliberately
narrow 2026 path, but it still writes `promotion_eligible=false` because it is a one-season,
one-seed reference. The historical V5.8 comparison remains non-promotion evidence.
