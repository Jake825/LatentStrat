# Baselines and Controls

LatentStrat implements three comparison families:

- Mean targets as a minimal regression control.
- Ridge match-OPR baselines fit from the feature table.
- A leakage-safe Statbotics baseline artifact built from pre-match `pred` fields.

Build the provider baseline with `latentstrat season build-statbotics-baseline`. Compare saved candidate and baseline predictions with `latentstrat artifacts evaluate-predictions`; no model loading or retraining is required.

## Comparison Contract

- Validate one row per `match_key` and one season per Statbotics artifact.
- Pair on exact match-key intersections and report unmatched coverage without imputation.
- Use identical actual outcomes for both models.
- Exclude tied matches from binary probability metrics while retaining them for score metrics.
- Report score MAE/MSE/RMSE, score-differential metrics, winner accuracy, Brier score, log loss, calibration, and event-cluster bootstrap intervals as available.
- Interpret LatentStrat-minus-Statbotics loss deltas consistently: negative loss deltas favor LatentStrat.

Shuffled targets, feature ablations, missing-data slices, and null inputs remain useful controls. Never promote a result that depends on a leakage-prone split, a tiny slice, or selection on the same held-out period being reported.
