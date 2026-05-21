# Metrics And Calibration

Evaluate LatentStrat as a probabilistic ML system, not just a classifier.

## Continuous Targets

Use continuous metrics for score-like targets:

- RMSE penalizes large misses.
- MAE is easier to interpret in point units.
- Slice continuous metrics by target availability when missingness or score breakdown coverage varies.

## Binary Targets

For win probabilities and other binary outcomes, prefer:

- Brier score for mean squared probability error.
- Log loss for confidence-sensitive probability quality.
- Calibration bins for checking whether predicted probabilities match observed frequencies.

Raw accuracy can be useful for communication, but it is not the main quality criterion. A model that predicts 75 percent win probability should win roughly 75 percent of those cases over a calibrated sample.

## Calibration Reading

When reviewing calibration bins:

- Look for bins near the diagonal between predicted probability and observed win rate.
- Check bin counts before trusting sparse bins.
- Watch for overconfident tails, especially near 0.0 or 1.0.
- Compare train, validation, and test behavior for leakage or overfitting signs.
