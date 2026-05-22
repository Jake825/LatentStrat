# Model Evaluation

LatentStrat evaluation should answer whether the model produces useful,
calibrated predictions and interpretable learned representations. Raw accuracy
is not enough for FRC match modeling.

## Binary Outcomes

For win probabilities and other binary targets, prefer probability-aware
metrics:

- **Brier score**: mean squared probability error.
- **Log loss**: confidence-sensitive probability quality.
- **Calibration bins**: predicted probability compared with observed frequency.

A model that predicts 75 percent win probability should win roughly 75 percent
of those cases over a well-populated calibration bin. Check bin counts before
trusting sparse regions of the curve.

## Continuous Targets

For V5 auto and teleop point targets, inspect:

- RMSE for large misses.
- MAE for point-unit interpretation.
- Availability slices when optional sources such as scouting are present.

Continuous and binary targets should be interpreted separately. A model can
improve score prediction while hurting win-probability calibration, or the
reverse.

## Ordinal Endgame

The endgame head predicts cumulative ordinal logits for
`None < Level1 < Level2 < Level3`. Evaluation reports:

- Class accuracy.
- Mean absolute class error.
- Expected-level MAE from cumulative probabilities.

Read these as per-slot robot outcomes, not alliance-level scores.

## Judged Awards

Award metrics are computed only where the NaN-masked ontology has finite target
entries. This prevents censored axes from producing false negatives. Reports
include finite-entry BCE, positive counts, and average precision when both
positive and negative finite labels exist.

## Baselines And Controls

Current implemented baselines are:

- Mean baselines.
- Ridge match-OPR baselines.

Do not describe Statbotics as an implemented baseline unless runtime code is
added for that integration. Statbotics can still be used as an external
comparison if the endpoint, field timing, and join keys are defined explicitly.

Evidence packets also compare the model with controls such as shuffled team
slots and null-label controls. Treat gains with suspicion if they disappear
against controls, only appear in tiny slices, or rely on features that were not
available at prediction time.

## Attention And Zero-Out Diagnostics

PMA attention tables and zero-out diagnostics are architecture-specific review
tools. V5 attention tables include missing-slot flags and assign zero attention
to masked slots. They can show which slots influenced pooled alliance
representations and how predictions respond when a slot is masked.

These artifacts are diagnostics, not causal proof. Use them alongside metrics,
calibration, feature availability, and scouting context.

## Embeddings

Embedding inspection exports durable `Z_base` vectors, PCA projections, cosine neighbors,
archetype similarities, PMA attention, and zero-out diagnostics. Use these to
form hypotheses about learned structure:

- PCA can reveal broad clusters but compresses the full embedding space.
- Cosine neighbors can suggest teams with similar learned profiles.
- Archetypes depend on the chosen reference statistics.

Embedding stories should be supported by model metrics and source data. Do not
treat embedding coordinates as direct scouting measurements.

## Review Checklist

Before claiming a model or feature change improved LatentStrat:

- Compare Brier score and log loss for binary targets.
- Review calibration bins and bin counts.
- Compare RMSE and MAE for continuous targets.
- Compare against mean and ridge match-OPR baselines.
- Review shuffled/null controls when available.
- Check availability slices for optional scouting or enrichment features.
- Confirm feature timing with the intended prediction point.
