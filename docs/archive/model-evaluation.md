---
tags:
  - latentstrat
  - model-evaluation
  - metrics
aliases:
  - "Model Evaluation"
  - "Evaluation Guide"
related:
  - "[[metrics-and-artifacts]]"
  - "[[experiment-ledger]]"
  - "[[embedding-inspection]]"
  - "[[model-selection]]"
---

# Model Evaluation

For a student-friendly guide to the output files, start with [Metrics And Artifacts](metrics-and-artifacts.md).

LatentStrat evaluation should answer whether the model produces useful, calibrated predictions and interpretable learned representations. Raw accuracy is not enough for FRC match modeling.

## Binary Outcomes

For win probabilities and other binary targets, prefer probability-aware metrics:

- **Brier score**: mean squared probability error.
- **Log loss**: confidence-sensitive probability quality.
- **Calibration bins**: predicted probability compared with observed frequency.

A model that predicts 75 percent win probability should win roughly 75 percent of those cases over a well-populated calibration bin. Check bin counts before trusting sparse regions of the curve.

V5.7 bonus and special heads output logits. Convert them to probabilities only for metrics and reporting; training uses masked `BCEWithLogitsLoss`.

## Continuous Targets

For V5 auto, teleop, V5.7 atomic-count, and committed-foul targets, inspect:

- RMSE for large misses.
- MAE for point-unit interpretation.
- Availability slices when optional sources such as scouting are present.

Continuous and binary targets should be interpreted separately. A model can improve score prediction while hurting win-probability calibration, or the reverse.

V5.7 score-breakdown targets may be sparse when TBA omits breakdowns. Metrics should include only finite target entries.

## Learning-To-Rank Sidecars

Rankings, alliance selections, and playoff sidecars are post-event auxiliary training labels. Evaluate them as representation-shaping losses, not pre-match prediction inputs:

- Qualification rank pairs should give lower numerical ranks higher team-value scores.
- Playoff alliance pairs should give better finish orders higher alliance-value scores.
- Alliance selection triplets should place captains closer to selected picks than to computed passed-over teams.

## Ordinal Endgame

The endgame head predicts cumulative ordinal logits for `None < Level1 < Level2 < Level3`. Evaluation reports:

- Class accuracy.
- Mean absolute class error.
- Expected-level MAE from cumulative probabilities.

Read these as per-slot robot outcomes, not alliance-level scores.

## Judged Awards

Award metrics are computed only where the NaN-masked ontology has finite target entries. This prevents censored axes from producing false negatives. Reports include finite-entry BCE, positive counts, and average precision when both positive and negative finite labels exist.

## Baselines And Controls

Current implemented baselines are:

- Mean baselines.
- Ridge match-OPR baselines.

The mean baseline predicts the train-split mean for continuous targets. It is a sanity floor: a useful model should beat it on validation rows.

The ridge match-OPR baseline builds a sparse team/alliance design matrix from the match table and fits `sklearn.linear_model.Ridge` with no intercept. It is a linear team-contribution baseline, not a nonlinear interaction model.

Do not describe Statbotics as an implemented baseline unless runtime code is added for that integration. Statbotics can still be used as an external comparison if the endpoint, field timing, and join keys are defined explicitly.

LatentStrat's V5.8 walk-forward reports are designed to speak the same bias-variance language used by FRC linear models. OPR is a low-bias, high-variance linear estimate; EPA is a historical biased anchor; pRidge is a linear prior-regularized compromise. LatentStrat uses the V5.6.4 Day Zero embedding as its prior anchor, then learns nonlinear alliance interactions through the Set Transformer. The current implementation exports comparable LatentStrat metrics, but it does not yet implement Statbotics EPA or pRidge baselines inside the repo.

For common next-match reporting, use:

- **Next-match phase score MSE**: MSE over red and blue predicted `auto_pts + teleop_pts`.
- **Next-match total score MSE**: MSE against FMS totals when total-score and committed-foul targets are available.
- **Match accuracy**: whether `p_red_win >= 0.5` matches the `red_win` label.
- **Win Brier score** and **win log loss**: probability-quality metrics for `p_red_win`. Blue win probability is `1 - p_red_win`.

## Evidence Packets And Controls

`build-evidence-packet` groups model metrics, baselines, shuffled controls, null-label controls, availability slices, and embedding diagnostics into one review packet.

Use controls to catch false progress:

- A shuffled-team control should damage performance if the model is really using team identity.
- A shuffled-target or null-label control should fail to produce meaningful validation signal.
- Availability slices should show whether a gain only appears in rows with an optional source such as scouting data.

Treat gains with suspicion if they disappear against controls, only appear in tiny slices, or rely on features that were not available at prediction time.

## Calibration Review

Calibration asks whether predicted probabilities match observed frequencies. For example, matches predicted around 70 percent red win probability should have red win about 70 percent of the time over a sufficiently populated bin.

When reviewing calibration:

- Check Brier score and log loss before celebrating raw accuracy.
- Inspect calibration bins and bin counts; sparse bins are weak evidence.
- Watch for long-run overconfidence: accuracy can improve while log loss gets worse.
- Report probability orientation clearly. LatentStrat reports `p_red_win`; blue win probability is `1 - p_red_win`.

## Attention And Zero-Out Diagnostics

PMA attention tables and zero-out diagnostics are architecture-specific review tools. V5 attention tables include missing-slot flags, while learned ghost slots remain visible to attention. They can show which slots influenced pooled alliance representations and how predictions respond when a slot is masked.

These artifacts are diagnostics, not causal proof. Use them alongside metrics, calibration, feature availability, and scouting context.

## Embeddings

Embedding inspection exports durable `Z_base` vectors, PCA projections, cosine neighbors, archetype similarities, PMA attention, and zero-out diagnostics. Use these to form hypotheses about learned structure:

- PCA can reveal broad clusters but compresses the full embedding space.
- Cosine neighbors can suggest teams with similar learned profiles.
- Archetypes depend on the chosen reference statistics.

Embedding stories should be supported by model metrics and source data. Do not treat embedding coordinates as direct scouting measurements.

## Review Checklist

Before claiming a model or feature change improved LatentStrat:

- Compare Brier score and log loss for binary targets.
- Review calibration bins and bin counts.
- Compare RMSE and MAE for continuous targets.
- Compare against mean and ridge match-OPR baselines.
- Review shuffled/null controls when available.
- Check availability slices for optional scouting or enrichment features.
- Confirm feature timing with the intended prediction point.

## Related

- [Metrics and artifacts](metrics-and-artifacts.md): common metric definitions and local artifact examples.
- [Experiment ledger](experiment-ledger.md): run-by-run evidence and design lessons.
- [Embedding inspection](embedding-inspection.md): latent-space diagnostics.
- [Model selection](model-selection.md): model comparison discipline.
- [Season training](season-training.md): walk-forward validation context.
