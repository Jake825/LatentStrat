---
name: latentstrat-model-evaluation
description: Use when evaluating LatentStrat training runs, model metrics, calibration, baselines, controls, evidence packets, attention diagnostics, zero-out diagnostics, embeddings, PCA projections, cosine neighbors, archetype reports, or whether a feature/model change improved quality. Covers current mean and ridge match-OPR baselines and prevents claiming a Statbotics baseline exists before explicit integration.
---

# LatentStrat Model Evaluation

Use this skill for interpreting LatentStrat model quality, artifacts, and diagnostics.

## Core Directives

1. For binary outcomes, prioritize calibration and probabilistic quality over raw accuracy. Use Brier score, log loss, and calibration bins.
2. State current baselines accurately. LatentStrat currently implements mean and ridge match-OPR baselines; Statbotics is an external comparison only after explicit integration.
3. Interpret continuous and binary targets separately. Do not collapse model quality into one score.
4. Use controls and slices to detect leakage, missing-data artifacts, and brittle improvements.
5. Treat embeddings, attention, and zero-out diagnostics as evidence, not ground truth.

## References

- Read `references/metrics-and-calibration.md` for regression metrics, Brier score, log loss, and calibration bins.
- Read `references/baselines-and-controls.md` for current baselines, shuffled/null controls, and external Statbotics comparisons.
- Read `references/embedding-inspection.md` for PCA embeddings, cosine neighbors, archetypes, and interpretation limits.
- Read `references/artifact-guide.md` for evaluation artifacts, evidence packets, availability slices, PMA attention, and zero-out diagnostics.

## Coordinate With Other Skills

- Use `$latentstrat-feature-pipeline` when a metric issue may come from feature construction, missing data, or leakage.
- Use `$pytorch-set-transformer` when interpreting architecture-specific outputs.
- Use `$frc-time-aware-analysis` when evaluating whether a backtest used only information known at prediction time.
