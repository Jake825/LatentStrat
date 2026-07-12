---
name: latentstrat-modeling
description: Develop and evaluate LatentStrat's season Set Transformer, training loop, losses, baselines, metrics, calibration, embeddings, attention, and diagnostics. Use for model-facing tensors, architecture or training changes, saved-prediction evaluation, and quality claims. Do not use for raw feature-table construction, provider access, scouting ingestion, or documentation-only work.
---

# LatentStrat Modeling

Use the canonical season modules under `src/latentstrat/season/` for model, training, evaluation, metrics, plots, and walk-forward behavior.

## Invariants

- Treat alliances as unordered sets of three slots and preserve permutation-tolerant behavior.
- Keep missing-slot routing distinct from diagnostic zero-slot ablation.
- Interpret continuous, binary, count, ordinal, and sidecar targets with their appropriate losses and metrics.
- Prefer calibration and proper scoring rules over raw accuracy for probabilities.
- Compare saved predictions on identical match keys and disclose selection or validation bias.
- Treat embeddings and attention diagnostics as evidence, not causal truth.

## References

- Read `references/architecture.md` for the forward pass and representation.
- Read `references/tensor-contracts.md` before changing model inputs or missing-team behavior.
- Read `references/training-and-regularization.md` for optimization and losses.
- Read `references/metrics-and-calibration.md` for evaluation metrics.
- Read `references/baselines-and-controls.md` for mean, ridge, and Statbotics comparisons.
- Read `references/artifact-guide.md` for prediction and diagnostic artifacts.
- Read `references/diagnostics.md` and `references/embedding-inspection.md` for model inspection.
