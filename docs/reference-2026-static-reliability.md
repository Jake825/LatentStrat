# 2026 Static Architecture Reliability Study

**Current evidence:** The first full development grid was rejected before test training because
no common epoch from 15-40 preserved score-differential RMSE, Brier score, and log loss across all
architectures. See the [2026 static training postmortem](2026-static-training-postmortem.md). The
project is paused before another architecture campaign.

This experiment asks one narrow question before LatentStrat adds event adaptation: do the
teammate- and opponent-interaction architectures improve out-of-time forecasts reliably, or were
the earlier one-seed results sensitive to optimization and random initialization?

The study keeps the representation deliberately fixed: one unconstrained 16-dimensional
`Z_base` row per 2026 team, initialized from the leakage-safe pre-2026 prior. It compares the
`additive`, `teammate-set`, and `full-match` architectures without `Z_event`, scouting inputs,
semantic latent partitions, or cross-season adapters.

## Evidence Contract

- Week 4 is development-only. It selects a shared gradient-clipping threshold and a shared fixed
  epoch count for every architecture.
- Seeds 2026, 2027, and 2028 are trained independently with that frozen configuration.
- Weeks 6 and 8 are the formal out-of-time tests. Week 10 is a championship stress test and is not
  allowed to rescue a failed primary result.
- Each final model trains once on all eligible matches through week `T-1`, freezes, and predicts
  week `T`. Test-week matches and completed-event auxiliary outcomes cannot update their own
  predictions.
- Raw win probabilities are the primary probability evidence. A secondary sequential calibrator
  may use only earlier held-out test predictions; it remains the identity transform until at least
  1,500 prior rows exist.
- Interaction uncertainty uses a hierarchical bootstrap that samples seeds and then event
  clusters. The durable output reports loss deltas, percentile intervals, and probability of
  improvement.
- A verified Statbotics pre-match artifact is required before the experiment can issue its final
  external-comparison verdict. Training may finish without it, but the result remains
  `awaiting-statbotics` and provisional.

The complete study remains `promotion_eligible=false`. It is one-season architecture evidence,
not a production-model promotion.

## Commands

Run the smoke gate first:

```powershell
uv run latentstrat season run-static-reliability-study `
  --features data/features/season/features_v58_2026.parquet `
  --events data/features/season/events_2026.parquet `
  --prior-checkpoint artifacts/pretraining/prior/prior_v564_latent16/checkpoint.pt `
  --output artifacts/smoke/2026-static-reliability-smoke `
  --tensorboard `
  --smoke
```

After the smoke artifacts, TensorBoard telemetry, and browser views pass review, use
`--full-study` for the fixed three-seed campaign. The runner measures development runtime, applies
a 20% safety margin, and refuses an experiment that cannot fit its CPU budget.

If Statbotics was unavailable during training, finalize the saved predictions later without
loading or fitting a PyTorch model:

```powershell
uv run latentstrat artifacts finalize-static-reliability `
  --study artifacts/reference/2026-static-reliability `
  --statbotics data/baselines/statbotics/statbotics_predictions_2026.parquet
```

Inspect the exact installed options with both commands' `--help` output before automation.

## Artifacts And Interpretation

The study directory contains the manifest and coverage contract, raw and calibrated saved
predictions, long-form metrics, calibration bins, seed-level paired bootstraps, hierarchical
bootstrap results, interaction classifications, external comparison and verdict, development
optimization/confirmation evidence, training history, parameter utilization, auxiliary metrics,
and fold checkpoints.

`supported` requires both practical and statistical evidence on the formal weeks; `harmful`
requires repeated practically worse results; otherwise the component is `uncertain`. An
interaction classification is provisional until the matrix is complete and the Statbotics
comparison has been finalized.

TensorBoard logs are isolated under `runs/reference-2026-static-reliability/<study-id>/`. Use it
for optimization, runtime, task activity, and `Z_base` diagnostics. Use the Streamlit Research
Workbench for completed prediction evidence, calibration, coverage, uncertainty, and failure
slices. Partial and smoke runs must remain visibly incomplete in both surfaces.
