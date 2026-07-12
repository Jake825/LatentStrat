# 2026 Static Robot-State Reference Study

**Current evidence:** The completed local run and its score errors are analyzed in the
[2026 static training postmortem](2026-static-training-postmortem.md). The reference remains
useful as a fixed baseline, but it is not evidence for promotion or a new architecture.

This study is the deliberately simple reference point for future LatentStrat architectures. It
uses one unconstrained 16-dimensional `Z_base` for each 2026 team. Event keys define temporal
boundaries and reporting grain but are not learned inputs. There is no `Z_event`, program/season
hierarchy, scouting encoder, frozen world-model target, semantic latent partition, or cross-season
adapter.

The reference vocabulary is compact: row 0 is the learned ghost and one row is allocated for each
2026 team in the study table. Prior initialization copies each team's pre-2026 vector by FRC team
number into that compact row; it does not carry thousands of unused prior-table rows into the
parameter count.

Supervision uses direct 2026 alliance score, auto, teleop, tower, atomic-count, bonus,
special-rule, and foul fields; per-team auto and endgame statuses; match winner; and past-only
official ranking, selection, playoff, and judged-award outcomes.

The study compares three otherwise matched architectures:

| Model | Teammate interaction | Opponent interaction |
|---|---:|---:|
| `additive` | No | No |
| `teammate-set` | Shared SAB and PMA | No |
| `full-match` | Shared SAB and PMA | Cross-alliance attention |

All variants retain permutation-tolerant three-team alliances, learned ghost routing, shared
red/blue heads, and an anti-symmetric red-win logit. Static checkpoints use schema `7`; diagnostic
loading is strict and does not synthesize a missing event table.

## Temporal Protocol

Initialization is selected on development evidence only: train through week 3, evaluate week 4,
and compare a leakage-safe pre-2026 prior with seeded random initialization. The selection order is
score-differential RMSE, Brier score, log loss, then official auxiliary probe loss.

For test week `T`:

1. Train through `T-2` and select the epoch on `T-1`.
2. Discard those weights.
3. Reinitialize and refit through `T-1` for exactly the selected number of epochs.
4. Freeze and predict `T` once.

The full reference uses test weeks 6, 8, and 10 with seed 2026. It fails closed when canonical week
metadata is missing, when non-official or unplayed rows reach training, or when a full-study test
week has fewer than 500 played matches across five events. Auxiliary event outcomes become
eligible only after their event is complete and never become match inputs.

## Run The Acceptance Smoke

```powershell
latentstrat season validate-reference-2026 `
  --features data/features/season/features_2026.parquet `
  --events data/features/season/events_2026.parquet `
  --prior-checkpoint artifacts/pretraining/prior/prior_v564_latent16/checkpoint.pt `
  --statbotics data/baselines/statbotics/statbotics_predictions_2026.parquet `
  --output artifacts/reference/2026-static-smoke `
  --epochs 2 `
  --smoke
```

`--smoke` is the default. It caps training at two epochs and one test fold but still exercises both
initializations, all architectures, checkpoint round trips, artifacts, Streamlit, and TensorBoard.
Its manifest is incomplete and not valid architectural evidence. Pass `--full-study` only after the
smoke dashboards pass and the timing estimate fits the 100-minute CPU budget.

## Artifact And Dashboard Contract

The directory contains `manifest.json`, long-form `metrics.csv`, `auxiliary_metrics.csv`,
`calibration.csv`, `paired_bootstrap.csv`, `coverage.json`,
`walk_forward_predictions.parquet`, `training_history.csv`, `parameter_utilization.csv`, the
development initialization table, and per-fold checkpoints with hashes.

Final scoreboards include mean and unweighted ridge controls, a fold-local recency-weighted rolling
pRidge control, and Statbotics when a verified pre-match artifact is supplied. When Statbotics is
present, every model's comparative metrics are restricted to the same exact `match_key`
intersection; unmatched rows remain in coverage and the durable prediction table.

Streamlit separates development selection from final evidence; presents metric direction,
calibration counts and ECE, coverage, uncertainty, official auxiliary probes, parameter
utilization, and saved predictions by variant; and suppresses event trajectories for static
checkpoints. Incomplete studies never receive interaction conclusions.

TensorBoard logs are isolated under:

```text
runs/reference-2026-static/<study-id>/
  development/<initialization>/
  <architecture>/fold_<test-week>/selection/
  <architecture>/fold_<test-week>/refit/
```

Each run records the resolved contract, losses and active counts for every task, score and winner
metrics, calibration error, optimizer groups, gradient/clipping statistics, runtime, parameter
utilization, and `Z_base` norms and update distances. Weight and gradient histograms are bounded and
written only at initialization and the final selected/refit state.

The resulting evidence remains `promotion_eligible=false`: one season and one seed can establish a
reference, not a production promotion decision. PCA and neighbors remain post-training diagnostics.
