# Current State

Last audited locally: July 12, 2026.

LatentStrat V6.1 is a stabilization milestone. The supervised season model remains schema-`6`
compatible, while public package names, CLI commands, generated paths, and artifact manifests are
organized for continued pretraining work.

## Recommended Stack

- V5.6.4 Day Zero prior: 16D team base embeddings from narrative, normalized EPA trajectory, and
  cultural targets.
- V5.8 tagged baseline: annotated tag `v5.8-baseline` and tracked
  `baselines/v5.8-baseline.json`.
- V6.1 season model: supervised Set Transformer with `Z_base + Z_event`, existing readout heads,
  nested forward outputs, and strict schema-`6` checkpoint compatibility.
- 2026 static reference path: schema-`7` checkpoints with one 16D `Z_base`, three controlled match
  architectures, nested temporal selection/refit, isolated TensorBoard, and Workbench review. It
  is a research reference and does not replace the schema-`6` default.
- Static reliability follow-up: the smoke path completed, but the first full week-4 development
  grid rejected every common duration from epochs 15-40 after selecting clip `5`. No three-seed
  test folds ran. Its failure analysis is preserved in the
  [2026 static training postmortem](2026-static-training-postmortem.md).
- Fixed-100 Championship diagnostic: all six development and six final-refit trajectories completed
  at 100 epochs. Direct official-total supervision modestly improved the additive static model,
  while both interaction architectures were harmful relative to additive strength at this fixed
  horizon. The remaining `-56.6`-point additive bias and early development optima make this
  non-promotable evidence. See the
  [authoritative report](2026-static-championship-core-report.md).
- Physics-consistent multitask follow-up: the corrected smoke completed, including nonzero ranking,
  playoff, and award probes and the isolated TensorBoard contract. The formal eight-cell timing
  pilot projected `1,810.1` minutes with its 20% margin, so the runner rejected the campaign before
  all 24 development trajectories. No Championship predictions or scientific result exist. See the
  [timing-gate report](2026-static-multitask-physics-timing-gate.md).
- CPU-first PyTorch runtime: shared seeded optimizer steps, accumulation, clipping, cosine or one-cycle scheduling, TensorBoard telemetry, and epoch-exact resume checkpoints across training workflows.
- Match-breakdown V1: offline all-years 16D alliance-result artifact.
- Match-breakdown V2: explicit structured-objective ablation. It is implemented but not promoted.

## Canonical Paths

| Purpose | Path |
|---|---|
| Prior features | `data/features/pretraining/prior/` |
| Match-breakdown corpus | `data/pretraining/match-breakdown/corpus.sqlite` |
| Match-breakdown features | `data/features/pretraining/match-breakdown/` |
| Prior artifacts | `artifacts/pretraining/prior/` |
| Match-breakdown artifacts | `artifacts/pretraining/match-breakdown/` |
| Season artifacts | `artifacts/season/` |
| Walk-forward artifacts | `artifacts/walk-forward/` |
| 2026 static training postmortem | `artifacts/reference/2026-static-training-postmortem/` |
| Fixed-100 Championship diagnostic | `artifacts/reference/2026-static-championship-core/` |
| Physics-consistent multitask study | `artifacts/reference/2026-static-multitask-physics/` |
| Experimental frozen targets | `artifacts/experimental/frozen-targets/` |
| Archived local outputs | `artifacts/archive/` |

New artifact writers emit `manifest.json`. Generated outputs remain ignored locally.

## Historical Baseline Evidence

The tagged V5.8 replay remains the reproducible historical comparison boundary:

```text
artifacts/baselines/v5.8/walk-forward/
```

Its average row reports:

| Metric | Value |
|---|---:|
| Next-match phase score MSE | `8829.77` |
| Next-match total score MSE | `8971.60` |
| Match accuracy | `0.7164` |
| Win Brier score | `0.2090` |
| Win log loss | `0.7227` |
| Validation matches | `18164` |

These values are development evidence rather than an unbiased promotion estimate. Each historical fold restored the epoch with the best loss on the same held-out week used for its reported predictions. The metric-only Statbotics comparison workflow can evaluate those saved predictions without retraining, but it marks the result as ineligible for promotion.

## Caveats

- Calibration remains the main season-model weakness.
- The first full static-reliability development grid selected clip `5` but rejected every common
  duration from epochs 15-40: score-differential and probability metrics peaked at incompatible
  times for the interaction models. No formal test-week folds were run from that rejected setup.
- The completed static reference systematically underpredicted alliance scores in weeks 6, 8, and
  10. The measured causes and remaining hypotheses are separated in the postmortem; this evidence
  does not justify a larger model or a temporal-state architecture by itself.
- The fixed-100 Championship diagnostic still underpredicted official alliance totals by `-56.6`
  points in its best neural configuration. Official-total additive improved score RMSE by `2.76%`
  over phase-core additive, but rolling pRidge remained slightly better on score, differential, and
  calibrated probability metrics. Interaction-model development metrics peaked around epochs
  3-7 and deteriorated substantially by epoch 100 while training loss kept falling.
- The physics-consistent multitask study is diagnostic, not promotion evidence. It reuses the same
  Championship divisions and cannot establish whether temporal state is necessary or sufficient.
  Its formal matrix has not run because the conservative local CPU estimate exceeded the fixed
  12-hour budget.
- The runtime overhaul changes optimizer trajectories; it is implemented and regression-tested, but empirical promotion still requires the documented multi-seed calibration and non-inferiority comparison.
- Honest temporal evaluation is implemented for the static 2026 reference path. The reliability
  follow-up stopped at its development gate, and the Championship diagnostic used one seed and a
  reused holdout, so multi-seed promotion evidence is still absent. The next controlled experiment
  first tests whether structured supervision rescues interaction generalization. A later controlled
  study should compare event or week context against the retained static control.
- Match-breakdown artifacts are offline representation artifacts, not leakage-safe walk-forward
  promotion evidence.
- Match-breakdown runtime score attachment remains disabled.
- Frozen award, ranking, pick, and venue-mode workflows remain experimental.
- Flat CLI aliases and historical schema metadata readers are compatibility surfaces scheduled for
  V6.2 removal review.

See [V6.1](V6.1.md), [Architecture](architecture.md), and
[Evaluation and artifacts](evaluation-and-artifacts.md).
