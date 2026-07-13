# 2026 Physics-Consistent Multitask Study

> Status: the corrected smoke passed, but the complete eight-cell timing pilot rejected formal training at `1,810.1` projected minutes with margin versus a 720-minute budget. See the [timing-gate report](2026-static-multitask-physics-timing-gate.md). No scientific result is available.

This non-promotable diagnostic asks:

> Does dense breakdown supervision, particularly when coupled through an exact differentiable 2026 score composer, improve the out-of-time Championship generalization of Full-match relative to Additive without harming official-score accuracy, relative-strength estimation, or winner probabilities?

The study keeps the static 16-dimensional `Z_base`, exact match-key boundary, optimizer, and 2026 Championship holdout from the completed fixed-100 diagnostic. It changes only the supervised objective and compares Additive with Full-match. It does not test temporal or event state.

## Matrix

Each cell runs seeds `2026`, `2027`, and `2028` for 100 development epochs.

| Arm | Supervision |
|---|---|
| `primary-only` | Direct official score and independent winner |
| `dense-breakdown` | Primary objectives plus atomic breakdown labels |
| `physics-consistent` | Dense objectives plus score, ordering, and winner consistency |
| `full-structured` | Physics objectives plus split-local ranking, playoff, and corrected robot-award probes |

The two architectures are `additive` and `full-match`. One common epoch is selected for each arm/architecture from the sealed 16-event DCMP holdout using the seed-mean, ridge-normalized score RMSE, differential RMSE, Brier, and log-loss index. Final models are reinitialized, fit on all eligible pre-Championship matches for that duration, and evaluated once on the eight Championship divisions. Einstein is excluded.

## Physics contract

The cached 2026 breakdown corpus provides seven hub-period counts, per-team autonomous and endgame tower status, alliance minor and major foul counts, and diagnostic bonus/status labels. The hard composer derives hub, tower, and opponent-awarded foul points. It must reproduce every eligible cached alliance score exactly before training can start.

The model predicts atomic nonnegative counts with `softplus`, categorical tower distributions, a direct official score, and an anti-symmetric winner logit. Redundant phase and total fields are not separate targets. The composed score is not optimized against the official score; atomic labels ground the breakdown path, while a bidirectional Smooth L1 consistency term couples it to the directly supervised score path. No consistency term uses `detach` or stop-gradient.

The positive zero-intercept score-to-winner scale is fit on clean development-training outcomes with 5% label smoothing. Smoothing is confined to this scale fit because official winner labels are perfectly separated by the sign of official score differential; without it, the logistic slope has no finite maximum-likelihood solution. The authoritative winner target and winner BCE are not smoothed.

All loss weights are fixed. DQ matches contribute no supervised or consistency loss and are excluded from primary metrics. Ties remain eligible for score and breakdown losses but not winner losses. All auxiliary outcomes are split-local labels, never predictive inputs.

## Command

Prepare and hash the data boundary without training:

```powershell
latentstrat season run-static-multitask-study `
  --study-dir artifacts/reference/2026-static-multitask-physics `
  --tensorboard-dir runs/2026-static-multitask-physics `
  --full-study `
  --prepare-only
```

Run the formal campaign only after committing that pre-training contract:

```powershell
latentstrat season run-static-multitask-study `
  --study-dir artifacts/reference/2026-static-multitask-physics `
  --tensorboard-dir runs/2026-static-multitask-physics `
  --max-wall-minutes 720 `
  --full-study
```

Use `--resume` only for exact phase-safe continuation. The runner refuses a dirty formal launch, verifies the split and physics hashes, runs a two-epoch timing pilot over every arm/architecture cell, and fails before formal training if the full development/refit projection plus 20% margin exceeds 12 hours.

## Evidence boundary

Championship is reused diagnostic evidence, and only three seeds are used. Results therefore remain `promotion_eligible=false`. Improvements require held-out or Championship evidence with practical effect size, hierarchical seed/division bootstrap support, consistent seed direction, and leave-one-seed/division stability. Lower training loss, later epoch selection, or lower head disagreement alone cannot establish rescued overfitting.

The generated `docs/2026-static-multitask-physics-report.md` becomes authoritative only after every required leaf reaches a terminal state. Until then, this page describes an experiment contract, not a result.
