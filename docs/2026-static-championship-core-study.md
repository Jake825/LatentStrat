---
tags:
  - latentstrat
  - experiment
  - static-state
  - championship
aliases:
  - "2026 Fixed 100 Epoch Championship Diagnostic"
related:
  - "[[2026-static-training-postmortem]]"
  - "[[experiment-ledger]]"
  - "[[current-state]]"
---

# 2026 Fixed-100-Epoch Static Championship Diagnostic

This non-promotable diagnostic asks one question:

> Under a fixed 100-epoch, fixed-loss static `Z_base` model, does direct official-total supervision materially reduce Championship score bias without harming relative-strength or probability quality, and do teammate or opponent interaction modules add useful information beyond additive team strength?

The experiment does not compare static and temporal state. Persistent bias would motivate a controlled time/event-context experiment; it would not prove that temporal state is either the cause or the solution.

## Matrix

The study crosses two score targets with three match architectures:

| Supervision | Architecture |
|---|---|
| Phase-core | Additive |
| Phase-core | Teammate-set |
| Phase-core | Full-match |
| Official-total-core | Additive |
| Official-total-core | Teammate-set |
| Official-total-core | Full-match |

Each leaf runs once for exactly 100 epochs on seed `2026`. There is no early stopping, best-checkpoint restoration, learned task weighting, event state, scouting input, or auxiliary objective. The fixed objective is standardized score MSE plus unweighted winner BCE plus normalized active-`Z_base` regularization, with all three weights fixed at `1.0`.

## Frozen evidence boundary

`split_manifest.parquet` freezes membership by exact match key before training. Development uses 15,373 earlier official matches and a 1,642-match holdout from 16 non-FIRST-Championship district or regional championship events. Final refits use all 17,029 eligible pre-Championship matches. The only final evaluation is the 1,119 matches in `2026arc`, `2026cur`, `2026dal`, `2026gal`, `2026hop`, `2026joh`, `2026mil`, and `2026new`; the 16 Einstein matches at `2026cmptx` are excluded.

Matches with an alliance DQ stay in the prediction table but do not contribute to training or primary metrics. Ties contribute to score training and score metrics but not winner training or probability metrics. Surrogates remain eligible and receive a separate slice. The expected primary Championship set is 1,108 non-DQ matches and 1,106 non-DQ, non-tied winner rows.

## Execution

Inspect the grouped CLI before running:

```powershell
uv run latentstrat season run-static-championship-diagnostic --help
```

The command defaults to a two-epoch smoke. The formal command is:

```powershell
uv run latentstrat season run-static-championship-diagnostic `
  --features data/features/season/features_v58_2026.parquet `
  --events data/features/season/events_2026.parquet `
  --prior-checkpoint artifacts/pretraining/prior/prior_v564_latent16/checkpoint.pt `
  --historical-reference artifacts/reference/2026-static-z-base/walk_forward_predictions.parquet `
  --output artifacts/reference/2026-static-championship-core `
  --epochs 100 `
  --budget-minutes 360 `
  --tensorboard `
  --tensorboard-logdir runs/2026-static-championship-core `
  --full-study
```

Formal execution requires a clean Git commit. A two-epoch timing pilot covers all six leaf configurations and refuses the campaign when the projected 1,200 formal epochs plus a 20% margin exceed 360 CPU minutes.

## Dashboard contract

TensorBoard uses one training run per phase, supervision mode, and architecture. Epoch is the only scalar step. The dashboard contains total and component losses, score and winner metrics, learning rate, gradient and clipping summaries, runtime, and two `Z_base` movement summaries. Configuration is text, and final HParams outcomes are written once after epoch 100. A small `hparams_metrics` child run is required on Windows because TensorBoard 2.20 cannot resolve an empty HParams metric group correctly.

The Research Workbench recognizes the study manifest and separates history, the exact-subset scoreboard, transferred calibration, bootstrap uncertainty, leave-one-division-out sensitivity, coverage, and provenance. Its labels say “fixed-100 diagnostic effect”; they do not claim best-achievable architecture performance.

## Evidence and report

The durable study directory contains the split and frozen contracts, six development and six refit histories, calibrators, epoch-100 Championship predictions, baselines, metrics, calibration, bias slices, event bootstraps, leave-one-division-out results, parameter utilization, and hashes. `report_data.json` drives one report whose bytes are copied to both the artifact directory and `docs/2026-static-championship-core-report.md`.

The formal report is intentionally absent until all 12 leaves are terminal. The implementation and two-epoch smoke do not answer the scientific question.
