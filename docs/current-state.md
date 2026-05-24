---
tags:
  - latentstrat
  - current-state
aliases:
  - "Current LatentStrat State"
  - "Project Snapshot"
related:
  - "[[changelog]]"
  - "[[experiment-ledger]]"
  - "[[model-architecture-reference]]"
  - "[[metrics-and-artifacts]]"
---

# Current State

This page is the project snapshot for the current LatentStrat implementation. When model behavior, tensor dimensions, training defaults, or accepted experiments change, update this page with the change.

Last audited locally: May 24, 2026.

Local source sanity at audit time:

- `ruff check .`: passed.
- `pytest -q`: passed, `108 passed`.

## Version Stack

The current recommended stack is:

- **V5.6.4 prior**: a 16-dimensional Day Zero team coordinate map trained from OpenAI narrative vectors, Statbotics normalized EPA trajectory, and raw cultural targets.
- **V5.7.2 season training**: a stable Set Transformer training loop with homoscedastic log-var clamping, cosine learning-rate decay, AdamW decay on non-embedding weights, active-row embedding regularization, and best-validation restoration.
- **V5.8 validation**: walk-forward temporal validation over canonical `event_week`, with TBA raw Week 0 bundled into model-facing Week 1.

The current code treats the Python implementation as the source of truth. The deep documentation pages are intended to make that implementation readable without forcing a new contributor to start in the source code.

## Current Defaults

| Setting | Current value |
|---|---:|
| Prior latent dimension | `16` |
| Season latent dimension | `16` |
| OpenAI target dimension | `256` |
| Prior maximum team number | `12500` |
| Prior embedding rows | `12501` |
| Training mini-batch size | `256` |
| Attention heads | `1` |
| Set Transformer FFN dimension | `16` |
| Team dropout rate | `0.03` |
| Prior training epochs | `1000` |
| Default walk-forward fold epochs | `5` |
| Long local walk-forward fold epochs | `50` |

Training CLIs should write TensorBoard logs under `runs/` by default and expose `--no-tensorboard` for quiet runs.

## Recommended Local Artifacts

Use these as the current known-good reference artifacts unless a newer run is explicitly promoted:

| Purpose | Path |
|---|---|
| Current prior checkpoint | `artifacts/prior_v564_latent16/checkpoint.pt` |
| Current prior inspection | `artifacts/prior_v564_latent16/inspection/` |
| Current walk-forward run | `artifacts/v58_walk_forward_v564_latent16_50ep_2026/` |
| Current prior features used for V5.6.4 training | `data/prior_features_v563_2026.parquet` |
| Current season features | `data/features_v58_2026.parquet` |
| Current sidecars | `data/v58_sidecars_2026/` |

V5.6.4 is a training-loss/model update over the V5.6.3 prior feature schema. That is why the local V5.6.4 prior run uses `data/prior_features_v563_2026.parquet`.

## Current Local Evidence

The latest long walk-forward run is:

```text
artifacts/v58_walk_forward_v564_latent16_50ep_2026/
```

Its `AVERAGE` row reports:

| Metric | Value |
|---|---:|
| Next-match phase score MSE | `8953.20` |
| Next-match total score MSE | `9055.87` |
| Match accuracy | `0.7192` |
| Win Brier score | `0.2025` |
| Win log loss | `0.6796` |
| Validation matches | `18164` |

Compared with the earlier V5.6.1 16D 5-epoch folded run, this improved score MSE and match accuracy, but worsened calibration metrics. The current model is therefore better at point and winner discrimination in this local comparison, but not yet better calibrated.

## Known Caveats

- Calibration remains the biggest open model-quality issue. Brier score and log loss worsened in the 50-epoch V5.6.4 walk-forward run.
- V5.6.4 did not fully resolve the future/sibling/ghost near-duplicate clustering in prior inspection. The local comparison still shows very high cosine similarity among future/sibling/ghost-token rows.
- Statbotics EPA and pRidge are not implemented in-repo baselines. They are external comparison ideas until explicit runtime code is added.
- The current 2026 feature and sidecar row counts are local artifact evidence, not a permanent guarantee about all future seasons.

## Where To Read Next

- [Student primer](student-primer.md): plain-language explanation.
- [Changelog](changelog.md): semantic versions tied to Git commits and local artifact eras.
- [Model architecture reference](model-architecture-reference.md): exact tensor dimensions, parameter counts, and equations.
- [Experiment ledger](experiment-ledger.md): run-by-run artifact log.
- [Schemas and artifacts reference](schemas-and-artifacts-reference.md): generated data and output contracts.
- [CLI reference](cli-reference.md): current commands and their purpose.
