# Current State

Last audited locally: June 2, 2026.

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
| Experimental frozen targets | `artifacts/experimental/frozen-targets/` |
| Archived local outputs | `artifacts/archive/` |

New artifact writers emit `manifest.json`. Generated outputs remain ignored locally.

## Baseline Evidence

The tagged V5.8 replay remains the comparison boundary:

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

## Caveats

- Calibration remains the main season-model weakness.
- Match-breakdown artifacts are offline representation artifacts, not leakage-safe walk-forward
  promotion evidence.
- Match-breakdown runtime score attachment remains disabled.
- Frozen award, ranking, pick, and venue-mode workflows remain experimental.
- Flat CLI aliases and historical schema metadata readers are compatibility surfaces scheduled for
  V6.2 removal review.

See [V6.1](V6.1.md), [Architecture](architecture.md), and
[Evaluation and artifacts](evaluation-and-artifacts.md).
