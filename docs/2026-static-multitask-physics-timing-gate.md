# 2026 Physics-Consistent Multitask Timing Gate

## Verdict

The formal 2026 Physics-Consistent Multitask Interaction Study did not start. Its predeclared timing gate projected `1,508.4` minutes of execution, or `1,810.1` minutes after the required 20% safety margin. This exceeds the fixed 720-minute local CPU budget by `1,090.1` minutes.

The rejection happened before any of the 24 formal development trajectories. No selected epochs, final refits, Championship predictions, comparison classifications, or scientific final report were produced. The declared report question therefore remains unanswered.

## Accepted pre-training evidence

- Implementation commit: `d11962e`.
- Committed pre-training boundary: `0780450`.
- Exact physics audit: `36,328` played alliance breakdowns, maximum score-composition error `0`.
- Development train: `15,373` matches, `15,029` primary-clean.
- Sealed DCMP holdout: `1,642` matches, `1,630` primary-clean.
- Final train: `17,029` matches, `16,673` primary-clean.
- Championship test: `1,119` matches, `1,108` primary-clean.
- Score-to-winner consistency scale: `alpha=0.0268568`, `sigma_logit=3.88953`, fit on `14,986` clean development-training winners with 5% smoothing confined to the scale fit.
- The corrected smoke completed all eight development and eight refit leaves at two epochs. Its TensorBoard hierarchy had 16 primary runs, 16 HParams records, one value per tag per epoch, every active loss family, nonzero award-probe counts, and no browser-console errors.

## Formal timing pilot

Every pilot cell used seed `2026`, two epochs, the exact development split, batch size `256`, AdamW, cosine scheduling, and clipping at `5`. The pilot measured training epochs and fixed leaf work such as preparation, checkpointing, prediction generation, and artifact writing.

| Arm | Architecture | Mean epoch seconds | Fixed leaf seconds | Total two-epoch seconds |
|---|---|---:|---:|---:|
| `primary-only` | Additive | `12.805` | `50.382` | `75.992` |
| `primary-only` | Full-match | `18.046` | `47.236` | `83.327` |
| `dense-breakdown` | Additive | `9.472` | `44.939` | `63.882` |
| `dense-breakdown` | Full-match | `17.579` | `52.909` | `88.067` |
| `physics-consistent` | Additive | `8.341` | `28.763` | `45.445` |
| `physics-consistent` | Full-match | `28.546` | `55.031` | `112.122` |
| `full-structured` | Additive | `9.866` | `28.409` | `48.140` |
| `full-structured` | Full-match | `34.973` | `61.309` | `131.254` |

The projection conservatively covers all 24 100-epoch development leaves and all 24 possible 100-epoch refits because selected durations are unknowable before development. This is the only pre-run estimate that guarantees the full allowed matrix fits the budget.

## Thread-count diagnostic

The default PyTorch process used 16 intra-op threads. Bounded probes on the slowest `full-structured/full-match` cell showed:

| Threads | Mean epoch seconds | Total two-epoch leaf seconds |
|---:|---:|---:|
| `4` | `23.101` | `98.255` |
| `8` | `19.553` | `86.796` |
| `16` | `34.973` | `131.254` |

Eight threads materially reduce oversubscription for the worst cell, but that isolated improvement is insufficient to demonstrate that the entire worst-case 48-leaf matrix fits 12 hours. Changing the runtime and replacing the predeclared full-matrix gate after seeing the failure would require a new committed contract and timing campaign.

## What was learned

Confirmed:

- The exact physics composer, masking, objective arms, corrected award candidates, resume path, durable artifacts, and TensorBoard profile work end to end in smoke.
- The originally unregularized score-to-winner scale was mathematically ill-posed because official winners are perfectly separated by official score differential. The pre-training gate caught and corrected it before formal training.
- CPU thread oversubscription is material for the small Full-match model.
- Under the committed runtime and conservative completion rule, the requested study cannot fit the 12-hour budget.

Not learned:

- Whether dense breakdown targets improve Championship generalization.
- Whether physics consistency improves forecasts or merely aligns heads.
- Whether corrected competition and award probes help.
- Whether Full-match is rescued relative to Additive.
- Whether larger supervision delays overfitting.

## Decision required

A new experiment contract must choose one of three paths before training:

1. Preserve all arms, seeds, and horizons but use faster hardware or a larger wall-time budget.
2. Preserve the 12-hour CPU budget but explicitly reduce the matrix or maximum refit assumption.
3. Perform and commit a runtime-only optimization study, then repeat the complete eight-cell timing gate with the optimized thread/controller settings.

No option should be inferred silently. The existing pre-training sidecars and hashes remain valid evidence; the formal study remains paused and `promotion_eligible=false`.
