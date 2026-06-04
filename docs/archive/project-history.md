---
tags:
  - latentstrat
  - project-history
aliases:
  - "Project History"
  - "LatentStrat History"
related:
  - "[[changelog]]"
  - "[[experiment-ledger]]"
  - "[[current-state]]"
  - "[[prior-training]]"
  - "[[V5.7]]"
---

# Project History

This page records the main LatentStrat design experiments and what the local artifact evidence suggested. It is an engineering log, not a final paper.

For run-by-run artifact paths, metrics, and experiment outcomes, see the [Experiment Ledger](experiment-ledger.md).

For the Git-linked code timeline behind these milestones, see the [Changelog](changelog.md).

## V5.5: Text Autoencoder Prior

The older prior idea used text as a live input path:

```text
Text narrative -> OpenAI vector -> encoder -> 16D -> decoder -> OpenAI vector
```

The problem was runtime complexity. The live model should not need to process text strings to know who a team is.

Design lesson:

- Text is useful for pretraining.
- Team identity at match time should be a direct numeric lookup.

## V5.6: Transductive Team Dictionary

V5.6 changed the prior into a coordinate map:

```text
team_number -> embedding row -> decoder target
```

The model learns row `2290` directly for team `frc2290`. The decoder is discarded after training.

Major ideas:

- Generate one row per team number.
- Include known teams, historical gaps, and projected future teams.
- Export only the embedding table.

Early versions used a larger 20,000-row table. Later runs moved the practical default to 12,500 because the expected next-season team range did not require 20,000 rows.

## Ghost Token

Team `0` became a learned ghost robot instead of zero padding.

Reason:

- Missing robot slots still affect alliance dynamics.
- A learned ghost lets the Set Transformer always process the same number of slots.
- The model can learn what an empty or disabled slot means.

## Latent Dimension Sweep

The prior bottleneck experiment tested latent sizes such as:

```text
2, 4, 8, 16, 32, 64, 128, 256
```

The practical comparison that mattered most later was 8D versus 16D in walk-forward validation.

Local finding:

- `prior_v561_latent8` underperformed `prior_v561_latent16` in the available walk-forward comparison.

Design lesson:

- The default returned to 16D.

## V5.6.1: EPA Multi-Task Prior

V5.6.1 added Statbotics EPA as an additional target. The motivation was that OpenAI text embeddings encode flavor and context, while EPA encodes strength.

Design lesson:

- Text semantics and quantitative strength are complementary.

## V5.6.2: Cultural Prior

V5.6.2 added cultural targets:

- Rookie-year delta.
- Seasons played.
- Award count.
- Blue banner count.
- Championship appearance and win counts.
- Technical award count.

The goal was to make the Day Zero vector aware of institutional robustness, software history, and long-term team culture.

Problem found:

- The normalized EPA trajectory was not contributing useful gradients because the observed masks were effectively empty in the relevant table.

Design lesson:

- Prior training needs sanity checks that fail when an entire target family has no observations.

## V5.6.3: EPA Trajectory Fix

V5.6.3 fixed normalized EPA extraction and treated the four-year EPA trajectory as one grouped task.

Key change:

- Use Statbotics normalized EPA from the correct field.
- Train a grouped 4D trajectory with one EPA log variance.
- Fail before training if the feature table has zero observed EPA values.

Design lesson:

- Trajectory shape matters more than four unrelated scalar EPA tasks.

## V5.6.4: Feature-Summed OpenAI Loss

V5.6.4 changed the OpenAI loss reduction from per-element mean MSE to per-team vector squared distance.

Reason:

- A 256-dimensional OpenAI target can be underweighted if every dimension is averaged away.
- Summing across features restores narrative gradient strength.

Observed local artifact:

- `artifacts/prior_v564_latent16/inspection/v564_vs_v563_prior_summary.csv` compares the V5.6.3 and V5.6.4 latent spaces.
- V5.6.4 did not fully solve future/sibling/ghost-token near-duplicate clustering; that remains a current caveat.

Design lesson:

- Loss reduction matters as much as model shape in multi-task training.

## V5.7: Match-Spine Expansion

V5.7 expanded the match feature table with generic score-breakdown targets:

- Atomic counts.
- Committed fouls.
- Bonus binaries.
- Special binaries.

It also added relational sidecars:

- Rankings.
- Alliance selections.
- Playoffs.

Design lesson:

- Keep the match row as the spine.
- Keep post-event sidecars as auxiliary labels, not pre-match inputs.

## V5.7 Long-Run Collapse

A 100-epoch V5.7 full-season run showed overfitting behavior:

- Validation was best early.
- Training loss continued to improve.
- Validation loss later exploded.

Diagnosis:

- Sidecar tasks can be smaller and easier to memorize.
- Homoscedastic log variances can collapse into overconfidence without bounds.

## V5.7.2: Stability Patch

V5.7.2 added:

- Log-var clamping.
- Best-validation restoration.
- Cosine learning-rate decay.
- AdamW decay on non-embedding weights.

Design lesson:

- Long TensorBoard runs should be allowed, but saved checkpoints should restore the best validation state.

## V5.8: Walk-Forward Validation

V5.8 introduced canonical week-based validation:

```text
train weeks <= N
validate week N + 1
reset to prior for every fold
```

Week 0 is bundled into Week 1. Sidecars are pre-filtered by week.

Design lesson:

- Random validation is useful for smoke tests, but walk-forward validation is closer to real Friday-night prediction.

## Current Local Walk-Forward Evidence

The current comparison artifacts are summarized in the [Experiment Ledger](experiment-ledger.md). Current reading:

- V5.6.4 plus longer folded training improved score MSE and accuracy.
- Calibration got worse in that run, especially log loss.
- The next modeling question is not only "can it pick winners?" but "can it be well calibrated?"

## Open Questions

The next useful investigations are:

- Improve calibration without losing score prediction.
- Compare against explicit OPR, EPA, and pRidge baselines in-repo.
- Review whether sidecar task weights should be bounded differently.
- Inspect slices by week, event type, team archetype, and data availability.
- Decide when venue-mode fine-tuning should be used during live events.

## Related

- [Changelog](changelog.md): semantic versions tied to Git commits and unreleased artifact eras.
- [Experiment ledger](experiment-ledger.md): dense run table with artifact paths, outcomes, and design lessons.
- [Current state](current-state.md): current recommended stack and caveats.
- [V5.7 notes](V5.7.md): match-spine, sidecar, stability, and walk-forward notes for the V5.7/V5.8 transition.
- [Prior training](prior-training.md): detailed Day Zero prior process.
