---
name: latentstrat-temporal-integrity
description: Design or audit LatentStrat known-as-of boundaries, temporal feature eligibility, walk-forward evaluation, live-event snapshots, and leakage controls. Use when time availability is the central question. Do not use automatically for every feature, scouting, provider, model, dashboard, or documentation task that merely contains dates.
---

# LatentStrat Temporal Integrity

Make predictions, simulations, and backtests honest about what was knowable at the decision time.

## Workflow

1. Define the prediction or decision timestamp and `known_as_of` boundary.
2. Classify each input as pre-match, live, post-match, post-event, or post-season.
3. Fit transforms and select checkpoints without using the reported evaluation period as hidden training or selection data.
4. Exclude final rankings, awards, playoff results, full-event aggregates, and later strength estimates from earlier predictions.
5. Label biased development comparisons separately from promotion-eligible evaluation.

## References

- Read `references/known-as-of.md` for feature eligibility and snapshots.
- Read `references/season-event-phases.md` for phase definitions.
- Read `references/leakage-patterns.md` before reviewing features, joins, simulations, dashboards, or validation splits.
- Read `references/source-links.md` for authoritative source locations.
