---
name: latentstrat-feature-pipeline
description: Use when building, reviewing, or documenting LatentStrat feature tables from TBA, Statbotics, scouting, or other FRC data sources into Pandas/PyArrow Parquet files for train-features. Covers the TBA match-table spine, left joins, team slot columns, make_team_index_map, dtype and tensor boundaries, missing data, split-safe normalization, and feature-table validation.
---

# LatentStrat Feature Pipeline

Use this skill for work that turns raw FRC data into the Parquet feature tables consumed by LatentStrat training.

## Core Directives

1. Treat the TBA match table as the dataset spine. Join enrichment data onto existing match rows and team slots; do not let scouting or external analytics create the primary row set.
2. Prefer Pandas/PyArrow-friendly extraction for ML workflows. Keep API pulls normalized into columns that can be validated and written to Parquet.
3. Preserve key grain. Match-level features join by `match_key`; event features join by `event_key`; team-event and team-match features must also align with a specific team slot.
4. Keep team keys as `frc####` strings in persisted Parquet. `make_team_index_map` creates contiguous nullable `Int64` index columns during training, and that mapping is per training run.
5. Do not blanket-cast Parquet columns to `float32`. Nullable ints, timestamps, strings, and categorical columns are intentionally preserved. Convert tensor-bound values deliberately at the PyTorch dataset boundary.
6. Preserve temporal integrity. Use `$frc-time-aware-analysis` for `known_as_of` rules and do not use post-match or post-event facts as pre-match features.

## References

- Read `references/match-spine-and-joins.md` for the feature-table spine, scouting merge prefixes, and join grain.
- Read `references/parquet-and-dtypes.md` for PyArrow boundaries, persisted dtypes, and tensor conversion rules.
- Read `references/team-indexing-and-tensors.md` for `make_team_index_map`, team slot tensors, and missing team constraints.
- Read `references/missing-data-and-validation.md` for incomplete sources, imputation discipline, and validation checks.

## Coordinate With Other Skills

- Use `$tba-api` for TBA keys, schedules, matches, rankings, awards, and raw score breakdowns.
- Use `$statbotics` for EPA interpretation and optional external enrichment.
- Use `$latentstrat-scouting-db` for SQLite scouting schema and merge prefixes.
- Use `$frc-time-aware-analysis` before deciding whether a feature was available at prediction time.
- Use `$pytorch-set-transformer` when changing tensor shapes or model-facing feature contracts.
- Use `$latentstrat-model-evaluation` when deciding whether a feature improved model quality.
