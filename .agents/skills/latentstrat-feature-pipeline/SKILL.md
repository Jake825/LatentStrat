---
name: latentstrat-feature-pipeline
description: Build or review LatentStrat season feature-table code that turns FRC match data into Pandas/PyArrow Parquet and model-ready tensors. Use for the match spine, joins, team slots, dtypes, indexing, missingness, and split-safe transformations. Do not use for provider-only API work, scouting importer/schema work, model internals, metric interpretation, or documentation-only tasks.
---

# LatentStrat Feature Pipeline

Work from the canonical season pipeline in `src/latentstrat/season/features.py` and `src/latentstrat/season/data.py`. Treat `src/latentstrat/features.py` and `src/latentstrat/data.py` as deprecated V6.1 aliases.

## Invariants

- Preserve the TBA match table as the row spine and enrich it with left joins at explicit match, event, team-event, or team-match grain.
- Persist team keys as `frc####` strings; create run-local integer mappings only at the training boundary.
- Preserve nullable integers, strings, timestamps, masks, and categorical values in Parquet. Convert tensor-bound values deliberately.
- Keep missingness observable and validate all six team slots.
- Fit normalization and other learned transforms on training data only.
- Reject post-match or post-event facts as inputs to earlier predictions.

## References

- Read `references/match-spine-and-joins.md` for row grain and joins.
- Read `references/parquet-and-dtypes.md` for persisted schema and tensor conversion.
- Read `references/team-indexing-and-tensors.md` for team maps and slot contracts.
- Read `references/missing-data-and-validation.md` for missingness and checks.
