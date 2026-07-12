---
name: latentstrat-scouting-ingestion
description: Audit, normalize, map, import, and validate FRC scouting CSV or spreadsheet data for LatentStrat's SQLModel SQLite database. Use for unfamiliar scouting sources, key cleanup, schema ownership, importer code, rerun behavior, and scouting-to-Parquet validation. Do not use for generic spreadsheets, TBA-only features, model architecture, or downstream metric analysis.
---

# LatentStrat Scouting Ingestion

Move scouting data from an understood raw source into the schema in `src/frc/scouting.py`, then verify its merge through `src/latentstrat/season/features.py`.

## Workflow

1. Audit the source and establish row grain, keys, missingness, duplicates, and timing.
2. Normalize identifiers and values without silently changing ambiguous source meaning.
3. Map fields to the existing SQLModel schema or define an explicit schema extension.
4. Implement an idempotent importer with `session.merge(...)` and focused tests.
5. Rebuild season features and verify prefixes, join grain, row count, and temporal eligibility.

## References

- Read `references/source-audit.md` before mapping an unfamiliar file.
- Read `references/normalization.md` for key and value conversion.
- Read `references/schema-mapping.md` for row grains, tables, keys, and merge prefixes.
- Read `references/importer-workflow.md` before writing importer code.
- Read `references/validation.md` before finalizing ingestion work.

Use the templates and Python skeletons in `assets/` only when they match the source format and requested deliverable.
