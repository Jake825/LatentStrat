# Scouting Ingestion Validation

Validate each boundary independently and then validate the end-to-end merge.

## Focused Checks

- Required keys normalize to official forms.
- Invalid keys, alliance colors, ranges, units, and ambiguous values fail clearly.
- Duplicate source rows follow the declared conflict policy.
- Re-running the importer updates existing primary keys without increasing row count.
- Schema creation and optional schema extensions work in a temporary SQLite database.
- Scouting merges preserve the TBA match-spine row count and produce expected prefixes and slot placement.
- Post-match observations remain excluded from earlier predictive inputs.
- Parquet round trips preserve intended dtypes and missingness.

Run targeted tests first, normally including `pytest tests/test_latentstrat_features.py`. For an end-to-end local check, initialize with `latentstrat scouting init` and build with `latentstrat season build-features`; inspect each command's `--help` for current required options.

When live provider access is unavailable, use synthetic match rows and a temporary scouting database.
