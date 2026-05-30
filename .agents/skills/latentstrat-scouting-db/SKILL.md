---
name: latentstrat-scouting-db
description: Use when understanding or modifying LatentStrat's scouting SQLite database, SQLModel schema, feature merge behavior, scouting CLI commands, Parquet feature integration, merge prefixes, primary keys, and leakage boundaries for scouting data.
---

# LatentStrat Scouting Database

## Overview

Use this skill for repo-specific knowledge about LatentStrat's scouting database and how scouting rows become Parquet feature columns.

## Core Facts

- Scouting data is transactional SQLite via SQLModel.
- Training reads Parquet, not SQLite.
- `build-features` left-joins scouting tables into the TBA-derived match table when `data/scouting/scouting.db` exists.
- Current schema lives in `src/frc/scouting.py`.
- Feature merge logic lives in `src/latentstrat/features.py`.

## References

- Read [references/schema-and-merge.md](references/schema-and-merge.md) for current tables, primary keys, field names, merge prefixes, and validation commands.
- Use [assets/schema-change-checklist.md](assets/schema-change-checklist.md) before proposing schema edits.
- Use `$scouting-adapter-writer` when writing importers.
- Use `$frc-scouting-data-types` when deciding which table owns a field.
- Use `$statbotics` when adding predictive features and checking pre/post-match leakage.
- Use `$frc-time-aware-analysis` for repo-wide `known_as_of` boundaries and model-input eligibility beyond EPA-specific checks.
- Use `$latentstrat-feature-pipeline` when explaining how scouting rows become model-facing Parquet columns.
