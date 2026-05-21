---
name: scouting-data-normalization
description: Use when normalizing messy FRC scouting data for LatentStrat ingestion. Covers TBA key normalization, match key construction, alliance colors, booleans, numeric parsing, categorical cleanup, missing values, units, multi-scout conflicts, and column mapping from varied team spreadsheets or CSV exports.
---

# Scouting Data Normalization

## Overview

Use this skill to turn messy scouting source fields into stable, typed values that can be written to LatentStrat's SQLModel scouting database.

## Core Rules

- Normalize identifiers into TBA-compatible keys before database writes.
- Preserve source meaning. Do not coerce ambiguous values silently.
- Treat source-specific fields as schema extension candidates when they are important and do not fit current models.
- Keep leakage timing attached to each normalized field.
- Prefer explicit conversion helpers over inline ad hoc parsing in adapters.

## References

- Read [references/normalization-rules.md](references/normalization-rules.md) for key, value, and conflict normalization rules.
- Use [assets/column-mapping-plan-template.md](assets/column-mapping-plan-template.md) to define source-to-LatentStrat mappings.
- Use `$tba-api` to verify official keys, playoff match keys, alliance colors, and event schedules.
- Use `$frc-scouting-data-types` to classify scouting fields before mapping.
- Use `$latentstrat-scouting-db` to confirm target fields and schema limits.
- Use `$frc-time-aware-analysis` to preserve `known_as_of` timing and prevent normalized fields from becoming leaky model inputs.
- Use `$frc-field-geometry` to normalize spatial scouting values into explicit units and coordinate frames for shot maps, paths, poses, heatmaps, and zone labels.
