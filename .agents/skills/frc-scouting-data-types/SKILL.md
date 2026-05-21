---
name: frc-scouting-data-types
description: Use when classifying FRC scouting data types for LatentStrat. Covers pit scouting, event context, match context, alliance strategy, team-event status, team-match performance observations, timing/leakage categories, and how each scouting concept maps to LatentStrat database tables.
---

# FRC Scouting Data Types

## Overview

Use this skill to classify what kind of scouting data a source contains before mapping it into LatentStrat.

## Core Questions

- What entity does one row describe?
- Is the data global, event-specific, match-specific, alliance-specific, or team-match-specific?
- Was the value known before the match, observed during the match, or recorded after the match?
- Does the value fit an existing LatentStrat scouting model?

## References

- Read [references/scouting-data-taxonomy.md](references/scouting-data-taxonomy.md) for the scouting taxonomy and table mapping.
- Use [assets/scouting-field-catalog-template.md](assets/scouting-field-catalog-template.md) to catalog fields from a new source.
- Use `$scouting-source-audit` for full source profiling.
- Use `$latentstrat-scouting-db` to verify current schema fields.
- Use `$statbotics` when mixing scouting fields with predictive model features.
- Use `$frc-time-aware-analysis` to classify whether scouting fields were known before a match, observed during it, or only available after it.
- Use `$frc-field-geometry` for spatial scouting fields such as shot locations, robot paths, heatmaps, pose estimates, field zones, or coordinate-frame labels.
