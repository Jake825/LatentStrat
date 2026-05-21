---
name: tba-api
description: Use when working with, understanding, or using data from The Blue Alliance (TBA) API for the FIRST Robotics Competition (FRC), especially with the tbapy Python library. Covers retrieving, validating, normalizing, and analyzing FRC teams, events, matches, rankings, awards, alliances, and year-specific score breakdowns.
---

# The Blue Alliance API and FRC Data

## Overview

Use this skill to write reliable Python code against The Blue Alliance API through `tbapy`, especially in LatentStrat workflows that ingest or analyze FRC match data.

LatentStrat already depends on `tbapy==1.3.2` and uses `TBA_API_KEY` for live TBA ingestion. Preserve raw TBA payloads where practical because score breakdowns, rankings, playoff structures, awards, and advancement rules vary across seasons.

## Key Formats

TBA uses strict key formats. Normalize identifiers before joining scouting, model, or provider data.

- Team keys: `frc####` with no leading zeros, such as `frc254`.
- Event keys: `[year][event_code]`, such as `2026ilch` or `2024cmpop`.
- Match keys: `[event_key]_[comp_level][match_number]` or `[event_key]_[comp_level][set_number]m[match_number]`, such as `2026ilch_qm12` or `2026ilch_sf1m2`.
- Competition levels: `qm` for qualifications; `ef`, `qf`, `sf`, and `f` for playoff matches.
- District keys: `[year][district_abbrev]`, such as `2026fim` or `2024ne`.

## Core Practices

- Prefer narrow TBA pulls. Fetch the specific event, team, match, ranking, award, or alliance endpoint needed for the task.
- Preserve raw context. Keep raw JSON for complex objects such as match score breakdowns so downstream analytics can inspect unexpected fields.
- Use `TBA_API_KEY`, not `TBA_AUTH_KEY`, in LatentStrat examples and environment setup.
- Handle unplayed matches and incomplete data. Future or missing matches may have `None` for `actual_time` and `score_breakdown`, and `winning_alliance` may be an empty string.
- Use defensive `.get()` access for nested payloads. TBA can add fields mid-season and score fields differ by game.

## Versioning Guardrails

Never assume two seasons share the same `score_breakdown` schema. FRC releases a new game every year, so scoring elements, ranking point fields, foul fields, and bonus fields can change completely.

Before using newer seasons, uncommon fields, or recently added TBA API functionality, verify endpoint behavior against the current TBA API docs or official FIRST materials. For current-year tournament structure, prefer the official FIRST Game Manual and season materials.

## References

- Read [references/frc-concepts.md](references/frc-concepts.md) for FRC terminology, tournament structures, advancement systems, historical quirks, and score-breakdown context.
- Read [references/tbapy-patterns.md](references/tbapy-patterns.md) for Python patterns using `tbapy`, Pandas normalization, missing-data handling, and historical defensive guards.
- Use `$frc-competition-structure` when event type, district/regional semantics, multi-division events, Einstein/finals fields, offseason events, or analysis grain affect how TBA data should be interpreted.
- Use `$latentstrat-feature-pipeline` when TBA data is being shaped into Pandas/PyArrow Parquet feature tables for LatentStrat training.
