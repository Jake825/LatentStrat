---
name: scouting-source-audit
description: Use when profiling unknown FRC scouting files before ingestion into LatentStrat. Covers identifying row grain, source columns, key fields, data quality issues, duplicate rules, leakage risk, mapping candidates, and schema extension recommendations for messy CSV, spreadsheet, or exported scouting data.
---

# Scouting Source Audit

## Overview

Use this skill before writing importer code for a new scouting source. The goal is to understand what the file contains, what each row represents, which LatentStrat scouting table it maps to, and what risks need to be resolved.

## Workflow

1. Inspect the file shape, sheet names, headers, row counts, sample values, and missingness.
2. Identify row grain: pit/team, event, match, alliance-match, team-event, or team-match.
3. Identify key fields needed for LatentStrat joins: `team_key`, `event_key`, `match_key`, and `alliance_color`.
4. Classify fields by scouting type and leakage timing.
5. Produce a mapping plan and list any source-specific fields that require schema edits.

## References

- Read [references/source-audit-workflow.md](references/source-audit-workflow.md) for the audit checklist and decision rules.
- Use [assets/source-audit-report-template.md](assets/source-audit-report-template.md) as the output structure for audit reports.
- Use `$frc-scouting-data-types` to classify scouting concepts.
- Use `$scouting-data-normalization` for key and value normalization rules.
- Use `$latentstrat-scouting-db` to confirm current SQLModel tables and merge prefixes.
- Use `$tba-api` when resolving match keys, alliance colors, schedules, or official team/event identifiers.
- Use `$frc-time-aware-analysis` when deciding whether fields are pre-match inputs, live-event state, post-match observations, or post-event summaries.
