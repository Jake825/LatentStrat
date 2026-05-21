# Source Audit Workflow

Use this workflow to profile a new scouting source before writing adapter code.

## Inspect the Source

- Record file format, sheet or table names, row count, column count, and header rows.
- Capture a small sample of raw rows without rewriting values.
- Detect merged headers, repeated title rows, footer rows, hidden totals, and empty spacer columns.
- Check whether the source contains one event, multiple events, one team, or multiple teams.

## Determine Row Grain

Classify each table or sheet into exactly one primary grain:

- Pit/team: one row per team, not tied to an event or match.
- Event: one row per event-wide condition.
- Match: one row per match, independent of teams.
- Team-event: one row per team at one event.
- Alliance-match: one row per red or blue alliance in one match.
- Team-match: one row per team in one match.

If a sheet mixes grains, split the mapping plan by row group. Do not write one adapter loop that produces ambiguous records.

## Identify Join Keys

Every import path must produce the keys required by its target table:

- `team_key`: TBA style, such as `frc254`.
- `event_key`: TBA style, such as `2026ilch`.
- `match_key`: TBA style, such as `2026ilch_qm12`.
- `alliance_color`: exactly `red` or `blue` where required.

If a source lacks a required key, document how to derive it. Use `$tba-api` to resolve official match schedules and alliance membership instead of guessing.

## Assess Data Quality

Check for:

- Duplicate primary keys after normalization.
- Missing teams, matches, or alliance colors.
- Values outside expected ranges, such as driver ratings outside 1 to 5.
- Inconsistent booleans, categorical labels, units, or free-text spelling.
- Multi-scout duplicate rows that need aggregation or source-preserving conflict handling.

## Classify Leakage Risk

Mark each field as one of:

- Static or pre-match: candidate model input.
- During-match or post-match: label, auxiliary target, diagnostic, or scouting artifact.
- Unknown timing: do not use as a model input until timing is clarified.

## Produce Recommendations

The final audit should include:

- Source summary and row-grain classification.
- Required key derivation plan.
- Column mapping to existing LatentStrat fields.
- Normalization rules needed for each mapped field.
- Data quality risks and explicit rejection rules.
- Schema extension proposal for important source-specific fields that do not fit the current schema.
- Adapter and test plan.
