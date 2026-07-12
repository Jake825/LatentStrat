# Scouting Normalization

Normalize values before constructing SQLModel records.

## Identifiers

- Normalize teams to `frc####` without source formatting such as `254.0`.
- Use official event and match keys, including real playoff set/match notation.
- Normalize alliance colors to exactly `red` or `blue`; reject unresolved required colors.
- Fail fast on missing primary keys with source row context.

## Values

- Convert blank strings, spreadsheet nulls, and NaN to `None` for optional fields.
- Parse common boolean forms explicitly and document source-specific labels.
- Parse spreadsheet numerics defensively, then validate integer and rating constraints.
- Record units for all measurements and convert only under an explicit mapping rule.
- Trim obvious whitespace/case variants while preserving meaningful free text.

## Duplicate Scouts

Choose and document one policy per target key: authoritative row, deterministic aggregation, or schema extension preserving individual observations. Never average categories or discard conflicts implicitly.

Use `assets/column-mapping-plan-template.md` to record raw columns, normalized values, target fields, timing, and rejection behavior.
