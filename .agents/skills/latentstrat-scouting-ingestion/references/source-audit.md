# Scouting Source Audit

Profile an unfamiliar source before writing importer code.

## Inspect

- Record file type, sheets or tables, header rows, dimensions, and representative raw rows.
- Detect merged headers, title/footer rows, totals, hidden columns, empty spacers, and mixed events.
- Measure missingness, duplicate candidates, category variants, and numeric ranges.

## Establish Grain

Classify each table or row group as exactly one of:

- team/pit;
- event;
- match;
- team-event;
- alliance-match;
- team-match.

Split mixed-grain sheets into separate mapping paths.

## Establish Keys and Timing

- Identify or derive `team_key`, `event_key`, `match_key`, and `alliance_color` as required by the target table.
- Never guess official playoff match keys or alliance membership.
- Classify each field as pre-match, live/during-match, post-match, post-event, or unknown.
- Reject unknown-timing fields as predictive inputs until their availability is established.

## Deliverable

Produce a source summary, grain decision, key-derivation rules, column mapping, normalization rules, duplicate policy, rejection rules, schema-extension candidates, and test plan. Use `assets/source-audit-report-template.md` when a written report is requested.
