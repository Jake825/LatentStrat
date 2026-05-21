# Missing Data And Validation

FRC data is incomplete. Handle that incompleteness explicitly.

## Missing Source Data

Common gaps include:

- Missing or null TBA score breakdown fields.
- Unplayed matches or matches without final scores.
- Scouting rows missing for some teams or matches.
- Statbotics endpoints whose returned keys vary by version or endpoint.

Missing enrichment should usually become null feature columns on the TBA match spine, not dropped matches.

## Imputation Discipline

Do not fill missing values just to make a table look complete. Choose based on feature meaning:

- Use explicit missing indicators when missingness may carry signal.
- Use domain-safe defaults only when zero or false truly means "none observed."
- Keep unknown categorical values distinct from known negative values.
- Do not impute post-event aggregates into pre-match rows unless the `known_as_of` boundary permits it.

## Validation Checks

Before trusting a feature table:

- Check row count against the expected TBA match spine.
- Check that `match_key` remains unique.
- Check that all six team slot key columns are present.
- Check that model-training rows have nonmissing team index inputs after `make_team_index_map`.
- Check that target columns and feature columns match their timing assumptions.
- Run `git diff --check` for docs/skill edits and project-specific validators when runtime code changes.
