# Adapter Writing Workflow

Use this workflow when implementing an importer for a specific scouting source.

## Inputs

Require or create:

- Source audit report.
- Column mapping plan.
- Example source file or representative fixture.
- Target event key or derivation rule.
- Decision on schema extension fields.

## Implementation Pattern

- Read the source with Pandas or a format-specific parser.
- Normalize keys and values with explicit helpers.
- Build SQLModel records for the target table grain.
- Write records with `session.merge(...)`.
- Commit once per import unless the file is too large.
- Fail fast on missing primary keys, invalid alliance colors, or impossible match keys.

## Current Target Models

- `TeamScouting`: global pit/team data keyed by `team_key`.
- `EventScouting`: event-wide context keyed by `event_key`.
- `MatchScouting`: match-wide context keyed by `match_key`.
- `TeamEventScouting`: team state at an event keyed by `team_key` and `event_key`.
- `MatchAllianceScouting`: alliance strategy keyed by `match_key` and `alliance_color`.
- `TeamMatchScouting`: one team in one match keyed by `team_key` and `match_key`.

## Schema Extension Rule

When source fields do not fit the current schema:

- Do not silently drop important fields.
- Propose the SQLModel field or table change.
- Describe feature merge prefix impact.
- Add tests for table creation, importer writes, feature merge columns, and Parquet round trip.
- Update docs that describe scouting ingestion and the database layer.

## Validation

After implementation, run targeted tests first, then rebuild a small feature file:

```powershell
pytest tests/test_latentstrat_features.py
python -m latentstrat.cli build-features --event-key 2026ilch --output data/features/event/features_2026ilch.parquet
```

If live TBA access is unavailable, use focused unit tests with a synthetic feature table and temporary SQLite database.
