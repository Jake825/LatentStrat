# Scouting Schema Mapping

The schema source of truth is `src/frc/scouting.py`. Training reads Parquet; it does not query scouting SQLite directly.

| Model | Primary key | Grain | Feature prefix |
| --- | --- | --- | --- |
| `TeamScouting` | `team_key` | global team/pit | `{color}_team_{slot}_pit_` |
| `EventScouting` | `event_key` | event | `event_scout_` |
| `MatchScouting` | `match_key` | match | `match_scout_` |
| `TeamEventScouting` | `team_key`, `event_key` | team-event | `{color}_team_{slot}_event_scout_` |
| `MatchAllianceScouting` | `match_key`, `alliance_color` | alliance-match | `{color}_alliance_scout_` |
| `TeamMatchScouting` | `team_key`, `match_key` | team-match | `{color}_team_{slot}_match_scout_` |

Team-match rows also require a valid `alliance_color` for slot merging. Current fields and constraints must be read from the SQLModel classes rather than duplicated into importer code.

## Ownership Rules

- Use team/pit only for values independent of event and match.
- Use team-event for changing readiness or condition within an event.
- Use match for match-wide conditions independent of alliance or team.
- Use alliance-match for alliance strategy or coordination.
- Use team-match for one team's observed performance in one match.

If an analytically valuable field fits no current model, propose the SQLModel change, migration implications, importer behavior, Parquet prefix, documentation update, and tests. Use `assets/schema-change-checklist.md` for the proposal.
