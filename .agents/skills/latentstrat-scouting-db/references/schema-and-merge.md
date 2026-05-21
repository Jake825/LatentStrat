# Schema and Feature Merge Reference

LatentStrat's scouting data layer is separate from training. Scouting applications or importers write normalized rows to SQLite, then `build-features` merges them into a flat Parquet feature file.

## Tables

| SQLModel | Table | Primary key | Purpose |
| --- | --- | --- | --- |
| `TeamScouting` | `team_scouting` | `team_key` | Global pit/team data |
| `EventScouting` | `event_scouting` | `event_key` | Event-wide conditions |
| `MatchScouting` | `match_scouting` | `match_key` | Match-wide context |
| `TeamEventScouting` | `team_event_scouting` | `team_key`, `event_key` | Team state at one event |
| `MatchAllianceScouting` | `match_alliance_scouting` | `match_key`, `alliance_color` | Alliance strategy in one match |
| `TeamMatchScouting` | `team_match_scouting` | `team_key`, `match_key` | One team's match observations |

## Current Fields

- `TeamScouting`: `drive_base`, `robot_weight_lbs`, `frame_dimensions_inches`, `programming_language`.
- `EventScouting`: `carpet_condition`, `venue_notes`.
- `MatchScouting`: `event_key`, `field_fault_occurred`, `audience_delay_minutes`, `referee_strictness_rating`.
- `TeamEventScouting`: `passed_inspection`, `is_functional`, `major_breakdown_notes`.
- `MatchAllianceScouting`: `coopertition_agreed`, `coordinated_auto_run`, `strategic_meltdown`.
- `TeamMatchScouting`: `alliance_color`, `auto_pieces_scored`, `teleop_pieces_scored`, `endgame_status`, `driver_ability_rating`, `played_defense`, `defended_by_team`, `scouter_name`.

## Merge Prefixes

Merged feature columns are prefixed by scope:

- `event_scout_`
- `match_scout_`
- `red_alliance_scout_` and `blue_alliance_scout_`
- `{color}_team_{slot}_pit_`
- `{color}_team_{slot}_event_scout_`
- `{color}_team_{slot}_match_scout_`

Team-match rows are filtered by `alliance_color` before slot merge. Invalid or missing alliance colors prevent expected feature columns from joining.

## Match Spine To PyTorch

Scouting rows are useful to PyTorch only after `build-features` joins them onto the TBA-derived match/team-slot spine and writes Parquet. Training reads that Parquet file; it does not query SQLite.

Keep the join grain explicit:

- Global team scouting joins each matching team slot.
- Team-event scouting joins by `event_key` and team slot.
- Team-match scouting joins by `match_key`, `team_key`, and `alliance_color`.
- Match and alliance scouting joins at match or alliance scope, then receives the documented prefixes.

Use `$latentstrat-feature-pipeline` for the broader feature-table contract and `$frc-time-aware-analysis` before treating a scouting field as a pre-match input.

## CLI Commands

Initialize the database:

```powershell
python -m latentstrat.cli init-scouting-db --path data/scouting.db
```

Build features with scouting data:

```powershell
python -m latentstrat.cli build-features --event-key 2026ilch --output data/features_2026ilch.parquet
```

Build features without scouting data:

```powershell
python -m latentstrat.cli build-features --event-key 2026ilch --no-scouting --output data/features_2026ilch.parquet
```

## Leakage Boundaries

- Pit/team and team-event fields are static or pre-match candidates.
- Match, alliance, and team-match performance fields are usually post-match observations.
- Dynamic match scouting should not become default pre-match model input without explicit timing controls.

## Schema Extension Expectations

Schema changes should include:

- SQLModel field or table update.
- Importer update.
- Feature merge behavior if the field should appear in Parquet.
- Docs update.
- Tests for database creation, importer behavior, merge output, and Parquet round trip.
