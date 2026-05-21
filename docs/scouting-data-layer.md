# Scouting Data Layer

LatentStrat uses SQLite and SQLModel for transactional scouting data. This layer
is separate from PyTorch training: scouting applications write normalized rows to
SQLite, then `build-features` merges those rows into a flat Parquet feature file.
Training reads only Parquet.

For importer examples and key-normalization rules, see
[Scouting Data Ingestion Guide](scouting-data-ingestion.md).

## Schema

The scouting schema mirrors TBA's key hierarchy:

- `team_scouting`: global team/pit scouting keyed by `team_key`.
- `event_scouting`: event context keyed by `event_key`.
- `match_scouting`: match context keyed by `match_key`.
- `team_event_scouting`: team weekend status keyed by `team_key + event_key`.
- `match_alliance_scouting`: alliance strategy keyed by `match_key + alliance_color`.
- `team_match_scouting`: six-scout match rows keyed by `team_key + match_key`.

Initialize the local database with:

```bash
latentstrat init-scouting-db --path data/scouting.db
```

`data/*.db` and SQLite sidecar files are ignored by Git.

## Feature Merge

When `build-features` runs, it looks for `data/scouting.db` by default. If the
database exists, available scouting rows are left-joined into the TBA match table
before Parquet is written:

```bash
latentstrat build-features --event-key 2026ilch --output data/features_2026ilch.parquet
```

If the database is missing, LatentStrat prints a message and writes TBA-only
features. Use `--no-scouting` to disable scouting lookup explicitly.

Merged columns are prefixed by scope, for example:

- `event_scout_carpet_condition`
- `match_scout_field_fault_occurred`
- `red_alliance_scout_coordinated_auto_run`
- `red_team_1_pit_drive_base`
- `red_team_1_event_scout_passed_inspection`
- `red_team_1_match_scout_teleop_pieces_scored`

## Leakage Discipline

Static data, such as pit scouting or team-event status, can become model inputs
in a future model version.

Dynamic match scouting metrics, such as teleop pieces scored, are post-match
observations. They are safe to store in Parquet and can become auxiliary targets,
but they should not be used as pre-match inputs.
