# Match Spine And Joins

Use the TBA match table as the primary row set for LatentStrat ML features.

## Spine Columns

The feature table is match-grain. The current TBA-derived spine includes:

- `season`, `event_key`, `match_key`, `comp_level`, `set_number`, `match_number`.
- Three red team slots and three blue team slots as `red_team_1_key` through `blue_team_3_key`.
- Match targets such as alliance scores, score differential, win flags, and score breakdown-derived targets.
- Timing and ordering columns such as `event_week`, `time`, `actual_time`, `predicted_time`, and `sort_ordinal`.

Do not replace this spine with scouting rows or Statbotics rows. Enrichment sources are joined onto this table.

## Join Rules

- Use `match_key` for one-row-per-match sources.
- Use `event_key` for event-level sources.
- Use `event_key` plus `team_key` for team-event sources.
- Use `match_key` plus `team_key` for team-match scouting or per-team performance observations.
- Join team-level rows by comparing `team_key` to each alliance slot, not by exploding the match row into six separate training rows unless a future feature design explicitly requires that grain.

Prefer left joins from the TBA spine so missing scouting or external analytics does not drop official matches.

## Scouting Merge Prefixes

LatentStrat's scouting merge currently prefixes features by table and slot:

- `event_scout_` for event context.
- `match_scout_` for match context.
- `{color}_alliance_scout_` for alliance strategy rows.
- `{color}_team_{slot}_pit_` for pit scouting rows.
- `{color}_team_{slot}_event_scout_` for team-event scouting rows.
- `{color}_team_{slot}_match_scout_` for team-match scouting rows.

Keep these prefixes stable when documenting or reviewing feature tables. If a new scouting concept does not fit the current schema, propose a schema change and merge behavior rather than silently dropping it.

## Optional External Enrichment

Statbotics data can be an external feature source and also has a separate saved-prediction baseline workflow. Do not confuse baseline predictions with required model inputs. If using Statbotics as enrichment:

- Inspect returned keys before hard-coding EPA paths.
- Join by shared keys such as `match_key`, `event_key`, and `team`.
- Preserve whether a field is pre-match, post-match, current, or endpoint/version-dependent.
- Validate every field against the prediction timestamp before joining it.
