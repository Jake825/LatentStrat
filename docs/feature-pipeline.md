# Feature Pipeline

LatentStrat turns FRC data into a typed match-grain Parquet table before any
PyTorch training starts. The feature table is the contract between provider
ingestion, scouting enrichment, and `train-features`.

## Match Table Spine

The TBA match table is the dataset spine. Each row is one played match with:

- `season`, `event_key`, `match_key`, `comp_level`, `set_number`, and
  `match_number`.
- Six team slot keys: `red_team_1_key` through `blue_team_3_key`.
- Raw targets such as scores, score differential, win flags, and
  score-breakdown-derived columns.
- Timing and ordering columns such as `event_week`, `time`, `actual_time`,
  `predicted_time`, and `sort_ordinal`.

Enrichment sources should left-join onto this spine. Scouting rows, Statbotics
rows, or external spreadsheets should not become the primary training row set
unless a future model intentionally changes the row grain.

## Scouting Joins

Optional scouting data lives in `data/scouting.db`. When present,
`build-features` joins SQLModel scouting tables onto the TBA spine and writes
prefixed columns to Parquet:

- `event_scout_` for event context.
- `match_scout_` for match context.
- `{color}_alliance_scout_` for alliance strategy rows.
- `{color}_team_{slot}_pit_` for pit scouting rows.
- `{color}_team_{slot}_event_scout_` for team-event scouting rows.
- `{color}_team_{slot}_match_scout_` for team-match scouting rows.

Team-level scouting joins must match the relevant team slot. Match-level rows
join by `match_key`; event-level rows join by `event_key`; team-event rows join
by `event_key` and `team_key`; team-match rows join by `match_key` and
`team_key`.

If a scouting source has useful fields that do not fit the current schema,
propose an explicit schema update, merge behavior, and tests instead of hiding
the fields in importer-only logic.

## Optional Statbotics Enrichment

Statbotics can be useful for external comparison or future feature work, but it
is not currently an implemented in-repo baseline. If enriching feature tables
with Statbotics data:

- Inspect returned keys before hard-coding EPA paths.
- Record whether a field is pre-match, post-match, current, or
  endpoint/version-dependent.
- Join through shared keys such as `match_key`, `event_key`, and team keys.
- Validate the feature timing before using the column for prediction.

## Parquet Boundary

`write_feature_table` validates the DataFrame, removes training-derived
`*_idx` and `*_z` columns, and writes with the `pyarrow` engine.
`read_feature_table` reads with the same engine.

Do not blanket-cast Parquet columns to `float32`. The persisted table should
preserve useful pandas dtypes, including nullable integer columns, timestamps,
strings, and numeric feature columns. Tensor-bound values are cast deliberately
when `MatchTensorDataset` builds PyTorch tensors.

## Team Indexing

Training maps string FRC team keys such as `frc254` to embedding indices in
memory:

1. `make_team_index_map` scans all six team slot columns.
2. Non-empty team keys are mapped to contiguous integers.
3. Nullable pandas `Int64` index columns are added for the current training run.
4. `MatchTensorDataset` converts those index matrices to `torch.long`.

The mapping is per training run. Unknown teams require rebuilding the mapping
with an updated feature table or a deliberate future inference policy. The
current model does not define a persistent public vocabulary or unknown-team
embedding.

## Missing Data

Missing enrichment data should usually remain as null columns on the TBA spine,
not cause official matches to be dropped. Choose imputation based on feature
meaning:

- Add missing indicators when missingness may carry signal.
- Use `0` or `False` only when that truly means "none observed."
- Keep unknown categorical values distinct from known negative values.
- Do not fill pre-match rows with post-match or post-event aggregates.

Training and evaluation require nonmissing team index inputs. Missing scouting
or optional feature values are a feature-design issue; missing team slots are a
model-contract issue.

## Commands

Build a season feature table:

```bash
latentstrat build-features --season 2026 --output data/features_2026.parquet
```

Train from that local Parquet file:

```bash
latentstrat train-features data/features_2026.parquet --output artifacts/features_run
```
