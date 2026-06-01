---
tags:
  - latentstrat
  - feature-pipeline
  - data-sources
aliases:
  - "Feature Pipeline"
  - "Feature Builder"
related:
  - "[[data-sources]]"
  - "[[schemas-and-artifacts-reference]]"
  - "[[season-training]]"
  - "[[scouting-data-layer]]"
---

# Feature Pipeline

For a guided overview of where each provider fits, see [Data Sources](data-sources.md) and the [Documentation Hub](index.md).

LatentStrat turns FRC data into a typed match-grain Parquet table before any PyTorch training starts. The feature table is the contract between provider ingestion, scouting enrichment, and `train-features`.

## Match Table Spine

The TBA match table is the dataset spine. Each row is one played match with:

- `season`, `event_key`, `match_key`, `comp_level`, `set_number`, and `match_number`.
- Six team slot keys: `red_team_1_key` through `blue_team_3_key`.
- Raw targets such as red/blue auto points, teleop points, score differential, win flags, per-slot 2026 endgame state, and score-breakdown-derived columns.
- Timing and ordering columns such as `raw_event_week`, canonical `event_week`, `time`, `actual_time`, `predicted_time`, and `sort_ordinal`.

Enrichment sources should left-join onto this spine. Scouting rows, Statbotics rows, or external spreadsheets should not become the primary training row set unless a future model intentionally changes the row grain.

## V5.7 Score-Breakdown Targets

V5.7 adds a versioned mapper layer between TBA's season-specific `score_breakdown` JSON and the model-facing Parquet schema. The 2026 mapper writes generic columns so future PyTorch code does not depend on game-specific names:

- Atomic count targets such as `red_atomic_auto_count`, `red_atomic_shift2_count`, and `blue_atomic_endgame_count`.
- Committed foul targets such as `red_committed_foul_pts`; TBA foul points are awarded to the opponent, so these columns are intentionally inverted.
- Bonus binary targets such as `red_bonus_energized`, `red_bonus_supercharged`, and `red_bonus_traversal`.
- Special binary targets such as `red_special_g206_penalty`.

If an official match is missing its score breakdown, the match row remains on the spine and the V5.7 targets are written as `NaN`. Training masks those values instead of imputing false zeros.

## V5.8 Temporal Weeks

`event_week` is the model-facing canonical season week used for walk-forward validation. TBA's raw `week` is preserved as `raw_event_week`; canonical `event_week` starts at `1`, so raw Week 0 is bundled into Week 1. If a raw week is missing, LatentStrat falls back to dense chronological event order using event date/sort metadata. Sidecar Parquet files receive the same `event_week` by joining on `event_key`.

## V5.7 Relational Sidecars

Rankings, alliance selections, and playoff outcomes have post-event or event-level grain, so they are written as optional sidecar Parquet files rather than merged into match rows:

- `rankings_YYYY.parquet`: `event_key`, `team_key`, `qual_rank`, `matches_played`, and available TBA ranking stats.
- `selections_YYYY.parquet`: `captain_team_key`, `pick_team_key`, `pick_order`, and `passed_over_team_key`. Passed-over teams are computed from rankings during sidecar generation.
- `playoffs_YYYY.parquet`: alliance team keys and `playoff_finish_order`.

Sidecars are auxiliary training labels only; they are not pre-match input features.

## Scouting Joins

Optional scouting data lives in `data/scouting/scouting.db`. When present, `build-features` joins SQLModel scouting tables onto the TBA spine and writes prefixed columns to Parquet:

- `event_scout_` for event context.
- `match_scout_` for match context.
- `{color}_alliance_scout_` for alliance strategy rows.
- `{color}_team_{slot}_pit_` for pit scouting rows.
- `{color}_team_{slot}_event_scout_` for team-event scouting rows.
- `{color}_team_{slot}_match_scout_` for team-match scouting rows.

Team-level scouting joins must match the relevant team slot. Match-level rows join by `match_key`; event-level rows join by `event_key`; team-event rows join by `event_key` and `team_key`; team-match rows join by `match_key` and `team_key`.

If a scouting source has useful fields that do not fit the current schema, propose an explicit schema update, merge behavior, and tests instead of hiding the fields in importer-only logic.

## Judged Awards

TBA event awards are post-event auxiliary labels. They are never pre-match inputs. `build-features` parses the tracked judged awards into per-slot columns:

- Cultural axes: `impact`, `ei`.
- Machine axes: `auto`, `quality`, `design`, `control`, `excellence`.

The award target columns intentionally contain `NaN` for unknown or censored axes. Impact winners emit both Impact and EI positives. EI winners emit an EI positive, and emit an Impact `0.0` only when the team was still chronologically eligible to win Impact at that event. Machine award winners emit exactly one positive axis and leave unrelated machine axes as `NaN`.

## Optional Statbotics Enrichment

Statbotics can be useful for external comparison or future feature work, but it is not currently an implemented in-repo baseline. If enriching feature tables with Statbotics data:

- Inspect returned keys before hard-coding EPA paths.
- Record whether a field is pre-match, post-match, current, or endpoint/version-dependent.
- Join through shared keys such as `match_key`, `event_key`, and team keys.
- Validate the feature timing before using the column for prediction.

## Parquet Boundary

`write_feature_table` validates the DataFrame, removes training-derived `*_idx` and `*_z` columns, and writes with the `pyarrow` engine. `read_feature_table` reads with the same engine.

Do not blanket-cast Parquet columns to `float32`. The persisted table should preserve useful pandas dtypes, including nullable integer columns, timestamps, strings, and numeric feature columns. Tensor-bound values are cast deliberately when `MatchTensorDataset` builds PyTorch tensors.

## V5.6.4 Prior Features

`build-prior-features` writes a separate prior Parquet file for offline pretraining. It builds a fixed transductive universe for team numbers `0..12500` by default. Row `0` is the learned ghost robot. Known teams are pulled from paginated TBA team endpoints; unknown historical gaps and future rookie numbers receive generated narrative anchors.

The output contains one row per team number with:

- `team_number`
- `team_key`
- `archetype`: `ghost_token`, `anchor`, `ghost`, `sibling`, or `future`
- `target_season`
- cleaned natural-language `narrative`
- stable `narrative_hash`
- `embedding_model` and `llm_dim`
- `openai_narrative_vector`, a 256-D text embedding
- `norm_epa_t_minus_4` through `norm_epa_t_minus_1`, sourced from Statbotics normalized EPA for the four completed seasons before the target season
- `norm_epa_observed_t_minus_4` through `norm_epa_observed_t_minus_1`
- raw cultural targets: `raw_rookie_year_delta`, `raw_seasons_played`, `raw_total_award_count`, `raw_blue_banner_count`, `raw_championship_appearance_count`, `raw_championship_win_count`, and `raw_technical_award_count`

Anchor and Ghost narratives are built from TBA team profile, event history, and awards where `year < target_season`. Sibling narratives describe nearby known teams by number to ground empty historical slots. Future rookie narratives use the projected registration formula and modern COTS-era technical grounding. There is no explicit hardware vector or award-decay vector. Historical awards are injected into natural text, and the OpenAI embedding cache prevents duplicate API calls for unchanged narratives. The normalized EPA trajectory uses Statbotics' own normalized EPA field, not a local z-score of raw EPA. For target season `2026`, the source years are `2022..2025`. Missing team-years stay `NaN` with observed masks set to `False`; those axes are skipped in the EPA trajectory loss. Cultural targets are raw unnormalized counts, and Team `0`, Sibling, and Future rows use finite zero cultural targets.

## Team Indexing

Training maps string FRC team keys such as `frc254` to V5 embedding indices in memory:

1. `make_v5_team_index_maps` scans all six team slot columns.
2. `team_base_idx` maps `frc####` directly to numeric row `####` for team numbers up to the V5.6 prior maximum; `frc0` is the learned ghost robot.
3. `team_event_idx` maps `(event_key, team_key)` pairs to contiguous integers starting at `1`.
4. Blank or explicit missing slots use `team_base_idx=0` and `team_event_idx=0`; base row `0` is learned, while event row `0` remains the no-delta row.
5. Per-slot `missing_team_mask` columns are true only for explicit blank/missing team slots.
6. `MatchTensorDataset` converts these columns to PyTorch tensors.

Team numbers above the current V5.6 prior maximum raise a clear error. Rebuild the prior with a larger `max_team_number` before training on those teams.

## Missing Data

Missing enrichment data should usually remain as null columns on the TBA spine, not cause official matches to be dropped. Choose imputation based on feature meaning:

- Add missing indicators when missingness may carry signal.
- Use `0` or `False` only when that truly means "none observed."
- Keep unknown categorical values distinct from known negative values.
- Do not fill pre-match rows with post-match or post-event aggregates.

V5.6 represents explicit missing team slots with the learned ghost base row and the no-delta event row. Those slots are still visible to attention. DQ and surrogate flags remain diagnostics; they are not treated as missing robots by default.

## Commands

Build a season feature table:

```bash
latentstrat build-features --season 2026 --output data/features/season/features_2026.parquet
```

Build the same table with V5.7 sidecars:

```bash
latentstrat build-features --season 2026 --output data/features/season/features_2026.parquet \
  --sidecar-output-dir data/sidecars/v58_2026
```

Train from that local Parquet file:

```bash
latentstrat train-features data/features/season/features_2026.parquet --output artifacts/season/features_run
```

Train with optional V5.7 sidecar losses:

```bash
latentstrat train-features data/features/season/features_2026.parquet \
  --rankings-sidecar data/sidecars/v58_2026/rankings_2026.parquet \
  --selections-sidecar data/sidecars/v58_2026/selections_2026.parquet \
  --playoffs-sidecar data/sidecars/v58_2026/playoffs_2026.parquet
```

## Related

- [Data sources](data-sources.md): source-system timing and leakage boundaries.
- [Schemas and artifacts reference](schemas-and-artifacts-reference.md): column groups emitted by feature and sidecar builds.
- [Season training](season-training.md): how feature tables are consumed by the Set Transformer.
- [Scouting data layer](scouting-data-layer.md): local scouting SQLite schema and feature merge behavior.
