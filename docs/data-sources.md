---
tags:
  - latentstrat
  - data-sources
  - feature-pipeline
aliases:
  - "LatentStrat Data Sources"
related:
  - "[[feature-pipeline]]"
  - "[[prior-training]]"
  - "[[season-training]]"
  - "[[schemas-and-artifacts-reference]]"
---

# Data Sources

LatentStrat uses several data sources, but each source has a different role. The most important rule is timing: the model should not use information that would not have been known at the prediction point.

## Summary Table

| Source | Used for | Typical commands | Timing role |
|---|---|---|---|
| The Blue Alliance | Teams, events, matches, score breakdowns, awards, rankings, alliances | `build-features`, `build-prior-features`, `sync-match-breakdowns` | Match spine and historical labels |
| Statbotics | Historical normalized EPA trajectory for prior training | `build-prior-features` | Prior-season strength signal |
| OpenAI embeddings | Narrative text targets for prior distillation | `build-prior-features` | Offline text semantics |
| Local scouting SQLite | Optional scouting observations | `init-scouting-db`, `build-features` | Local team/event/match scouting |
| Generated sidecars | Rankings, selections, playoff ordering | `build-features --sidecar-output-dir` | Auxiliary post-event labels |

## The Blue Alliance

The Blue Alliance is the main FRC data source. LatentStrat uses TBA for:

- Team profiles and team numbers.
- Event schedules and played matches.
- Match score breakdowns.
- Awards.
- Rankings.
- Alliance selections and playoff alliances.

TBA match data is the match-table spine. That means official match rows define the main training table. Other sources join onto those rows; they do not create new match rows.

## Score Breakdowns

FRC games change every year, so raw score-breakdown keys should not leak into the model. LatentStrat maps season-specific score-breakdown fields into generic model-facing columns.

For V5.7, the 2026 mapper writes groups such as:

- Atomic counts.
- Committed fouls.
- Bonus binary targets.
- Special binary targets.

Committed fouls are inverted because TBA reports foul points as points awarded to the opponent. If blue receives foul points, those points came from red committing fouls.

V6-Lite V1 also preserves raw TBA match payloads for offline historical score-archetype training:

```text
data/world_model/match_breakdowns.sqlite
```

This durable corpus is separate from the disposable TBA transport cache. Synchronization defaults to normal official event types `0..5`; FOC `6` and remote `7` require explicit flags. The requested range is `2015..2026`, but the current TBA OpenAPI breakdown schemas skip `2021`, so that season is retained in the audit and excluded from encoder training.

The encoder emits one row per played alliance only when both posted scores are present and both alliance breakdown dictionaries are non-null. It keeps raw incomplete match objects in SQLite so a refresh can update them later.

## Canonical Event Weeks

V5.8 uses `event_week` for walk-forward validation. TBA's raw week value is preserved as `raw_event_week`, but model-facing `event_week` starts at `1`. Raw Week 0 is bundled into Week 1. If TBA week data is missing, LatentStrat falls back to dense chronological event order.

This matters because walk-forward validation simulates the season:

- Train on weeks `<= N`.
- Validate on week `N + 1`.

## Statbotics

Statbotics contributes historical normalized EPA trajectory targets for prior training. For target season `2026`, the source years are `2022`, `2023`, `2024`, and `2025`.

LatentStrat uses Statbotics' normalized EPA field, not a local z-score of raw EPA. Missing team-years stay missing and are masked out of the EPA trajectory loss.

This gives the prior a strength trajectory:

- A team can be rising.
- A team can be falling.
- A team can be consistently strong.
- A team can be unknown because it has no historical EPA.

## OpenAI Narrative Embeddings

Prior training needs a semantic target for each team number. LatentStrat builds a narrative for each team number and embeds that text with OpenAI.

The narrative target helps encode things that a scalar metric cannot:

- Team identity.
- Geography.
- Historical context.
- Nearby team-number context for unassigned slots.
- Future rookie assumptions.

Embeddings are cached by a stable hash of the model, dimensions, and narrative text. If a run is interrupted, completed embeddings should be cache hits on the next run.

## Scouting SQLite

Local scouting data is optional. It lives in `data/scouting/scouting.db` by default and is created with:

```bash
latentstrat init-scouting-db --path data/scouting/scouting.db
```

When the database exists, `build-features` can merge scouting rows into the Parquet feature table. Training still reads only the Parquet file.

Scouting joins must preserve grain:

- Match context joins by `match_key`.
- Event context joins by `event_key`.
- Team-event rows join by `event_key` and `team_key`.
- Team-match rows join by `match_key` and `team_key`.

## Sidecar Files

Rankings, selections, and playoff outcomes are not pre-match inputs. They are post-event labels used to shape representations during training.

Sidecars include:

- Rankings: qualification rank pairs.
- Selections: captain, picked team, and computed passed-over team.
- Playoffs: alliance ordering by playoff finish.

In walk-forward validation, sidecars are filtered by week before training, so future event results cannot leak into earlier folds.

## Secrets And Caches

Live ingestion usually needs:

```powershell
TBA_API_KEY=...
OPENAI_API_KEY=...
```

Never copy real keys into docs, logs, screenshots, or committed files.

Common local storage paths include:

- TBA request cache at `data/cache/tba.sqlite`.
- Durable raw match-breakdown corpus at `data/world_model/match_breakdowns.sqlite`.
- Statbotics response cache at `data/cache/statbotics.sqlite`.
- OpenAI embedding cache at `data/cache/openai_embeddings.sqlite`.
- Scouting database at `data/scouting/scouting.db`.
- Durable consolidated embeddings at `data/embeddings/latentstrat_embeddings.sqlite`.
- Generated Parquet features under `data/features/`.
- Sidecar Parquet files under `data/sidecars/`.
- Model and run artifacts under grouped `artifacts/` subdirectories.
- TensorBoard logs under `runs/`.

The cache files are local development artifacts, not model design. Parquet feature tables remain file artifacts; provider caches and scouting stores use SQLite.

## Related

- [Feature pipeline](feature-pipeline.md): how source data becomes Parquet feature rows.
- [Prior training](prior-training.md): how OpenAI, Statbotics, and TBA history shape the Day Zero prior.
- [Season training](season-training.md): how match features and sidecars are used during training.
- [Schemas and artifacts reference](schemas-and-artifacts-reference.md): column groups and generated artifact contracts.
