# Statbotics Data Access Patterns

Use this reference when writing Python code against Statbotics data or combining Statbotics with TBA data.

## Prefer the LatentStrat Provider

In LatentStrat code, prefer the local wrapper because it centralizes caching and keeps provider usage consistent.

```python
from frc.providers.statbotics_provider import StatboticsProvider

provider = StatboticsProvider()  # uses data/cache/statbotics.sqlite by default
matches = provider.get_matches(year=2024, limit=10000)
```

For one-off scripts outside the project wrapper, use the official package directly.

```python
import statbotics

sb = statbotics.Statbotics()
team_year = sb.get_team_year(254, 2024)
event_matches = sb.get_matches(event="2024cada", limit=10000)
```

## Inspect Response Shapes

The pinned `statbotics==3.0.0` package uses nested dictionaries for much of the EPA data. Inspect keys before hard-coding paths.

```python
sample = provider.get_matches(year=2024, limit=1)[0]
print(sample.keys())
print(sample.get("pred", {}).keys())
print(sample.get("result", {}).keys())
```

Common patterns in `statbotics==3.0.0`:

- Match rows use `key`, `event`, `year`, `alliances`, `pred`, and `result`.
- `pred` contains pre-match alliance expectations such as expected scores and win probability.
- `result` contains actual scores and game-specific post-match fields.
- Team-year and team-event rows often use nested `epa` data such as `epa["stats"]`, `epa["breakdown"]`, and `epa["total_points"]`.
- Team-match rows may contain nested `epa`, including `epa["post"]`; do not use post-match fields as predictive features.

Older examples and alternate API versions may mention flat fields such as `epa_end`, `epa_pre`, or `red_epa_pre`. Treat those names as endpoint/version-dependent and verify them before use.

## Pagination and Bulk Pulls

List endpoints are limited. `get_matches` defaults to `limit=200`, and the max limit is 10000. For full-season ML datasets, page explicitly.

```python
def collect_matches(provider, year):
    rows = []
    limit = 10000
    offset = 0
    while True:
        page = provider.get_matches(year=year, limit=limit, offset=offset)
        rows.extend(page)
        if len(page) < limit:
            break
        offset += limit
    return rows
```

Use broad list endpoints plus local Pandas filtering for training datasets. Avoid loops that call `get_match` or `get_team_match` thousands of times.

```python
import pandas as pd

matches = collect_matches(provider, 2024)
df = pd.json_normalize(matches, sep="_")
quals = df[df["comp_level"] == "qm"]
```

## LatentStrat Feature Table Guidance

For LatentStrat ML workflows, normalize Statbotics responses into Pandas DataFrames with `pd.json_normalize` and left-join them onto the TBA match-table spine. Do not let Statbotics rows replace the TBA spine unless a future runtime task intentionally changes the dataset grain.

- Rename `key` to `match_key` and `event` to `event_key` before joining.
- Keep pre-match prediction columns separate from post-match result columns.
- Treat EPA fields as endpoint/version-dependent until keys are inspected.
- Preserve the DataFrame/Parquet boundary; do not convert all numeric columns to `float32` during enrichment.

Use `$latentstrat-feature-pipeline` for team slot joins, PyArrow behavior, and tensor-bound dtype conversion. Use `$latentstrat-model-evaluation` before claiming Statbotics is an in-repo baseline; the current implemented baselines are mean and ridge match-OPR.

## Leakage-Safe Feature Extraction

Use match `pred` fields for pre-match expectations and `result` fields only for labels.

```python
rows = []
for match in provider.get_matches(year=2024, limit=10000):
    pred = match.get("pred") or {}
    result = match.get("result") or {}
    rows.append(
        {
            "match_key": match["key"],
            "event_key": match["event"],
            "pred_red_score": pred.get("red_score"),
            "pred_blue_score": pred.get("blue_score"),
            "pred_red_win_prob": pred.get("red_win_prob"),
            "actual_red_score": result.get("red_score"),
            "actual_blue_score": result.get("blue_score"),
        }
    )
```

Do not use team-match `epa["post"]`, current team `epa`, or end-of-event summaries as features for earlier historical matches.

## Joining with TBA

Statbotics and TBA share event and match key conventions, but field names differ.

```python
stat_df = pd.json_normalize(provider.get_matches(year=2024, limit=10000), sep="_")
stat_df = stat_df.rename(columns={"key": "match_key", "event": "event_key"})

merged = tba_df.merge(
    stat_df[["match_key", "pred_red_score", "pred_blue_score"]],
    on="match_key",
    how="left",
)
```

Team identifiers from Statbotics are usually integers. Convert them before joining with TBA-style team keys.

```python
def team_key(team_number):
    return f"frc{int(team_number)}"
```

Use TBA for official raw truth and game-specific score breakdowns. Use Statbotics for derived prediction context and EPA summaries.
