# tbapy Patterns and Analytics

Use `tbapy` as the preferred Python wrapper for The Blue Alliance API v3. LatentStrat pins `tbapy==1.3.2` and reads live API credentials from `TBA_API_KEY`.

## Initialization

```python
import os

import tbapy

tba = tbapy.TBA(os.environ["TBA_API_KEY"])
```

For LatentStrat runtime code, prefer the existing `frc.providers.tba_provider.TbaProvider` wrapper because it centralizes caching, environment handling, and conversion to project models.

## Common Narrow Pulls

```python
team = tba.team("frc254")
event = tba.event("2026ilch")
event_matches = tba.event_matches("2026ilch")
team_event_matches = tba.team_matches("frc254", "2026ilch")
rankings = tba.event_rankings("2026ilch")
oprs = tba.event_oprs("2026ilch")
awards = tba.event_awards("2026ilch")
alliances = tba.event_alliances("2026ilch")
```

Prefer the narrowest endpoint that answers the question. Avoid fetching all team history or full seasons unless the analysis really needs that scope.

## Missing Data and Unplayed Matches

Anticipate unplayed matches and incomplete payloads:

- `actual_time` can be `None` for future or unscheduled matches.
- `score_breakdown` can be `None` or empty for unplayed, old, or incomplete matches.
- `winning_alliance` can be `"red"`, `"blue"`, or `""`.
- Nested fields can differ by year and sometimes change during a season.

```python
for match in event_matches:
    if getattr(match, "actual_time", None) is None:
        continue
    if not getattr(match, "score_breakdown", None):
        continue

    for color in ("red", "blue"):
        breakdown = match.score_breakdown.get(color, {})
        auto_points = breakdown.get("autoPoints", 0)
        print(f"{match.key} {color} auto points: {auto_points}")
```

## Normalizing Data for Analytics

Flatten nested TBA objects carefully and keep raw records when fields are season-specific.

```python
import json

import pandas as pd


def get_team_match_dataframe(matches):
    rows = []
    for match in matches:
        if not getattr(match, "score_breakdown", None):
            continue

        for color in ("red", "blue"):
            alliance = match.alliances[color]
            breakdown = match.score_breakdown.get(color, {})

            for team_key in alliance["team_keys"]:
                rows.append(
                    {
                        "match_key": match.key,
                        "comp_level": match.comp_level,
                        "team_key": team_key,
                        "alliance": color,
                        "total_points": alliance["score"],
                        "auto_points": breakdown.get("autoPoints", 0),
                        "rp": breakdown.get("rp", 0),
                        "raw_breakdown": json.dumps(breakdown),
                    }
                )

    return pd.DataFrame(rows)
```

## LatentStrat Feature Tables

For ML workflows, prefer DataFrame-oriented extraction that can feed the LatentStrat feature pipeline. The TBA match table should remain the match-grain spine, with one row per played match and six `frc####` team slot columns.

- Preserve `match_key`, `event_key`, `comp_level`, match ordering, scores, and raw score breakdown context.
- Keep team keys as strings such as `frc254`; do not precompute persistent embedding ids in TBA extraction code.
- Use left joins when adding scouting or external analytics so missing enrichment does not drop official matches.
- Write feature tables through the project feature pipeline so PyArrow dtypes, nullable columns, and training-derived columns are handled consistently.

Use `$latentstrat-feature-pipeline` for Parquet boundaries, team indexing, and tensor-ready feature contracts.

## Handling Historical Quirks

Use defensive guards when analyzing historical data. TBA's older data can be incomplete, and pre-2015 matches do not have score breakdowns.

### Guard Missing Score Breakdowns

```python
def get_foul_points(match):
    year = int(match.key[:4])
    if year < 2015 or not getattr(match, "score_breakdown", None):
        return None

    return {
        "blue_fouls": match.score_breakdown.get("blue", {}).get("foulPoints", 0),
        "red_fouls": match.score_breakdown.get("red", {}).get("foulPoints", 0),
    }
```

### Normalize Renamed Awards

```python
def normalize_award_name(award_name):
    name = award_name.lower()
    if "chairman" in name:
        return "FIRST Impact Award"
    if "dean's list" in name:
        return "FIRST Leadership Award"
    return award_name
```

### Handle Incomplete Legacy Events

```python
alliances = tba.event_alliances("2007gal")
if not alliances:
    print("Alliance data is unavailable for this legacy event.")
```

## Raw Payload Preservation

When building feature tables, preserve raw TBA objects or raw JSON columns for fields that may be needed later. This is especially important for match score breakdowns, district points, Regional advancement fields, rankings, awards, and playoff alliances.
