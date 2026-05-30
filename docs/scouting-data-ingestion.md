---
tags:
  - latentstrat
  - scouting-data
  - feature-pipeline
aliases:
  - "Scouting Data Ingestion"
  - "Scouting Import Guide"
related:
  - "[[scouting-data-layer]]"
  - "[[feature-pipeline]]"
  - "[[data-sources]]"
---

# Scouting Data Ingestion Guide

This guide defines the standard pattern for writing custom importers that move raw human-scouted data from CSVs or spreadsheets into LatentStrat's SQLModel scouting database.

LatentStrat uses a data lake workflow:

1. Raw scouting files are imported into `data/scouting/scouting.db`.
2. `build-features` merges scouting rows with TBA match data.
3. PyTorch training reads the resulting Parquet file and never queries SQLite during the training loop.

## Setup

Initialize the local scouting database before running importer scripts:

```powershell
python -m latentstrat.cli init-scouting-db --path data/scouting/scouting.db
```

Importer scripts should use the current scouting API:

```python
from sqlmodel import Session

from frc.scouting import TeamScouting, create_scouting_engine

engine = create_scouting_engine("data/scouting/scouting.db")

with Session(engine) as session:
    session.merge(TeamScouting(team_key="frc254", drive_base="Swerve"))
    session.commit()
```

Use `session.merge(...)` instead of `session.add(...)`. Re-running an importer after fixing a CSV should update existing primary keys instead of failing on duplicate rows.

## Key Normalization

Every importer must normalize raw scouting identifiers into TBA keys before writing database records. Most merge failures come from mismatched keys.

```python
from typing import Any

import pandas as pd


def clean_optional_str(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def optional_int(value: Any) -> int | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    return int(float(value))


def optional_float(value: Any) -> float | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    return float(value)


def yes_no(value: Any, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return True
    if text in {"no", "n", "false", "0"}:
        return False
    return default


def normalize_team_key(value: Any) -> str:
    return f"frc{int(float(value))}"


def normalize_qm_match_key(event_key: str, value: Any) -> str:
    return f"{event_key}_qm{int(float(value))}"
```

Qualification matches use keys like `2026ilch_qm12`. Playoff importers must use the real TBA match key format for the competition level they import, such as `sf` and `f` keys. If the raw file already contains TBA match keys, prefer those over rebuilding them.

## Static Pit Data

Use `TeamScouting` for global pit data that is not tied to a specific event or match.

```python
import pandas as pd
from sqlmodel import Session

from frc.scouting import TeamScouting, create_scouting_engine


def ingest_pit_data(csv_path: str, db_path: str = "data/scouting/scouting.db") -> None:
    df = pd.read_csv(csv_path)
    engine = create_scouting_engine(db_path)

    with Session(engine) as session:
        for _, row in df.iterrows():
            record = TeamScouting(
                team_key=normalize_team_key(row["Team Number"]),
                drive_base=clean_optional_str(row.get("Drive Base")),
                robot_weight_lbs=optional_float(row.get("Weight")),
                frame_dimensions_inches=clean_optional_str(row.get("Frame Size")),
                programming_language=clean_optional_str(row.get("Language")),
            )
            session.merge(record)
        session.commit()
```

Merged feature columns will use prefixes such as `red_team_1_pit_drive_base` and `blue_team_3_pit_robot_weight_lbs`.

## Event Context

Use `EventScouting` for event-wide conditions.

```python
import pandas as pd
from sqlmodel import Session

from frc.scouting import EventScouting, create_scouting_engine


def ingest_event_context(csv_path: str, db_path: str = "data/scouting/scouting.db") -> None:
    df = pd.read_csv(csv_path)
    engine = create_scouting_engine(db_path)

    with Session(engine) as session:
        for _, row in df.iterrows():
            record = EventScouting(
                event_key=clean_optional_str(row["Event Key"]),
                carpet_condition=clean_optional_str(row.get("Carpet")),
                venue_notes=clean_optional_str(row.get("Notes")),
            )
            session.merge(record)
        session.commit()
```

Merged feature columns use the `event_scout_` prefix.

## Match Context

Use `MatchScouting` for match-wide conditions independent of individual teams.

```python
import pandas as pd
from sqlmodel import Session

from frc.scouting import MatchScouting, create_scouting_engine


def ingest_match_context(
    csv_path: str,
    event_key: str,
    db_path: str = "data/scouting/scouting.db",
) -> None:
    df = pd.read_csv(csv_path)
    engine = create_scouting_engine(db_path)

    with Session(engine) as session:
        for _, row in df.iterrows():
            rating = optional_int(row.get("Referee Strictness"))
            record = MatchScouting(
                match_key=normalize_qm_match_key(event_key, row["Match Number"]),
                event_key=event_key,
                field_fault_occurred=yes_no(row.get("Field Fault")),
                audience_delay_minutes=optional_int(row.get("Delay Minutes")) or 0,
                referee_strictness_rating=rating,
            )
            session.merge(record)
        session.commit()
```

Merged feature columns use the `match_scout_` prefix.

## Team Event Status

Use `TeamEventScouting` for team state at a specific event, such as inspection or robot functionality.

```python
import pandas as pd
from sqlmodel import Session

from frc.scouting import TeamEventScouting, create_scouting_engine


def ingest_team_event_status(
    csv_path: str,
    event_key: str,
    db_path: str = "data/scouting/scouting.db",
) -> None:
    df = pd.read_csv(csv_path)
    engine = create_scouting_engine(db_path)

    with Session(engine) as session:
        for _, row in df.iterrows():
            record = TeamEventScouting(
                team_key=normalize_team_key(row["Team Number"]),
                event_key=event_key,
                passed_inspection=yes_no(row.get("Passed Inspection")),
                is_functional=yes_no(row.get("Functional"), default=True),
                major_breakdown_notes=clean_optional_str(row.get("Breakdown Notes")),
            )
            session.merge(record)
        session.commit()
```

Merged feature columns use slot-specific prefixes such as `red_team_1_event_scout_passed_inspection`.

## Alliance Strategy

Use `MatchAllianceScouting` for data that belongs to one alliance in one match. The `alliance_color` value must be exactly `"red"` or `"blue"`.

```python
import pandas as pd
from sqlmodel import Session

from frc.scouting import MatchAllianceScouting, create_scouting_engine


def ingest_alliance_strategy(
    csv_path: str,
    event_key: str,
    db_path: str = "data/scouting/scouting.db",
) -> None:
    df = pd.read_csv(csv_path)
    engine = create_scouting_engine(db_path)

    with Session(engine) as session:
        for _, row in df.iterrows():
            alliance_color = str(row["Alliance"]).strip().lower()
            if alliance_color not in {"red", "blue"}:
                raise ValueError(f"Invalid alliance color: {alliance_color!r}")

            record = MatchAllianceScouting(
                match_key=normalize_qm_match_key(event_key, row["Match Number"]),
                alliance_color=alliance_color,
                coopertition_agreed=yes_no(row.get("Coopertition")),
                coordinated_auto_run=yes_no(row.get("Coordinated Auto")),
                strategic_meltdown=yes_no(row.get("Strategic Meltdown")),
            )
            session.merge(record)
        session.commit()
```

Merged feature columns use `red_alliance_scout_` and `blue_alliance_scout_` prefixes.

## Team Match Scouting

Use `TeamMatchScouting` for one row per team per match. These are dynamic post-match observations, so they are safe to store in Parquet but should not be used as default pre-match inputs.

The current feature merge attaches these rows to robot slots by filtering on `alliance_color`. Do not insert `"unknown"` if you expect the row to merge into features. If the raw file does not include alliance color, resolve it from TBA match data before writing, or reject the row with a clear error.

```python
import pandas as pd
from sqlmodel import Session

from frc.scouting import TeamMatchScouting, create_scouting_engine


def ingest_team_match_data(
    csv_path: str,
    event_key: str,
    db_path: str = "data/scouting/scouting.db",
) -> None:
    df = pd.read_csv(csv_path)
    engine = create_scouting_engine(db_path)

    with Session(engine) as session:
        for _, row in df.iterrows():
            alliance_color = str(row["Alliance"]).strip().lower()
            if alliance_color not in {"red", "blue"}:
                raise ValueError(f"Invalid alliance color: {alliance_color!r}")

            record = TeamMatchScouting(
                team_key=normalize_team_key(row["Team Number"]),
                match_key=normalize_qm_match_key(event_key, row["Match Number"]),
                alliance_color=alliance_color,
                auto_pieces_scored=optional_int(row.get("Auto Pieces")) or 0,
                teleop_pieces_scored=optional_int(row.get("Teleop Pieces")) or 0,
                endgame_status=clean_optional_str(row.get("Endgame")),
                driver_ability_rating=optional_int(row.get("Driver Rating")) or 3,
                played_defense=yes_no(row.get("Played Defense")),
                defended_by_team=(
                    normalize_team_key(row["Defended By"])
                    if pd.notna(row.get("Defended By"))
                    else None
                ),
                scouter_name=clean_optional_str(row.get("Scouter")),
            )
            session.merge(record)
        session.commit()
```

Merged feature columns use slot-specific prefixes such as `red_team_2_match_scout_teleop_pieces_scored`.

## Leakage Discipline

Scouting data falls into two groups:

- Static or pre-match data: pit scouting, event context, and team-event status. These can become model inputs in a future model version.
- Dynamic match data: auto pieces, teleop pieces, defense, endgame, and driver ratings recorded during or after a match. These should be treated as future auxiliary targets or diagnostics, not default pre-match inputs.

The current default training workflow stores scouting columns in Parquet but does not add them to the default target map or model inputs.

## Validation

After ingestion, rebuild the feature store:

```powershell
python -m latentstrat.cli build-features --event-key 2026ilch --output data/features/event/features_2026ilch.parquet
```

Inspect the merged scouting columns:

```powershell
python -c "import pandas as pd; df=pd.read_parquet('data/features/event/features_2026ilch.parquet', engine='pyarrow'); print([c for c in df.columns if 'scout' in c])"
```

You can also query the SQLite database directly:

```python
from sqlmodel import Session, select

from frc.scouting import TeamMatchScouting, create_scouting_engine

engine = create_scouting_engine("data/scouting/scouting.db")

with Session(engine) as session:
    rows = session.exec(select(TeamMatchScouting)).all()
    print(len(rows))
```
