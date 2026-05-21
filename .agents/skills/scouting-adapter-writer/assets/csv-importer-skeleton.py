"""Template for a LatentStrat scouting CSV importer.

Copy this into a project script or test fixture and replace the mapping section
for the specific scouting source. Keep source-specific logic explicit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlmodel import Session

from frc.scouting import TeamMatchScouting, create_scouting_engine


def clean_optional_str(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def optional_int(value: Any) -> int | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    return int(float(value))


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


def normalize_alliance_color(value: Any) -> str:
    color = str(value).strip().lower()
    if color not in {"red", "blue"}:
        raise ValueError(f"Invalid alliance color: {color!r}")
    return color


def ingest_csv(
    csv_path: str | Path,
    *,
    event_key: str,
    db_path: str | Path = "data/scouting.db",
) -> int:
    df = pd.read_csv(csv_path)
    engine = create_scouting_engine(db_path)
    count = 0

    with Session(engine) as session:
        for row_number, row in df.iterrows():
            try:
                record = TeamMatchScouting(
                    team_key=normalize_team_key(row["Team Number"]),
                    match_key=normalize_qm_match_key(event_key, row["Match Number"]),
                    alliance_color=normalize_alliance_color(row["Alliance"]),
                    auto_pieces_scored=optional_int(row.get("Auto Pieces")) or 0,
                    teleop_pieces_scored=optional_int(row.get("Teleop Pieces")) or 0,
                    endgame_status=clean_optional_str(row.get("Endgame")),
                    driver_ability_rating=optional_int(row.get("Driver Rating")) or 3,
                    played_defense=yes_no(row.get("Played Defense")),
                    scouter_name=clean_optional_str(row.get("Scouter")),
                )
            except Exception as exc:
                raise ValueError(f"Invalid scouting row {row_number}: {exc}") from exc
            session.merge(record)
            count += 1
        session.commit()

    return count
