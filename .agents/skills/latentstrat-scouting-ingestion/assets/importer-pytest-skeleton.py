"""Template tests for a LatentStrat scouting importer."""

from __future__ import annotations

import pandas as pd
from sqlmodel import Session, select

from frc.scouting import TeamMatchScouting, create_db_and_tables, create_scouting_engine
from latentstrat.season.features import merge_scouting_features


def test_importer_writes_and_updates_rows(tmp_path):
    db_path = create_db_and_tables(tmp_path / "scouting.db")
    csv_path = tmp_path / "source.csv"
    pd.DataFrame(
        [
            {
                "Team Number": 254,
                "Match Number": 1,
                "Alliance": "Red",
                "Teleop Pieces": 7,
            }
        ]
    ).to_csv(csv_path, index=False)

    from path.to.importer import ingest_csv

    assert ingest_csv(csv_path, event_key="2026test", db_path=db_path) == 1
    assert ingest_csv(csv_path, event_key="2026test", db_path=db_path) == 1

    engine = create_scouting_engine(db_path)
    with Session(engine) as session:
        rows = session.exec(select(TeamMatchScouting)).all()
    assert len(rows) == 1
    assert rows[0].team_key == "frc254"


def test_importer_rows_merge_into_feature_table(tmp_path):
    db_path = create_db_and_tables(tmp_path / "scouting.db")
    csv_path = tmp_path / "source.csv"
    pd.DataFrame(
        [
            {
                "Team Number": 254,
                "Match Number": 1,
                "Alliance": "Red",
                "Teleop Pieces": 7,
            }
        ]
    ).to_csv(csv_path, index=False)

    from path.to.importer import ingest_csv

    ingest_csv(csv_path, event_key="2026test", db_path=db_path)
    feature_table = pd.DataFrame(
        [
            {
                "event_key": "2026test",
                "match_key": "2026test_qm1",
                "red_team_1_key": "frc254",
                "red_team_2_key": "frc1",
                "red_team_3_key": "frc2",
                "blue_team_1_key": "frc3",
                "blue_team_2_key": "frc4",
                "blue_team_3_key": "frc5",
            }
        ]
    )

    merged = merge_scouting_features(feature_table, db_path)
    assert int(merged["red_team_1_match_scout_teleop_pieces_scored"].iloc[0]) == 7
