import os
import sqlite3

import pandas as pd
import pytest
from sqlmodel import Session
from typer.testing import CliRunner

from frc.scouting import (
    EventScouting,
    MatchAllianceScouting,
    MatchScouting,
    TeamEventScouting,
    TeamMatchScouting,
    TeamScouting,
    create_db_and_tables,
    create_scouting_engine,
)
from latentstrat.cli import app
from latentstrat.config import default_options
from latentstrat.features import (
    merge_scouting_features,
    read_feature_table,
    train_feature_table,
    write_feature_table,
)
from latentstrat.secrets import load_environment


def _feature_table(row_count=8):
    rows = []
    for idx in range(row_count):
        red = [idx % 6, (idx + 1) % 6, (idx + 2) % 6]
        blue = [(idx + 3) % 6, (idx + 4) % 6, (idx + 5) % 6]
        rows.append(
            {
                "season": 2026,
                "event_key": "2026features",
                "match_key": f"2026features_qm{idx + 1}",
                "comp_level": "qm",
                "set_number": 1,
                "match_number": idx + 1,
                "red_team_1_key": f"frc{red[0] + 1}",
                "red_team_2_key": f"frc{red[1] + 1}",
                "red_team_3_key": f"frc{red[2] + 1}",
                "blue_team_1_key": f"frc{blue[0] + 1}",
                "blue_team_2_key": f"frc{blue[1] + 1}",
                "blue_team_3_key": f"frc{blue[2] + 1}",
                "red_score_raw": 100 + idx,
                "blue_score_raw": 90 + idx,
                "red_has_surrogate": False,
                "blue_has_surrogate": False,
                "red_has_dq": False,
                "blue_has_dq": False,
                "scheduled_time": pd.Timestamp("2026-03-01", tz="UTC"),
                "actual_time": pd.NaT,
                "post_result_time": pd.NaT,
                "sort_time": 1_780_000_000 + idx,
                "sort_ordinal": idx + 1,
                "event_week": 0,
                "red_total_score": 100 + idx,
                "blue_total_score": 90 + idx,
                "red_foul_pts": idx % 4,
                "blue_foul_pts": (idx + 1) % 4,
                "win_margin": 10,
                "fouls_drawn": (idx % 4) - ((idx + 1) % 4),
                "red_win": True,
            }
        )
    return pd.DataFrame(rows)


def test_dotenv_loader_hydrates_tba_key_without_committed_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("TBA_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("TBA_API_KEY=test-key\n", encoding="utf-8")

    assert load_environment(env_path)
    assert os.environ["TBA_API_KEY"] == "test-key"


def test_feature_parquet_round_trip_preserves_source_table_shape(tmp_path):
    table = _feature_table()
    table["red_team_1_idx"] = pd.Series([0] * len(table), dtype="Int64")
    table["red_team_1_match_scout_teleop_pieces_scored"] = pd.Series(
        [7] * len(table), dtype="Int64"
    )
    path = write_feature_table(table, tmp_path / "features.parquet")

    loaded = read_feature_table(path)

    assert path.exists()
    assert len(loaded) == len(table)
    assert "red_team_1_idx" not in loaded.columns
    assert not any(column.endswith("_idx") for column in loaded.columns)
    assert loaded["red_team_1_key"].iloc[0] == "frc1"
    assert "red_team_1_match_scout_teleop_pieces_scored" in loaded.columns
    assert str(loaded["red_team_1_match_scout_teleop_pieces_scored"].dtype) == "Int64"


def test_train_feature_table_adds_indices_and_writes_artifacts(tmp_path):
    table = _feature_table()
    table["red_team_1_match_scout_teleop_pieces_scored"] = pd.Series(
        [7] * len(table), dtype="Int64"
    )
    output = tmp_path / "artifacts"
    opts = default_options().model_copy(
        update={"epochs": 1, "mini_batch_size": 4, "use_early_stopping": False}
    )

    result = train_feature_table(
        table,
        opts,
        output_dir=output,
        verbose=False,
    )

    assert "red_team_1_idx" in result.prepared.columns
    assert result.prepared["red_team_1_idx"].dtype == "Int64"
    assert len(result.team_index_map) == 6
    assert (output / "feature_match_table.csv").exists()
    assert (output / "feature_history.csv").exists()
    assert (output / "feature_continuous_metrics.csv").exists()
    assert (output / "feature_binary_metrics.csv").exists()
    assert (output / "feature_set_attention.csv").exists()
    assert (output / "feature_zero_out_diagnostics.csv").exists()


def test_init_scouting_db_creates_expected_tables(tmp_path):
    db_path = create_db_and_tables(tmp_path / "scouting.db")
    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "select name from sqlite_master where type='table'"
            ).fetchall()
        }

    assert {
        "team_scouting",
        "event_scouting",
        "match_scouting",
        "team_event_scouting",
        "match_alliance_scouting",
        "team_match_scouting",
    }.issubset(tables)


def test_init_scouting_db_cli_creates_database(tmp_path):
    db_path = tmp_path / "scouting.db"
    runner = CliRunner()

    result = runner.invoke(app, ["init-scouting-db", "--path", str(db_path)])

    assert result.exit_code == 0, result.output
    assert db_path.exists()


def test_merge_scouting_features_adds_prefixed_columns(tmp_path):
    db_path = create_db_and_tables(tmp_path / "scouting.db")
    engine = create_scouting_engine(db_path)
    with Session(engine) as session:
        session.add(EventScouting(event_key="2026features", carpet_condition="New"))
        session.add(
            MatchScouting(
                match_key="2026features_qm1",
                event_key="2026features",
                field_fault_occurred=True,
            )
        )
        session.add(TeamScouting(team_key="frc1", drive_base="Swerve", robot_weight_lbs=120.5))
        session.add(
            TeamEventScouting(
                team_key="frc1", event_key="2026features", passed_inspection=True
            )
        )
        session.add(
            MatchAllianceScouting(
                match_key="2026features_qm1",
                alliance_color="red",
                coordinated_auto_run=True,
            )
        )
        session.add(
            TeamMatchScouting(
                team_key="frc1",
                match_key="2026features_qm1",
                alliance_color="red",
                teleop_pieces_scored=7,
                played_defense=True,
            )
        )
        session.commit()

    merged = merge_scouting_features(_feature_table(), db_path)
    first = merged.iloc[0]

    assert first["event_scout_carpet_condition"] == "New"
    assert bool(first["match_scout_field_fault_occurred"])
    assert bool(first["red_alliance_scout_coordinated_auto_run"])
    assert first["red_team_1_pit_drive_base"] == "Swerve"
    assert bool(first["red_team_1_event_scout_passed_inspection"])
    assert int(first["red_team_1_match_scout_teleop_pieces_scored"]) == 7
    assert bool(first["red_team_1_match_scout_played_defense"])


def test_missing_scouting_db_leaves_feature_table_unchanged(tmp_path):
    table = _feature_table()

    merged = merge_scouting_features(table, tmp_path / "missing.db")

    pd.testing.assert_frame_equal(merged, table)


def test_train_features_cli_loads_parquet_and_writes_artifacts(tmp_path):
    path = write_feature_table(_feature_table(), tmp_path / "features.parquet")
    output = tmp_path / "cli_artifacts"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "train-features",
            str(path),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--mini-batch-size",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output / "feature_match_table.csv").exists()


def test_feature_file_errors_are_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="Feature file does not exist"):
        read_feature_table(tmp_path / "missing.parquet")
    with pytest.raises(ValueError, match="Feature table is empty"):
        write_feature_table(pd.DataFrame(), tmp_path / "empty.parquet")
