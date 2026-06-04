import os
import sqlite3

import pandas as pd
import pytest
import torch
from sqlmodel import Session
from typer.testing import CliRunner

from frc.models import TbaAward
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
from latentstrat.experimental.frozen_targets import TargetSpaceOptions, WorldModelOptions
from latentstrat.features import (
    add_award_features,
    build_event_sidecars,
    build_feature_sidecars,
    event_week_table_from_feature_table,
    load_season_checkpoint_model,
    merge_scouting_features,
    read_feature_table,
    train_feature_table,
    write_feature_table,
)
from latentstrat.model import init_model
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
                "red_auto_pts": 20 + idx % 3,
                "red_teleop_pts": 80 + idx,
                "blue_auto_pts": 18 + idx % 3,
                "blue_teleop_pts": 72 + idx,
                "red_foul_pts": idx % 4,
                "blue_foul_pts": (idx + 1) % 4,
                "win_margin": 10,
                "fouls_drawn": (idx % 4) - ((idx + 1) % 4),
                "red_win": True,
            }
        )
    table = pd.DataFrame(rows)
    for color in ("red", "blue"):
        for slot in (1, 2, 3):
            table[f"{color}_team_{slot}_endgame_status"] = "Level1"
    return table


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
    assert "red_team_1_base_idx" in result.prepared.columns
    assert "red_team_1_event_idx" in result.prepared.columns
    assert result.prepared["red_team_1_idx"].dtype == "Int64"
    assert result.team_index_map["frc1"] == 1
    assert len(result.team_event_index_map) == 6
    assert (output / "feature_match_table.csv").exists()
    assert (output / "feature_history.csv").exists()
    assert (output / "feature_continuous_metrics.csv").exists()
    assert (output / "feature_binary_metrics.csv").exists()
    assert (output / "feature_common_metrics.csv").exists()
    assert (output / "feature_calibration.csv").exists()
    assert (output / "feature_availability_slices.csv").exists()
    assert (output / "feature_endgame_metrics.csv").exists()
    assert (output / "feature_award_metrics.csv").exists()
    assert (output / "feature_set_attention.csv").exists()
    assert (output / "feature_zero_out_diagnostics.csv").exists()
    assert (output / "feature_training_loss.png").exists()
    assert (output / "feature_task_losses.png").exists()
    assert (output / "feature_win_calibration.png").exists()
    assert (output / "feature_attention_entropy.png").exists()
    assert (output / "feature_zero_out_delta_rmse.png").exists()
    assert (output / "feature_team_attention_summary.csv").exists()
    assert (output / "feature_team_zero_out_sensitivity.csv").exists()
    assert (output / "feature_task_precision_evolution.png").exists()
    assert (output / "feature_event_delta_summary.csv").exists()
    assert (output / "feature_event_delta_by_week.png").exists()
    assert not (output / "feature_pma_carry_support.png").exists()
    assert not (output / "feature_zero_out_war.png").exists()
    assert not (output / "feature_embedding_drift.csv").exists()
    assert not (output / "feature_selection_value.csv").exists()
    assert (output / "checkpoint.pt").exists()
    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_schema_version"] == 6
    assert checkpoint["world_model"]["enabled"] is False
    loaded = load_season_checkpoint_model(output / "checkpoint.pt")
    assert loaded.world_model_opts.enabled is False
    common = pd.read_csv(output / "feature_common_metrics.csv")
    assert set(common["split"]) == {"train", "validation"}
    assert "next_match_phase_score_mse" in common.columns
    assert "match_accuracy" in common.columns
    sensitivity = pd.read_csv(output / "feature_team_zero_out_sensitivity.csv")
    assert {"team_key", "target", "delta_rmse", "appearance_count"}.issubset(
        sensitivity.columns
    )


def test_award_ontology_masks_machine_award_non_axes_and_tracks_ei_eligibility():
    table = _feature_table(2)
    table.loc[0, "event_key"] = "2026week1"
    table.loc[1, "event_key"] = "2026week2"
    table.loc[0, "sort_ordinal"] = 1
    table.loc[1, "sort_ordinal"] = 2
    table.loc[0, "red_team_1_key"] = "frc1"
    table.loc[1, "red_team_1_key"] = "frc1"
    table.loc[1, "red_team_2_key"] = "frc2"
    awards = {
        "2026week1": [
            TbaAward(
                name="FIRST Impact Award",
                award_type=0,
                event_key="2026week1",
                recipient_list=[{"team_key": "frc1"}],
            ),
            TbaAward(
                name="Autonomous Award",
                award_type=74,
                event_key="2026week1",
                recipient_list=[{"team_key": "frc1"}],
            ),
        ],
        "2026week2": [
            TbaAward(
                name="Engineering Inspiration Award",
                award_type=9,
                event_key="2026week2",
                recipient_list=[{"team_key": "frc1"}, {"team_key": "frc2"}],
            )
        ],
    }

    out = add_award_features(table, awards)

    assert out.loc[0, "red_team_1_award_auto"] == 1.0
    assert pd.isna(out.loc[0, "red_team_1_award_quality"])
    assert out.loc[0, "red_team_1_award_impact"] == 1.0
    assert out.loc[0, "red_team_1_award_ei"] == 1.0
    assert pd.isna(out.loc[1, "red_team_1_award_impact"])
    assert out.loc[1, "red_team_1_award_ei"] == 1.0
    assert out.loc[1, "red_team_2_award_impact"] == 0.0
    assert out.loc[1, "red_team_2_award_ei"] == 1.0


class _FakeSidecarProvider:
    def get_event_rankings(self, event_key):
        return {
            "rankings": [
                {"team_key": "frc1", "rank": 1, "matches_played": 10},
                {"team_key": "frc2", "rank": 2, "matches_played": 10},
                {"team_key": "frc3", "rank": 3, "matches_played": 10},
                {"team_key": "frc4", "rank": 4, "matches_played": 10},
            ]
        }

    def get_event_alliances(self, event_key):
        return [
            {"picks": ["frc1", "frc3", "frc4"], "status": {"status": "won", "level": "f"}},
            {"picks": ["frc2"], "status": {"status": "eliminated", "level": "sf"}},
        ]


def test_sidecar_generation_preserves_schemas_and_passed_over_team():
    sidecars = build_event_sidecars(["2026features"], _FakeSidecarProvider())

    rankings = sidecars["rankings"]
    selections = sidecars["selections"]
    playoffs = sidecars["playoffs"]

    assert list(rankings[["event_key", "team_key", "qual_rank"]].iloc[0]) == [
        "2026features",
        "frc1",
        1,
    ]
    assert selections.loc[0, "captain_team_key"] == "frc1"
    assert selections.loc[0, "pick_team_key"] == "frc3"
    assert selections.loc[0, "passed_over_team_key"] == "frc2"
    assert int(playoffs.loc[0, "playoff_finish_order"]) == 1


def test_feature_sidecars_inherit_event_week_from_feature_table():
    feature_table = pd.DataFrame(
        {
            "event_key": ["2026features"],
            "raw_event_week": [0],
            "event_week": [1],
        }
    )

    sidecars = build_feature_sidecars(
        ["2026features"],
        _FakeSidecarProvider(),
        2026,
        event_weeks=event_week_table_from_feature_table(feature_table),
    )

    for table in sidecars.values():
        assert "raw_event_week" in table.columns
        assert "event_week" in table.columns
        assert set(table["event_week"].dropna()) == {1}


def test_train_feature_table_writes_phase_visuals_with_prior_and_selection_sidecar(tmp_path):
    table = _feature_table()
    output = tmp_path / "phase_visuals"
    prior = tmp_path / "prior.pt"
    embedding_table = torch.arange(7 * default_options().latent_dim, dtype=torch.float32).reshape(
        7, default_options().latent_dim
    )
    embedding_table = embedding_table / embedding_table.max()
    torch.save({"embedding_table": embedding_table}, prior)
    selections = pd.DataFrame(
        {
            "event_key": ["2026features", "2026features"],
            "alliance_number": [1, 2],
            "captain_team_key": ["frc1", "frc2"],
            "pick_team_key": ["frc3", "frc5"],
            "pick_order": [1, 2],
            "passed_over_team_key": ["frc2", "frc4"],
        }
    )
    opts = default_options().model_copy(
        update={"epochs": 1, "mini_batch_size": 4, "use_early_stopping": False}
    )

    train_feature_table(
        table,
        opts,
        output_dir=output,
        verbose=False,
        prior_checkpoint=prior,
        sidecar_tables={"selections": selections},
    )

    for name in (
        "feature_team_attention_summary.csv",
        "feature_team_zero_out_sensitivity.csv",
        "feature_task_precision_evolution.png",
        "feature_pma_carry_support.png",
        "feature_zero_out_war.png",
        "feature_embedding_drift.csv",
        "feature_embedding_drift_quiver.png",
        "feature_event_delta_summary.csv",
        "feature_event_delta_by_week.png",
        "feature_selection_value.csv",
        "feature_team_value_vs_draft_pick.png",
    ):
        path = output / name
        assert path.exists()
        assert path.stat().st_size > 0
    drift = pd.read_csv(output / "feature_embedding_drift.csv")
    assert {"team_key", "drift_norm", "pca_start_x", "pca_final_x"}.issubset(drift.columns)
    selection_values = pd.read_csv(output / "feature_selection_value.csv")
    assert {"actual_pick_number", "team_value", "pick_team_key"}.issubset(
        selection_values.columns
    )


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
            "--no-early-stopping",
            "--restore-best",
            "--lr-eta-min",
            "0.00001",
            "--loss-log-var-min",
            "-4",
            "--loss-log-var-max",
            "4",
            "--no-tensorboard",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output / "feature_match_table.csv").exists()
    assert (output / "feature_training_diagnostics.json").exists()
    assert (output / "checkpoint.pt").exists()
    history = pd.read_csv(output / "feature_history.csv")
    assert "learning_rate" in history.columns


def test_train_features_cli_freezes_prior_embeddings_and_records_metadata(tmp_path):
    path = write_feature_table(_feature_table(), tmp_path / "features.parquet")
    output = tmp_path / "cli_frozen_artifacts"
    prior = tmp_path / "prior.pt"
    rankings_path = tmp_path / "rankings.parquet"
    playoffs_path = tmp_path / "playoffs.parquet"
    embedding_table = torch.zeros((7, default_options().latent_dim))
    embedding_table[0] = 2.0
    embedding_table[1:] = 1.0
    torch.save({"embedding_table": embedding_table}, prior)
    pd.DataFrame(
        {
            "event_key": ["2026features"] * 4,
            "team_key": ["frc1", "frc2", "frc3", "frc4"],
            "qual_rank": [1, 2, 3, 4],
        }
    ).to_parquet(rankings_path, engine="pyarrow")
    pd.DataFrame(
        {
            "event_key": ["2026features", "2026features"],
            "playoff_finish_order": [1, 2],
            "team_1_key": ["frc1", "frc4"],
            "team_2_key": ["frc2", "frc5"],
            "team_3_key": ["frc3", "frc6"],
        }
    ).to_parquet(playoffs_path, engine="pyarrow")
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
            "--split-policy",
            "stratified-event-comp",
            "--freeze-team-embeddings",
            "--prior-checkpoint",
            str(prior),
            "--rankings-sidecar",
            str(rankings_path),
            "--playoffs-sidecar",
            str(playoffs_path),
            "--no-tensorboard",
        ],
    )

    assert result.exit_code == 0, result.output
    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["split_policy"] == "stratified-event-comp"
    assert checkpoint["frozen_embedding_tables"] == ["Z_base", "Z_event"]
    assert checkpoint["sidecar_tables"] == ["playoffs", "rankings"]
    assert torch.allclose(checkpoint["model_state_dict"]["Z_base.weight"][:7], embedding_table)
    assert torch.count_nonzero(checkpoint["model_state_dict"]["Z_event.weight"]).item() == 0
    assert (output / "feature_calibration.csv").exists()
    assert (output / "feature_rankings_sidecar.csv").exists()
    assert (output / "feature_playoffs_sidecar.csv").exists()
    assert (output / "feature_training_loss.png").exists()


def test_train_features_cli_rejects_freezing_random_embeddings(tmp_path):
    path = write_feature_table(_feature_table(), tmp_path / "features.parquet")

    result = CliRunner().invoke(
        app,
        [
            "train-features",
            str(path),
            "--freeze-team-embeddings",
            "--no-tensorboard",
        ],
    )

    assert result.exit_code != 0
    assert "requires --prior-checkpoint or --checkpoint" in result.output


def test_consolidate_event_cli_writes_embedding_store(tmp_path):
    path = write_feature_table(_feature_table(), tmp_path / "features.parquet")
    output = tmp_path / "cli_artifacts"
    db_path = tmp_path / "embeddings.sqlite"
    runner = CliRunner()
    train_result = runner.invoke(
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
            "--no-tensorboard",
        ],
    )
    assert train_result.exit_code == 0, train_result.output

    result = runner.invoke(
        app,
        [
            "consolidate-event",
            str(output / "checkpoint.pt"),
            "--event-key",
            "2026features",
            "--embedding-db",
            str(db_path),
            "--delta-weeks",
            "2.0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert db_path.exists()
    assert (output / "checkpoint_consolidated.pt").exists()


def test_feature_file_errors_are_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="Feature file does not exist"):
        read_feature_table(tmp_path / "missing.parquet")
    with pytest.raises(ValueError, match="Feature table is empty"):
        write_feature_table(pd.DataFrame(), tmp_path / "empty.parquet")


def test_active_integrated_v6_checkpoint_training_requires_world_model_bundle():
    opts = default_options()
    model = init_model(
        7,
        opts.latent_dim,
        len(opts.target_map),
        len(opts.binary_targets),
        opts,
        num_event_teams=7,
        world_model_opts=WorldModelOptions(
            award_embedding=TargetSpaceOptions(enabled=True, width=256)
        ),
    )

    with pytest.raises(ValueError, match="requires frozen_target_bundle"):
        train_feature_table(_feature_table(), opts, initial_model=model, verbose=False)
