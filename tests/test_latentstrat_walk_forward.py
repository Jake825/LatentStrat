import json

import numpy as np
import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

from latentstrat.cli import app
from latentstrat.config import default_options
from latentstrat.features import write_feature_table
from latentstrat.walk_forward import (
    _average_row,
    make_walk_forward_split,
    run_walk_forward_validation,
)
from latentstrat.world_model import TargetSpaceOptions, WorldModelOptions


def _walk_table() -> pd.DataFrame:
    rows = []
    for idx in range(9):
        week = idx // 3 + 1
        event_key = f"2026week{week}"
        red = [1, 2, 3]
        blue = [4, 5, 6]
        rows.append(
            {
                "season": 2026,
                "event_key": event_key,
                "match_key": f"{event_key}_qm{idx + 1}",
                "comp_level": "qm",
                "set_number": 1,
                "match_number": idx + 1,
                "red_team_1_key": f"frc{red[0]}",
                "red_team_2_key": f"frc{red[1]}",
                "red_team_3_key": f"frc{red[2]}",
                "blue_team_1_key": f"frc{blue[0]}",
                "blue_team_2_key": f"frc{blue[1]}",
                "blue_team_3_key": f"frc{blue[2]}",
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
                "raw_event_week": week - 1,
                "event_week": week,
                "red_total_score": 100 + idx,
                "blue_total_score": 90 + idx,
                "red_auto_pts": 20 + idx % 3,
                "red_teleop_pts": 80 + idx,
                "blue_auto_pts": 18 + idx % 3,
                "blue_teleop_pts": 72 + idx,
                "red_foul_pts": idx % 4,
                "blue_foul_pts": (idx + 1) % 4,
                "red_committed_foul_pts": float(idx % 3),
                "blue_committed_foul_pts": float((idx + 1) % 3),
                "red_committed_minor_foul_count": float(idx % 2),
                "blue_committed_minor_foul_count": float((idx + 1) % 2),
                "red_committed_major_foul_count": 0.0,
                "blue_committed_major_foul_count": 0.0,
                "red_atomic_endgame_count": float(idx % 2),
                "blue_atomic_endgame_count": float((idx + 1) % 2),
                "win_margin": 10,
                "fouls_drawn": 0,
                "red_win": True,
            }
        )
    table = pd.DataFrame(rows)
    for color in ("red", "blue"):
        for slot in (1, 2, 3):
            table[f"{color}_team_{slot}_endgame_status"] = "Level1"
    return table


def _prior_checkpoint(path, latent_dim: int | None = None):
    latent_dim = latent_dim or default_options().latent_dim
    torch.save({"embedding_table": torch.zeros((7, latent_dim))}, path)
    return path


def test_walk_forward_split_uses_week_one_then_week_two():
    table = _walk_table()

    split = make_walk_forward_split(table, train_max_week=1, val_exact_week=2)

    assert not np.any(split.train_mask & split.validation_mask)
    assert set(table.loc[split.train_mask, "event_week"]) == {1}
    assert set(table.loc[split.validation_mask, "event_week"]) == {2}


def test_run_walk_forward_writes_fold_and_average_metrics(tmp_path):
    features = write_feature_table(_walk_table(), tmp_path / "features.parquet")
    prior = _prior_checkpoint(tmp_path / "prior.pt")
    rankings = pd.DataFrame(
        {
            "event_key": ["2026week1", "2026week1", "2026week2", "2026week2"],
            "team_key": ["frc1", "frc2", "frc1", "frc2"],
            "qual_rank": [1, 2, 2, 1],
            "matches_played": [3, 3, 3, 3],
            "event_week": [1, 1, 2, 2],
        }
    )
    opts = default_options().model_copy(
        update={"epochs": 1, "mini_batch_size": 4, "use_early_stopping": False}
    )

    result = run_walk_forward_validation(
        features,
        tmp_path / "walk",
        prior_checkpoint=prior,
        opts=opts,
        sidecar_tables={"rankings": rankings},
        verbose=False,
    )

    assert (result.output_dir / "walk_forward_metrics.csv").exists()
    assert (result.output_dir / "walk_forward_history.csv").exists()
    assert (result.output_dir / "walk_forward_predictions.parquet").exists()
    assert (result.output_dir / "config.json").exists()
    config = json.loads((result.output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["workflow"] == "v6-walk-forward"
    assert config["sources"]["features"]["sha256"]
    assert config["sources"]["prior_checkpoint"]["sha256"]
    assert len(result.predictions) == 6
    assert set(result.predictions["val_week"]) == {2, 3}
    assert result.predictions["match_key"].str.startswith("2026week").all()
    assert result.predictions["pred_red_win_probability"].between(0, 1).all()
    assert "AVERAGE" in set(result.metrics["fold_number"])
    first_fold = result.metrics[result.metrics["fold_number"] == 1].iloc[0]
    assert int(first_fold["val_week"]) == 2
    assert np.isfinite(float(first_fold["val_match_mse"]))
    assert np.isfinite(float(first_fold["val_win_brier"]))
    assert np.isfinite(float(first_fold["val_next_match_phase_score_mse"]))
    assert "val_next_match_total_score_mse" in result.metrics.columns
    assert "val_match_accuracy" in result.metrics.columns
    assert "val_win_log_loss" in result.metrics.columns


def test_walk_forward_cli_runs_on_tiny_features(tmp_path):
    features = write_feature_table(_walk_table(), tmp_path / "features.parquet")
    prior = _prior_checkpoint(tmp_path / "prior.pt")
    output = tmp_path / "walk_cli"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "validate-walk-forward",
            "--features",
            str(features),
            "--prior-checkpoint",
            str(prior),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--mini-batch-size",
            "4",
            "--no-tensorboard",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output / "walk_forward_metrics.csv").exists()
    assert (output / "walk_forward_predictions.parquet").exists()
    assert (output / "config.json").exists()


def test_walk_forward_cli_accepts_latent_dim_override(tmp_path):
    features = write_feature_table(_walk_table(), tmp_path / "features.parquet")
    prior = _prior_checkpoint(tmp_path / "prior.pt", latent_dim=4)
    output = tmp_path / "walk_cli"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "validate-walk-forward",
            "--features",
            str(features),
            "--prior-checkpoint",
            str(prior),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--mini-batch-size",
            "4",
            "--latent-dim",
            "4",
            "--no-tensorboard",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output / "walk_forward_metrics.csv").exists()


def test_walk_forward_cli_writes_tensorboard_runs(tmp_path):
    features = write_feature_table(_walk_table(), tmp_path / "features.parquet")
    prior = _prior_checkpoint(tmp_path / "prior.pt")
    output = tmp_path / "walk_tensorboard"
    runs = tmp_path / "runs"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "validate-walk-forward",
            "--features",
            str(features),
            "--prior-checkpoint",
            str(prior),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--mini-batch-size",
            "4",
            "--tensorboard-logdir",
            str(runs),
            "--tensorboard-run-name",
            "walk_test",
        ],
    )

    assert result.exit_code == 0, result.output
    run_dir = runs / "walk_test"
    assert (run_dir / "summary").exists()
    assert any(run_dir.glob("summary/events.out.tfevents.*"))
    assert any(run_dir.glob("fold_*/events.out.tfevents.*"))


def test_walk_forward_rejects_deferred_score_embedding_integration(tmp_path):
    features = write_feature_table(_walk_table(), tmp_path / "features.parquet")
    prior = _prior_checkpoint(tmp_path / "prior.pt")
    events = tmp_path / "events.parquet"
    pd.DataFrame(
        [
            {"key": "2026week1", "event_type": 0},
            {"key": "2026week2", "event_type": 0},
            {"key": "2026week3", "event_type": 0},
        ]
    ).to_parquet(events, index=False)
    opts = default_options().model_copy(
        update={"epochs": 1, "mini_batch_size": 4, "use_early_stopping": False}
    )
    world = WorldModelOptions(
        score_embedding=TargetSpaceOptions(enabled=True, width=4),
        award_embedding=TargetSpaceOptions(enabled=False, width=256),
        rank_embedding=TargetSpaceOptions(enabled=False, width=4),
        pick_embedding=TargetSpaceOptions(enabled=False, width=4),
        target_epochs=1,
    )

    with pytest.raises(ValueError, match="score_embedding is offline-only"):
        run_walk_forward_validation(
            features,
            tmp_path / "walk",
            prior_checkpoint=prior,
            opts=opts,
            world_model_options=world,
            world_model_events_path=events,
            max_validation_week=2,
            verbose=False,
        )


def test_walk_forward_average_row_weights_common_metrics():
    metrics = pd.DataFrame(
        [
            {
                "fold_number": 1,
                "train_weeks": "1-1",
                "val_week": 2,
                "val_next_match_phase_score_mse": 100.0,
                "val_next_match_total_score_mse": 25.0,
                "val_match_accuracy": 0.0,
                "val_win_brier": 0.25,
                "val_win_log_loss": 1.0,
                "val_match_count": 1,
                "val_alliance_score_count": 2,
                "val_phase_score_sse": 200.0,
                "val_phase_score_count": 2,
                "val_total_score_sse": 50.0,
                "val_total_score_count": 2,
                "val_win_brier_sum": 0.25,
                "val_win_log_loss_sum": 1.0,
                "val_win_correct_count": 0,
                "val_win_count": 1,
            },
            {
                "fold_number": 2,
                "train_weeks": "1-2",
                "val_week": 3,
                "val_next_match_phase_score_mse": 1.0,
                "val_next_match_total_score_mse": 4.0,
                "val_match_accuracy": 1.0,
                "val_win_brier": 0.01,
                "val_win_log_loss": 0.1,
                "val_match_count": 3,
                "val_alliance_score_count": 6,
                "val_phase_score_sse": 6.0,
                "val_phase_score_count": 6,
                "val_total_score_sse": 24.0,
                "val_total_score_count": 6,
                "val_win_brier_sum": 0.03,
                "val_win_log_loss_sum": 0.3,
                "val_win_correct_count": 3,
                "val_win_count": 3,
            },
        ]
    )

    row = _average_row(metrics).iloc[0]

    assert row["fold_number"] == "AVERAGE"
    assert row["val_next_match_phase_score_mse"] == 206.0 / 8.0
    assert row["val_next_match_total_score_mse"] == 74.0 / 8.0
    assert row["val_match_accuracy"] == 3.0 / 4.0
    assert row["val_win_brier"] == 0.28 / 4.0
    assert row["val_match_count"] == 4
