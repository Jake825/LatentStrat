import json

import numpy as np
import pandas as pd
import pyarrow as pa
from typer.testing import CliRunner

from latentstrat.cli import app
from latentstrat.season.reliability_study import (
    EPOCH_CANDIDATES,
    _write_development_rejection,
    apply_sequential_calibrator,
    classify_interactions,
    confirmation_passes,
    eligible_epoch_counts,
    fit_sequential_calibrator,
    hierarchical_seed_event_bootstrap,
    minimum_campaign_seconds,
    reliability_calibration,
    reliability_metrics,
    select_gradient_threshold,
)


def _optimization_grid() -> pd.DataFrame:
    rows = []
    for threshold, clipping in ((2.5, 0.25), (5.0, 0.05), (10.0, 0.0)):
        for architecture, offset in (
            ("additive", 2.0),
            ("teammate-set", 1.0),
            ("full-match", 0.0),
        ):
            for epoch in (5, 10, 15, 20):
                rows.append(
                    {
                        "architecture": architecture,
                        "max_grad_norm": threshold,
                        "epoch": epoch,
                        "train_loss": 10.0 / epoch + offset,
                        "validation_loss": 11.0 / epoch + offset,
                        "clipping_fraction": clipping,
                        "validation_score_differential_rmse": 100 + offset + 10 / epoch,
                        "validation_winner_brier": 0.18 + offset / 100 + 0.01 / epoch,
                        "validation_winner_log_loss": 0.52 + offset / 100 + 0.01 / epoch,
                    }
                )
    return pd.DataFrame(rows)


def test_development_selection_is_mechanical_and_common_across_architectures():
    grid = _optimization_grid()
    threshold = select_gradient_threshold(
        grid, candidates=(2.5, 5.0, 10.0), warmup_epochs=5
    )
    epochs = eligible_epoch_counts(
        grid, selected_max_grad_norm=threshold, candidates=(15, 20)
    )

    assert threshold == 5.0
    assert epochs == (15, 20)

    confirmation = grid[
        np.isclose(grid["max_grad_norm"], threshold) & grid["epoch"].le(15)
    ].copy()
    assert confirmation_passes(confirmation, grid, selected_max_grad_norm=threshold)


def test_minimum_campaign_runtime_counts_every_fixed_run():
    assert minimum_campaign_seconds(
        10.0,
        development_epochs=40,
        confirmation_epochs=15,
        final_epochs=15,
        clip_count=3,
        architecture_count=3,
        seed_count=3,
        fold_count=3,
    ) == 9_720.0


def test_prediction_seed_identifier_is_parquet_compatible():
    frame = _reliability_predictions()
    frame["seed"] = frame["seed"].astype(str)
    frame.loc[frame.index[0], "seed"] = "control"

    table = pa.Table.from_pandas(frame, preserve_index=False)
    assert pa.types.is_large_string(table.schema.field("seed").type)


def test_rejected_development_grid_writes_a_final_incomplete_manifest(tmp_path):
    source = tmp_path / "features.parquet"
    source.write_bytes(b"fixture")
    output = tmp_path / "study"
    output.mkdir()

    manifest = _write_development_rejection(
        output,
        sources={"features": source},
        tests=(6, 8, 10),
        seeds=(2026, 2027, 2028),
        reason_code="rejected-no-common-epoch",
        reason="No common epoch count.",
        budget_minutes=240,
        selected_clip=5.0,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert EPOCH_CANDIDATES == tuple(range(15, 41))
    assert payload["resolved_config"]["stage"] == "development-rejected"
    assert payload["promotion"]["eligible"] is False


def _prior_calibration_predictions(rows: int = 1_600) -> pd.DataFrame:
    probability = np.linspace(0.05, 0.95, rows)
    outcome = probability >= 0.5
    return pd.DataFrame(
        {
            "test_week": np.full(rows, 6),
            "raw_pred_red_win_probability": probability,
            "actual_red_total_score": np.where(outcome, 100.0, 80.0),
            "actual_blue_total_score": np.where(outcome, 80.0, 100.0),
        }
    )


def test_sequential_calibration_requires_prior_heldout_rows():
    empty = fit_sequential_calibrator(pd.DataFrame())
    fitted = fit_sequential_calibrator(_prior_calibration_predictions())

    assert empty["method"] == "identity-no-prior-heldout-bank"
    assert empty["source_weeks"] == []
    assert fitted["method"] == "platt-logit-prior-heldout-weeks"
    assert fitted["source_weeks"] == [6]
    assert fitted["row_count"] == 1_600

    target = _prior_calibration_predictions(10).rename(
        columns={"raw_pred_red_win_probability": "pred_red_win_probability"}
    )
    calibrated = apply_sequential_calibrator(target, fitted)
    assert calibrated["raw_pred_red_win_probability"].between(0, 1).all()
    assert calibrated["calibrated_pred_red_win_probability"].between(0, 1).all()
    assert calibrated["calibration_source_weeks"].eq("[6]").all()


def _reliability_predictions() -> pd.DataFrame:
    rows = []
    for seed in (2026, 2027, 2028):
        for week in (6, 8):
            for event_number in (1, 2):
                event = f"w{week}e{event_number}"
                for match in range(1, 5):
                    actual_red = 120.0 if match % 2 else 80.0
                    actual_blue = 80.0 if match % 2 else 120.0
                    for model, error, probability in (
                        ("additive", 15.0, 0.62),
                        ("teammate-set", 8.0, 0.72),
                        ("full-match", 5.0, 0.78),
                    ):
                        red_probability = (
                            probability if actual_red > actual_blue else 1 - probability
                        )
                        rows.append(
                            {
                                "scope": "test",
                                "seed": seed,
                                "prediction_level": "seed",
                                "model": model,
                                "architecture": model,
                                "initialization": "pre-2026-prior",
                                "fold_number": 1 if week == 6 else 2,
                                "test_week": week,
                                "event_key": event,
                                "match_key": f"{event}_qm{match}",
                                "evidence_role": "primary",
                                "pred_red_total_score": actual_red + error,
                                "actual_red_total_score": actual_red,
                                "pred_blue_total_score": actual_blue - error,
                                "actual_blue_total_score": actual_blue,
                                "raw_pred_red_win_probability": red_probability,
                                "calibrated_pred_red_win_probability": red_probability,
                                "pred_red_win_probability": red_probability,
                                "observation_slice": "medium",
                            }
                        )
    return pd.DataFrame(rows)


def test_reliability_metrics_keep_raw_and_calibrated_probability_evidence_separate():
    predictions = _reliability_predictions()
    predictions["calibrated_pred_red_win_probability"] = 0.5
    metrics = reliability_metrics(predictions)
    calibration = reliability_calibration(predictions)

    winner = metrics[metrics["metric"].eq("winner_brier")]
    assert set(winner["probability_variant"]) == {"raw", "calibrated"}
    score = metrics[metrics["metric"].eq("score_differential_rmse")]
    assert set(score["probability_variant"]) == {"not-applicable"}
    assert set(calibration["probability_variant"]) == {"raw", "calibrated"}


def test_hierarchical_bootstrap_and_interaction_classification_are_deterministic():
    predictions = _reliability_predictions()
    first = hierarchical_seed_event_bootstrap(predictions, resamples=30, seed=2026)
    second = hierarchical_seed_event_bootstrap(predictions, resamples=30, seed=2026)
    pd.testing.assert_frame_equal(first, second)
    assert (first["delta_mean"] < 0).all()

    paired_rows = []
    for candidate, baseline in (
        ("teammate-set", "additive"),
        ("full-match", "teammate-set"),
    ):
        for experiment_seed in (2026, 2027, 2028):
            for week in (6, 8):
                for metric in (
                    "score_differential_squared_error",
                    "winner_brier",
                    "winner_log_loss",
                ):
                    paired_rows.append(
                        {
                            "candidate": candidate,
                            "baseline": baseline,
                            "experiment_seed": experiment_seed,
                            "test_week": week,
                            "metric": metric,
                            "candidate_loss_mean": 0.8,
                            "baseline_loss_mean": 1.0,
                            "delta_mean": -0.2,
                        }
                    )
    decisions = classify_interactions(first, pd.DataFrame(paired_rows))
    assert decisions["classification"].eq("supported").all()
    assert decisions["provisional"].all()


def test_reliability_cli_defaults_to_smoke_without_training(monkeypatch, tmp_path):
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return type(
            "Result",
            (),
            {
                "output_dir": tmp_path / "out",
                "selected_max_grad_norm": 5.0,
                "selected_epochs": 2,
                "conclusion_status": "awaiting-statbotics",
                "tensorboard_logdir": None,
            },
        )()

    monkeypatch.setattr("latentstrat.cli.run_reliability_study", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "season",
            "run-static-reliability-study",
            "--features",
            str(tmp_path / "features.parquet"),
            "--events",
            str(tmp_path / "events.parquet"),
            "--prior-checkpoint",
            str(tmp_path / "prior.pt"),
            "--output",
            str(tmp_path / "out"),
            "--no-tensorboard",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["kwargs"]["smoke"] is True
    assert captured["kwargs"]["max_epochs"] == 40
    assert captured["kwargs"]["budget_minutes"] == 240
    assert "Reliability study complete" in result.output
