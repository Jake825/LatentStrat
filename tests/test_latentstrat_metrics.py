import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from latentstrat.cli import app
from latentstrat.season.metrics import (
    evaluate_prediction_tables,
    event_cluster_paired_bootstrap,
    pair_prediction_tables,
)
from latentstrat.season.statbotics_baseline import (
    STATBOTICS_PREDICTION_COLUMNS,
    collect_statbotics_predictions,
)


class FakeStatboticsProvider:
    def __init__(self, rows, cache_path: Path | None = None):
        self.rows = rows
        self.calls = []
        self.cache_path = cache_path or Path("fake-statbotics.sqlite")

    def get_matches(self, **filters):
        self.calls.append(filters)
        offset = int(filters["offset"])
        limit = int(filters["limit"])
        return self.rows[offset : offset + limit]


def _stat_row(key: str, event: str, red: float, blue: float, probability: float):
    return {
        "year": 2026,
        "event": event,
        "key": key,
        "pred": {"red_score": red, "blue_score": blue, "red_win_prob": probability},
        "result": {"red_score": 9999, "blue_score": -9999},
    }


def _candidate_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_number": [1, 1, 2, 2],
            "val_week": [2, 2, 3, 3],
            "event_key": ["2026e1", "2026e1", "2026e2", "2026e3"],
            "match_key": ["2026e1_qm1", "2026e1_qm2", "2026e2_qm1", "2026e3_qm1"],
            "pred_red_total_score": [100.0, 90.0, 79.0, 50.0],
            "actual_red_total_score": [100.0, 80.0, 75.0, 60.0],
            "pred_blue_total_score": [80.0, 90.0, 71.0, 55.0],
            "actual_blue_total_score": [80.0, 80.0, 70.0, 65.0],
            "pred_red_win_probability": [0.8, 0.5, 0.7, 0.4],
        }
    )


def _statbotics_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026, 2026, 2026, 2026],
            "event_key": ["2026e1", "2026e1", "2026e2", "2026e4"],
            "match_key": ["2026e1_qm1", "2026e1_qm2", "2026e2_qm1", "2026e4_qm1"],
            "statbotics_pred_red_score": [95.0, 82.0, 72.0, 60.0],
            "statbotics_pred_blue_score": [82.0, 78.0, 68.0, 50.0],
            "statbotics_pred_red_win_probability": [0.7, 0.55, 0.65, 0.8],
        }
    )


def test_statbotics_collection_paginates_and_keeps_only_pre_match_fields():
    rows = [
        _stat_row("2026e1_qm1", "2026e1", 10, 8, 0.7),
        _stat_row("2026e1_qm2", "2026e1", 9, 9, 0.5),
        _stat_row("2026e2_qm1", "2026e2", 6, 11, 0.2),
    ]
    provider = FakeStatboticsProvider(rows)

    table = collect_statbotics_predictions(provider, 2026, page_size=2)

    assert list(table.columns) == list(STATBOTICS_PREDICTION_COLUMNS)
    assert len(provider.calls) == 2
    assert provider.calls[0]["offset"] == 0
    assert provider.calls[1]["offset"] == 2
    assert table.loc[table["match_key"] == "2026e1_qm1", "statbotics_pred_red_score"].item() == 10
    assert not any("result" in column for column in table.columns)


def test_statbotics_collection_rejects_response_drift_and_duplicates():
    malformed = [{"year": 2026, "event": "2026e1", "key": "2026e1_qm1", "pred": {}}]
    with pytest.raises(ValueError, match="pred keys"):
        collect_statbotics_predictions(FakeStatboticsProvider(malformed), 2026)

    duplicate = _stat_row("2026e1_qm1", "2026e1", 10, 8, 0.7)
    with pytest.raises(ValueError, match="duplicate"):
        collect_statbotics_predictions(FakeStatboticsProvider([duplicate, duplicate]), 2026)


def test_prediction_evaluation_reports_coverage_and_excludes_ties():
    result = evaluate_prediction_tables(
        _candidate_table(), _statbotics_table(), resamples=50, seed=7
    )

    assert result.coverage["matched_rows"] == 3
    assert result.coverage["candidate_only_rows"] == 1
    assert result.coverage["statbotics_only_rows"] == 1
    assert result.paired_predictions["is_tie"].sum() == 1
    winner = result.metrics[
        (result.metrics["scope"] == "aggregate")
        & (result.metrics["model"] == "latentstrat")
        & (result.metrics["metric"] == "winner_brier")
    ].iloc[0]
    score = result.metrics[
        (result.metrics["scope"] == "aggregate")
        & (result.metrics["model"] == "latentstrat")
        & (result.metrics["metric"] == "alliance_score_mae")
    ].iloc[0]
    assert winner["count"] == 2
    assert score["count"] == 6
    aggregate_calibration = result.calibration[result.calibration["scope"] == "aggregate"]
    assert len(aggregate_calibration) == 20
    assert aggregate_calibration.groupby("model")["count"].sum().to_dict() == {
        "latentstrat": 2,
        "statbotics": 2,
    }


def test_event_cluster_bootstrap_is_deterministic():
    paired, _ = pair_prediction_tables(_candidate_table(), _statbotics_table())
    first = event_cluster_paired_bootstrap(paired, resamples=100, seed=11)
    second = event_cluster_paired_bootstrap(paired, resamples=100, seed=11)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["cluster"]) == {"event_key"}
    assert first["probability_of_improvement"].between(0, 1).all()


def test_evaluate_predictions_cli_writes_nonpromotion_artifacts(tmp_path):
    candidate = tmp_path / "candidate.parquet"
    statbotics = tmp_path / "statbotics.parquet"
    output = tmp_path / "evaluation"
    _candidate_table().to_parquet(candidate, index=False)
    _statbotics_table().to_parquet(statbotics, index=False)

    result = CliRunner().invoke(
        app,
        [
            "artifacts",
            "evaluate-predictions",
            "--candidate",
            str(candidate),
            "--statbotics",
            str(statbotics),
            "--output",
            str(output),
            "--resamples",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    for name in (
        "paired_predictions.parquet",
        "metrics.csv",
        "calibration.csv",
        "paired_bootstrap.csv",
        "coverage.json",
        "manifest.json",
    ):
        assert (output / name).exists()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["promotion"]["eligible"] is False
    assert "selected epochs" in manifest["promotion"]["note"]


def test_build_statbotics_baseline_cli_uses_provider_without_training(tmp_path, monkeypatch):
    provider = FakeStatboticsProvider(
        [_stat_row("2026e1_qm1", "2026e1", 10, 8, 0.7)],
        cache_path=tmp_path / "cache.sqlite",
    )
    monkeypatch.setattr("latentstrat.cli._statbotics_provider", lambda: provider)
    output = tmp_path / "statbotics.parquet"

    result = CliRunner().invoke(
        app,
        [
            "season",
            "build-statbotics-baseline",
            "--season",
            "2026",
            "--output",
            str(output),
            "--page-size",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert output.with_suffix(".manifest.json").exists()
    saved = pd.read_parquet(output)
    np.testing.assert_allclose(saved["statbotics_pred_red_score"], [10.0])
