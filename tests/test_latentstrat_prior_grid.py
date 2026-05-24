from __future__ import annotations

import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

from latentstrat.cli import app
from latentstrat.config import PriorOpts
from latentstrat.pretrain_features import (
    CULTURE_TARGET_COLUMNS,
    NORM_EPA_OBSERVED_COLUMNS,
    NORM_EPA_TARGET_COLUMNS,
)
from latentstrat.prior_grid import (
    PriorGridDataset,
    make_prior_grid_split,
    run_prior_grid,
    trainable_parameter_count,
)
from latentstrat.prior_model import TeamPriorDistiller


def _grid_table(row_count: int = 8) -> pd.DataFrame:
    archetypes = ["anchor", "ghost", "sibling", "future"]
    team_numbers = list(range(row_count))
    return pd.DataFrame(
        {
            "team_number": team_numbers,
            "team_key": [f"frc{idx}" for idx in team_numbers],
            "archetype": [
                "ghost_token" if idx == 0 else archetypes[(idx - 1) % len(archetypes)]
                for idx in team_numbers
            ],
            "openai_narrative_vector": [
                [float((idx + offset) % 13) / 13.0 for offset in range(256)]
                for idx in team_numbers
            ],
            **{
                column: [
                    float("nan") if idx == 0 else float(idx + axis) / 10.0
                    for idx in team_numbers
                ]
                for axis, column in enumerate(NORM_EPA_TARGET_COLUMNS)
            },
            **{
                column: [idx != 0 for idx in team_numbers]
                for column in NORM_EPA_OBSERVED_COLUMNS
            },
            **{
                column: [0.0 if idx == 0 else float(idx + axis) for idx in team_numbers]
                for axis, column in enumerate(CULTURE_TARGET_COLUMNS)
            },
        }
    )


def test_production_prior_distiller_shapes_and_components():
    model = TeamPriorDistiller(PriorOpts(max_team_number=10, latent_dim=3))

    prediction, pred_norm_epa, pred_culture, latent = model(torch.tensor([0, 1, 2]))

    assert prediction.shape == (3, 256)
    assert pred_norm_epa.shape == (3, 4)
    assert pred_culture.shape == (3, 7)
    assert latent.shape == (3, 3)
    assert set(dict(model.named_children())) == {
        "team_embedding",
        "decoder",
        "openai_head",
        "norm_epa_head",
        "culture_head",
    }
    assert model.team_embedding.padding_idx is None
    assert not torch.allclose(model.team_embedding.weight[0], torch.zeros(3))


def test_grid_dataset_requires_v56_team_numbers():
    with pytest.raises(KeyError, match="missing team_number"):
        PriorGridDataset(
            pd.DataFrame(
                {
                    "team_key": ["frc1"],
                    "openai_narrative_vector": [[0.0] * 256],
                }
            )
        )


def test_grid_dataset_requires_v564_targets():
    with pytest.raises(KeyError, match="Missing: .*norm_epa_t_minus_1"):
        PriorGridDataset(
            pd.DataFrame(
                {
                    "team_number": [1],
                    "openai_narrative_vector": [[0.0] * 256],
                    **{
                        column: [0.0]
                        for column in NORM_EPA_TARGET_COLUMNS
                        if column != "norm_epa_t_minus_1"
                    },
                    **{column: [True] for column in NORM_EPA_OBSERVED_COLUMNS},
                    **{column: [0.0] for column in CULTURE_TARGET_COLUMNS},
                }
            )
        )


def test_prior_grid_split_is_deterministic():
    table = _grid_table(20)

    first = make_prior_grid_split(table, seed=123)
    second = make_prior_grid_split(table, seed=123)

    assert first.train_mask.tolist() == second.train_mask.tolist()
    assert first.validation_mask.tolist() == second.validation_mask.tolist()
    assert first.train_mask.any()
    assert second.validation_mask.any()
    ghost_row = table.index[table["team_number"] == 0][0]
    assert first.train_mask[ghost_row]
    assert not first.validation_mask[ghost_row]


def test_prior_grid_emits_one_result_per_latent_dim(tmp_path):
    features = tmp_path / "prior_features.parquet"
    output = tmp_path / "grid"
    _grid_table(12).to_parquet(features, engine="pyarrow", index=False)

    result = run_prior_grid(
        features,
        output,
        latent_dims=(2, 4),
        epochs=1,
        batch_size=4,
        seed=123,
    )

    assert result.results["latent_dim"].tolist() == [2, 4]
    assert result.results["status"].tolist() == ["ok", "ok"]
    row_count = result.results["train_rows"].iloc[0] + result.results["validation_rows"].iloc[0]
    assert int(row_count) == 12
    assert len(result.history) == 2
    assert (output / "prior_elbow_curve.png").stat().st_size > 0


def test_prior_grid_accepts_team_zero_as_trainable_ghost_row(tmp_path):
    table = _grid_table(6)

    dataset = PriorGridDataset(table)
    split = make_prior_grid_split(table, seed=123)

    assert int(dataset.team_numbers[0]) == 0
    assert split.train_mask[0]
    assert not split.validation_mask[0]


def test_prior_grid_parameter_count_increases_with_latent_dim():
    small = TeamPriorDistiller(PriorOpts(max_team_number=10, latent_dim=2))
    wide = TeamPriorDistiller(PriorOpts(max_team_number=10, latent_dim=4))

    assert trainable_parameter_count(wide) > trainable_parameter_count(small)


def test_prior_grid_records_oom_failure_and_continues(tmp_path, monkeypatch):
    features = tmp_path / "prior_features.parquet"
    _grid_table(8).to_parquet(features, engine="pyarrow", index=False)
    calls = {"count": 0}

    def fake_train(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("CUDA out of memory")
        return {
            "latent_dim": kwargs["latent_dim"],
            "status": "ok",
            "epochs": kwargs["epochs"],
            "train_rows": 4,
            "validation_rows": 4,
            "final_train_mse": 0.1,
            "final_validation_mse": 0.2,
            "best_validation_mse": 0.2,
            "final_train_openai_mse": 0.1,
            "final_validation_openai_mse": 0.2,
            "best_validation_openai_mse": 0.2,
            "final_train_norm_epa_mse": 0.3,
            "final_validation_norm_epa_mse": 0.4,
            "best_validation_norm_epa_mse": 0.4,
            "final_train_culture_mse": 0.5,
            "final_validation_culture_mse": 0.6,
            "best_validation_culture_mse": 0.6,
            "final_log_var_openai": 0.0,
            "final_log_var_epa": 0.0,
            "final_log_var_culture_mean": 0.0,
            "epochs_to_converge": 1,
            "trainable_parameters": 10,
            "elapsed_seconds": 0.01,
            "peak_vram_mb": float("nan"),
            "estimated_model_mb": 0.01,
            "estimated_optimizer_mb": 0.02,
            "estimated_training_state_mb": 0.03,
        }, pd.DataFrame(
            {
                "latent_dim": [kwargs["latent_dim"]],
                "epoch": [1],
                "train_loss": [0.1],
                "train_openai_mse": [0.1],
                "train_norm_epa_mse": [0.3],
                "train_culture_mse": [0.5],
                "validation_loss": [0.2],
                "validation_openai_mse": [0.2],
                "validation_norm_epa_mse": [0.4],
                "validation_culture_mse": [0.6],
                "log_var_openai": [0.0],
                "log_var_epa": [0.0],
                "log_var_culture_mean": [0.0],
            }
        )

    monkeypatch.setattr("latentstrat.prior_grid.train_prior_grid_run", fake_train)

    result = run_prior_grid(
        features,
        tmp_path / "grid",
        latent_dims=(2, 4),
        epochs=1,
    )

    assert result.results["status"].tolist() == ["oom", "ok"]
    assert len(result.failures) == 1
    assert result.failures["status"].iloc[0] == "oom"


def test_prior_grid_cli_writes_artifacts(tmp_path):
    features = tmp_path / "prior_features.parquet"
    output = tmp_path / "grid"
    _grid_table(8).to_parquet(features, engine="pyarrow", index=False)

    result = CliRunner().invoke(
        app,
        [
            "run-prior-grid",
            "--features",
            str(features),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--latent-dims",
            "2,4",
            "--batch-size",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    for name in (
        "prior_grid_results.csv",
        "prior_grid_history.csv",
        "prior_grid_failures.csv",
        "prior_elbow_curve.png",
    ):
        path = output / name
        assert path.exists()
        assert path.stat().st_size > 0


def test_prior_grid_cli_rejects_removed_dictionary_dims_option(tmp_path):
    features = tmp_path / "prior_features.parquet"
    _grid_table(8).to_parquet(features, engine="pyarrow", index=False)

    result = CliRunner().invoke(
        app,
        [
            "run-prior-grid",
            "--features",
            str(features),
            "--output",
            str(tmp_path / "grid"),
            "--epochs",
            "1",
            "--dictionary-dims",
            "4",
            "--latent-dims",
            "2",
        ],
    )

    assert result.exit_code != 0
    assert "dictionary-dims" in result.output


def test_prior_grid_cli_rejects_v55_shaped_features(tmp_path):
    features = tmp_path / "prior_features_v55.parquet"
    pd.DataFrame(
        {
            "team_key": ["frc1"],
            "openai_narrative_vector": [[0.0] * 256],
        }
    ).to_parquet(features, engine="pyarrow", index=False)

    result = CliRunner().invoke(
        app,
        [
            "run-prior-grid",
            "--features",
            str(features),
            "--output",
            str(tmp_path / "grid"),
            "--epochs",
            "1",
            "--latent-dims",
            "2",
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, KeyError)
    assert "missing team_number" in str(result.exception)
