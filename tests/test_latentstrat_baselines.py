import numpy as np
import pandas as pd

from latentstrat.baselines import fit_baselines, match_design_matrix
from latentstrat.config import default_options
from latentstrat.data import Split


def _table():
    return pd.DataFrame(
        {
            "red_team_1_idx": pd.Series([0, 1, 2, 0], dtype="Int64"),
            "red_team_2_idx": pd.Series([1, 2, 3, 1], dtype="Int64"),
            "red_team_3_idx": pd.Series([2, 3, 4, 2], dtype="Int64"),
            "blue_team_1_idx": pd.Series([3, 4, 0, 3], dtype="Int64"),
            "blue_team_2_idx": pd.Series([4, 0, 1, 4], dtype="Int64"),
            "blue_team_3_idx": pd.Series([0, 1, 2, 0], dtype="Int64"),
            "red_total_score": [10.0, 11.0, 12.0, 13.0],
            "blue_total_score": [9.0, 10.0, 11.0, 12.0],
            "win_margin": [1.0, 1.0, 1.0, 1.0],
            "fouls_drawn": [0.0, 1.0, 0.0, 1.0],
        }
    )


def test_match_design_matrix_uses_red_blue_offsets():
    design = match_design_matrix(_table())
    assert design.shape == (4, 10)
    assert design[0].sum() == 6


def test_fit_baselines_returns_mean_and_ridge_metrics():
    opts = default_options()
    table = _table()
    split = Split(
        train_mask=np.array([True, True, True, False]),
        validation_mask=np.array([False, False, False, True]),
        test_mask=np.zeros(4, dtype=bool),
        policy="fixture",
    )

    baselines = fit_baselines(table, split, opts)

    assert baselines.mean.predictions.shape == (4, 4)
    assert baselines.ridge.predictions.shape == (4, 4)
    assert set(baselines.mean.metrics["split"]) == {"train", "validation"}
