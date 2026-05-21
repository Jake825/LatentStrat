"""Mean and ridge match-OPR baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge

from latentstrat.config import LatentStratOptions, default_options
from latentstrat.data import Split, target_matrix


@dataclass
class MeanBaseline:
    values: np.ndarray
    predictions: np.ndarray
    metrics: pd.DataFrame


@dataclass
class RidgeBaseline:
    lambda_: float
    coefficients: np.ndarray
    num_teams: int
    predictions: np.ndarray
    metrics: pd.DataFrame


@dataclass
class Baselines:
    target_names: list[str]
    mean: MeanBaseline
    ridge: RidgeBaseline


def _idx_values(table: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = table[columns].to_numpy(dtype=object)
    if pd.isna(values).any():
        raise ValueError(f"Missing team indices in columns: {', '.join(columns)}")
    return values.astype(int)


def match_design_matrix(table: pd.DataFrame) -> sp.csr_matrix:
    red_columns = ["red_team_1_idx", "red_team_2_idx", "red_team_3_idx"]
    if all(column in table.columns for column in red_columns):
        blue_columns = ["blue_team_1_idx", "blue_team_2_idx", "blue_team_3_idx"]
        red_idx = _idx_values(table, red_columns)
        blue_idx = _idx_values(table, blue_columns)
        team_idx = np.concatenate([red_idx, blue_idx], axis=1)
        num_teams = int(team_idx.max()) + 1
        offsets = np.array([0, 0, 0, num_teams, num_teams, num_teams])
        cols = team_idx + offsets
        width = 2 * num_teams
    else:
        columns = ["team_1_idx", "team_2_idx", "team_3_idx"]
        team_idx = _idx_values(table, columns)
        num_teams = int(team_idx.max()) + 1
        cols = team_idx
        width = num_teams
    row_ids = np.repeat(np.arange(len(table)), cols.shape[1])
    col_ids = cols.reshape(-1)
    data = np.ones_like(col_ids, dtype=float)
    return sp.coo_matrix((data, (row_ids, col_ids)), shape=(len(table), width)).tocsr()


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    target_names: list[str],
) -> pd.DataFrame:
    rows = []
    for split_name, mask in (("train", train_mask), ("validation", validation_mask)):
        if not np.any(mask):
            rmse = np.full(len(target_names), np.nan)
            mae = np.full(len(target_names), np.nan)
        else:
            err = predicted[mask] - actual[mask]
            rmse = np.sqrt(np.nanmean(err**2, axis=0))
            mae = np.nanmean(np.abs(err), axis=0)
        for target, rmse_value, mae_value in zip(target_names, rmse, mae, strict=True):
            rows.append(
                {
                    "split": split_name,
                    "target": target,
                    "rmse": float(rmse_value),
                    "mae": float(mae_value),
                }
            )
    return pd.DataFrame(rows)


def fit_ridge_baseline(
    design_matrix: sp.spmatrix, targets: np.ndarray, l2_lambda: float
) -> np.ndarray:
    model = Ridge(alpha=l2_lambda, fit_intercept=False, solver="sparse_cg")
    model.fit(design_matrix, np.asarray(targets, dtype=float))
    coefficients = np.asarray(model.coef_, dtype=float)
    if coefficients.ndim == 1:
        coefficients = coefficients[None, :]
    return coefficients.T


def fit_baselines(
    table: pd.DataFrame,
    split: Split,
    opts: LatentStratOptions | None = None,
    *,
    ridge_lambda: float | None = None,
) -> Baselines:
    opts = opts or default_options()
    ridge_lambda = opts.l2 if ridge_lambda is None else ridge_lambda
    target_names = [mapping.target_name for mapping in opts.target_map]
    targets = target_matrix(table, target_names)
    design = match_design_matrix(table)
    train_targets = targets[split.train_mask]
    mean_values = np.nanmean(train_targets, axis=0)
    mean_predictions = np.tile(mean_values, (len(table), 1))
    coefficients = fit_ridge_baseline(design[split.train_mask], train_targets, ridge_lambda)
    if coefficients.ndim == 1:
        coefficients = coefficients[:, None]
    ridge_predictions = np.asarray(design @ coefficients)
    return Baselines(
        target_names=target_names,
        mean=MeanBaseline(
            values=mean_values,
            predictions=mean_predictions,
            metrics=regression_metrics(
                targets, mean_predictions, split.train_mask, split.validation_mask, target_names
            ),
        ),
        ridge=RidgeBaseline(
            lambda_=ridge_lambda,
            coefficients=coefficients,
            num_teams=(
                design.shape[1] // 2 if "red_team_1_idx" in table.columns else design.shape[1]
            ),
            predictions=ridge_predictions,
            metrics=regression_metrics(
                targets, ridge_predictions, split.train_mask, split.validation_mask, target_names
            ),
        ),
    )
