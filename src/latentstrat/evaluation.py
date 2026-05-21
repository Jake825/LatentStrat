"""Evaluation metrics and diagnostics for LatentStrat V4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from latentstrat.baselines import Baselines
from latentstrat.config import LatentStratOptions, default_options
from latentstrat.data import Split, TargetStats, target_matrix
from latentstrat.model import SetTransformerModel, require_torch
from latentstrat.training import match_team_matrices


@dataclass
class EvaluationReport:
    parameter_count: int
    rows_per_parameter: float
    continuous_metrics: pd.DataFrame
    binary_metrics: pd.DataFrame
    calibration: pd.DataFrame
    slices: dict[str, pd.DataFrame]
    set_attention: pd.DataFrame
    zero_out_diagnostics: pd.DataFrame
    baselines: Baselines | None = None


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def continuous_metrics(
    actual: np.ndarray, predicted: np.ndarray, split: Split, target_names: list[str]
) -> pd.DataFrame:
    rows = []
    for split_name, mask in (("train", split.train_mask), ("validation", split.validation_mask)):
        if not np.any(mask):
            rmse = np.full(len(target_names), np.nan)
            mae = np.full(len(target_names), np.nan)
        else:
            err = predicted[mask] - actual[mask]
            rmse = np.sqrt(np.nanmean(err**2, axis=0))
            mae = np.nanmean(np.abs(err), axis=0)
        for target, rmse_value, mae_value in zip(target_names, rmse, mae, strict=True):
            rows.append({"split": split_name, "target": target, "rmse": rmse_value, "mae": mae_value})
    return pd.DataFrame(rows)


def binary_metrics(
    actual: np.ndarray, predicted: np.ndarray, split: Split, binary_names: list[str]
) -> pd.DataFrame:
    rows = []
    if not binary_names:
        return pd.DataFrame(columns=["split", "target", "brier", "log_loss"])
    for split_name, mask in (("train", split.train_mask), ("validation", split.validation_mask)):
        if not np.any(mask):
            brier = np.full(len(binary_names), np.nan)
            log_loss = np.full(len(binary_names), np.nan)
        else:
            clamped = np.clip(predicted[mask], np.finfo(float).eps, 1 - np.finfo(float).eps)
            actual_split = actual[mask]
            brier = np.nanmean((clamped - actual_split) ** 2, axis=0)
            log_loss = -np.nanmean(
                actual_split * np.log(clamped) + (1 - actual_split) * np.log(1 - clamped), axis=0
            )
        for target, brier_value, log_value in zip(binary_names, brier, log_loss, strict=True):
            rows.append(
                {"split": split_name, "target": target, "brier": brier_value, "log_loss": log_value}
            )
    return pd.DataFrame(rows)


def calibration_table(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray, names: list[str]) -> pd.DataFrame:
    rows = []
    if not names or not np.any(mask):
        return pd.DataFrame(columns=["target", "bin_low", "bin_high", "count", "mean_probability", "observed_rate"])
    edges = np.linspace(0, 1, 6)
    for target_idx, name in enumerate(names):
        probs = predicted[mask, target_idx]
        outcomes = actual[mask, target_idx]
        bin_idx = np.digitize(probs, edges, right=True)
        bin_idx[probs == 0] = 1
        for idx in range(1, len(edges)):
            bin_mask = bin_idx == idx
            rows.append(
                {
                    "target": name,
                    "bin_low": edges[idx - 1],
                    "bin_high": edges[idx],
                    "count": int(np.sum(bin_mask)),
                    "mean_probability": float(np.nanmean(probs[bin_mask])) if np.any(bin_mask) else np.nan,
                    "observed_rate": float(np.nanmean(outcomes[bin_mask])) if np.any(bin_mask) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _team_idx_matrix(table: pd.DataFrame) -> np.ndarray:
    columns = [
        "red_team_1_idx",
        "red_team_2_idx",
        "red_team_3_idx",
        "blue_team_1_idx",
        "blue_team_2_idx",
        "blue_team_3_idx",
    ]
    return table[columns].to_numpy(dtype=int)


def _row_training_match_counts(table: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
    team_idx = _team_idx_matrix(table)
    counts = np.bincount(team_idx[train_mask].reshape(-1), minlength=int(team_idx.max()) + 1)
    return counts[team_idx]


def _row_has_first_event_team(table: pd.DataFrame) -> np.ndarray:
    team_idx = _team_idx_matrix(table)
    first_event = np.array([""] * (int(team_idx.max()) + 1), dtype=object)
    order = np.argsort(table["sort_ordinal"].to_numpy())
    events = table["event_key"].astype(str).to_numpy()
    for row in order:
        for team in team_idx[row]:
            if first_event[team] == "":
                first_event[team] = events[row]
    return np.array([np.any(first_event[team_idx[row]] == events[row]) for row in range(len(table))])


def slice_metrics(
    table: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray, split: Split, target_names: list[str]
) -> dict[str, pd.DataFrame]:
    train_counts = _row_training_match_counts(table, split.train_mask)
    unseen_counts = np.sum(train_counts == 0, axis=1)
    low_data_counts = np.sum(train_counts <= 5, axis=1)
    first_event = _row_has_first_event_team(table)
    masks = {
        "all_validation": split.validation_mask,
        "all_teams_seen_in_training": split.validation_mask & (unseen_counts == 0),
        "one_unseen_team": split.validation_mask & (unseen_counts == 1),
        "two_or_more_unseen_teams": split.validation_mask & (unseen_counts >= 2),
        "at_least_one_low_data_team": split.validation_mask & (low_data_counts >= 1),
        "two_or_more_low_data_teams": split.validation_mask & (low_data_counts >= 2),
        "first_observed_event": split.validation_mask & first_event,
    }
    rows = []
    for name, mask in masks.items():
        if np.any(mask):
            err = predicted[mask] - actual[mask]
            rmse = np.sqrt(np.nanmean(err**2, axis=0))
            mae = np.nanmean(np.abs(err), axis=0)
        else:
            rmse = np.full(len(target_names), np.nan)
            mae = np.full(len(target_names), np.nan)
        for target, rmse_value, mae_value in zip(target_names, rmse, mae, strict=True):
            rows.append({"slice": name, "target": target, "rmse": rmse_value, "mae": mae_value})
    return {"availability": pd.DataFrame(rows)}


def set_attention_table(table: pd.DataFrame, split: Split, red_weights: np.ndarray, blue_weights: np.ndarray) -> pd.DataFrame:
    rows = []
    split_names = np.array(["other"] * len(table), dtype=object)
    split_names[split.train_mask] = "train"
    split_names[split.validation_mask] = "validation"
    split_names[split.test_mask] = "test"
    for color, weights, prefix in (("red", red_weights, "red"), ("blue", blue_weights, "blue")):
        team_cols = [f"{prefix}_team_1_key", f"{prefix}_team_2_key", f"{prefix}_team_3_key"]
        keys = table[team_cols].astype(str).to_numpy()
        entropy = -np.sum(weights * np.log(np.maximum(weights, np.finfo(float).eps)), axis=1)
        max_slot = np.argmax(weights, axis=1)
        for idx in range(len(table)):
            rows.append(
                {
                    "row_index": idx + 1,
                    "split": split_names[idx],
                    "alliance_color": color,
                    "event_key": table["event_key"].iloc[idx],
                    "match_key": table["match_key"].iloc[idx],
                    "team_1_key": keys[idx, 0],
                    "team_2_key": keys[idx, 1],
                    "team_3_key": keys[idx, 2],
                    "team_1_attention": weights[idx, 0],
                    "team_2_attention": weights[idx, 1],
                    "team_3_attention": weights[idx, 2],
                    "max_attention_slot": int(max_slot[idx] + 1),
                    "max_attention_team_key": keys[idx, max_slot[idx]],
                    "max_attention_weight": weights[idx, max_slot[idx]],
                    "attention_entropy": entropy[idx],
                }
            )
    return pd.DataFrame(rows)


def _predict_cont(
    model: SetTransformerModel,
    red: np.ndarray,
    blue: np.ndarray,
    target_stats: TargetStats,
    *,
    pma_mode: str = "learned",
    red_zero_slot: int = 0,
    blue_zero_slot: int = 0,
) -> np.ndarray:
    torch = require_torch()
    with torch.no_grad():
        pred = model(
            torch.as_tensor(red, dtype=torch.long),
            torch.as_tensor(blue, dtype=torch.long),
            pma_mode=pma_mode,
            red_zero_slot=red_zero_slot,
            blue_zero_slot=blue_zero_slot,
        )
    pred_z = pred.cont_z.detach().cpu().numpy()
    return pred_z * target_stats.sigma + target_stats.mu


def zero_out_diagnostics(
    model: SetTransformerModel,
    table: pd.DataFrame,
    split: Split,
    target_stats: TargetStats,
    opts: LatentStratOptions,
) -> pd.DataFrame:
    columns = ["scenario", "target", "baseline_rmse", "scenario_rmse", "delta_rmse"]
    if not np.any(split.validation_mask):
        return pd.DataFrame(columns=columns)
    red, blue = match_team_matrices(table)
    actual = target_matrix(table, [mapping.target_name for mapping in opts.target_map])
    baseline = _predict_cont(model, red, blue, target_stats)
    baseline_rmse = np.sqrt(np.nanmean((baseline[split.validation_mask] - actual[split.validation_mask]) ** 2, axis=0))
    scenarios = [("uniform_pma", "uniform", 0, 0)] + [
        (f"zero_red_slot_{slot}", "learned", slot, 0) for slot in (1, 2, 3)
    ] + [(f"zero_blue_slot_{slot}", "learned", 0, slot) for slot in (1, 2, 3)]
    rows = []
    for scenario, pma_mode, red_slot, blue_slot in scenarios:
        predicted = _predict_cont(
            model, red, blue, target_stats, pma_mode=pma_mode, red_zero_slot=red_slot, blue_zero_slot=blue_slot
        )
        rmse = np.sqrt(np.nanmean((predicted[split.validation_mask] - actual[split.validation_mask]) ** 2, axis=0))
        for target, base, scen in zip(target_stats.target_names, baseline_rmse, rmse, strict=True):
            rows.append(
                {
                    "scenario": scenario,
                    "target": target,
                    "baseline_rmse": base,
                    "scenario_rmse": scen,
                    "delta_rmse": scen - base,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def evaluate_model(
    model: SetTransformerModel,
    table: pd.DataFrame,
    split: Split,
    target_stats: TargetStats,
    opts: LatentStratOptions | None = None,
    baselines: Baselines | None = None,
) -> EvaluationReport:
    torch = require_torch()
    opts = opts or default_options()
    red, blue = match_team_matrices(table)
    with torch.no_grad():
        raw = model(torch.as_tensor(red, dtype=torch.long), torch.as_tensor(blue, dtype=torch.long))
    pred_z = raw.cont_z.detach().cpu().numpy()
    pred_cont = pred_z * target_stats.sigma + target_stats.mu
    pred_bin = sigmoid(raw.bin_logits.detach().cpu().numpy())
    actual_cont = target_matrix(table, [mapping.target_name for mapping in opts.target_map])
    actual_bin = target_matrix(table, list(opts.binary_targets))
    parameter_count = model.parameter_count()
    rows_per_parameter = float(np.sum(split.train_mask) / parameter_count)
    red_weights = raw.red_pma_weights.detach().cpu().numpy().reshape(len(table), 3)
    blue_weights = raw.blue_pma_weights.detach().cpu().numpy().reshape(len(table), 3)
    return EvaluationReport(
        parameter_count=parameter_count,
        rows_per_parameter=rows_per_parameter,
        continuous_metrics=continuous_metrics(actual_cont, pred_cont, split, target_stats.target_names),
        binary_metrics=binary_metrics(actual_bin, pred_bin, split, list(opts.binary_targets)),
        calibration=calibration_table(actual_bin, pred_bin, split.validation_mask, list(opts.binary_targets)),
        slices=slice_metrics(table, actual_cont, pred_cont, split, target_stats.target_names),
        set_attention=set_attention_table(table, split, red_weights, blue_weights),
        zero_out_diagnostics=zero_out_diagnostics(model, table, split, target_stats, opts),
        baselines=baselines,
    )
