"""Evaluation metrics and diagnostics for LatentStrat V5 models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from latentstrat.baselines import Baselines
from latentstrat.config import LatentStratOptions, default_options
from latentstrat.season.data import Split, TargetStats, optional_target_matrix, target_matrix
from latentstrat.season.model import SetTransformerModel
from latentstrat.season.train import (
    award_target_tensor,
    endgame_target_matrix,
    match_team_matrices,
    match_v5_matrices,
)


@dataclass
class EvaluationReport:
    parameter_count: int
    rows_per_parameter: float
    continuous_metrics: pd.DataFrame
    binary_metrics: pd.DataFrame
    common_metrics: pd.DataFrame
    endgame_metrics: pd.DataFrame
    award_metrics: pd.DataFrame
    calibration: pd.DataFrame
    slices: dict[str, pd.DataFrame]
    set_attention: pd.DataFrame
    zero_out_diagnostics: pd.DataFrame
    team_zero_out_sensitivity: pd.DataFrame
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
            rows.append(
                {"split": split_name, "target": target, "rmse": rmse_value, "mae": mae_value}
            )
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


def _safe_metric(sse: float, count: int) -> float:
    return float(sse / count) if count > 0 else float("nan")


def _safe_rate(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _sse_and_count(predicted: np.ndarray, actual: np.ndarray) -> tuple[float, int]:
    valid = np.isfinite(predicted) & np.isfinite(actual)
    if not np.any(valid):
        return 0.0, 0
    err = predicted[valid] - actual[valid]
    return float(np.sum(err**2)), int(np.sum(valid))


def _stats_for_names(stats: TargetStats | None, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if stats is None:
        return np.zeros(len(names), dtype=float), np.ones(len(names), dtype=float)
    index = {name: idx for idx, name in enumerate(stats.target_names)}
    mu = np.zeros(len(names), dtype=float)
    sigma = np.ones(len(names), dtype=float)
    for out_idx, name in enumerate(names):
        if name in index:
            stat_idx = index[name]
            mu[out_idx] = float(stats.mu[stat_idx])
            sigma[out_idx] = float(stats.sigma[stat_idx])
    return mu, sigma


def _phase_score_predictions(
    table: pd.DataFrame, pred_cont_z: np.ndarray, target_stats: TargetStats
) -> tuple[np.ndarray, np.ndarray]:
    target_names = list(target_stats.target_names)
    predicted_cont = pred_cont_z * target_stats.sigma + target_stats.mu
    actual_cont = optional_target_matrix(table, target_names)
    index = {name: idx for idx, name in enumerate(target_names)}
    required = ("red_auto_pts", "red_teleop_pts", "blue_auto_pts", "blue_teleop_pts")
    if not all(name in index for name in required):
        empty = np.full((len(table), 2), np.nan, dtype=float)
        return empty, empty.copy()
    red_pred = predicted_cont[:, index["red_auto_pts"]] + predicted_cont[:, index["red_teleop_pts"]]
    blue_pred = (
        predicted_cont[:, index["blue_auto_pts"]] + predicted_cont[:, index["blue_teleop_pts"]]
    )
    red_actual = actual_cont[:, index["red_auto_pts"]] + actual_cont[:, index["red_teleop_pts"]]
    blue_actual = actual_cont[:, index["blue_auto_pts"]] + actual_cont[:, index["blue_teleop_pts"]]
    return np.column_stack([red_pred, blue_pred]), np.column_stack([red_actual, blue_actual])


def _total_score_predictions(
    table: pd.DataFrame,
    pred_phase: np.ndarray,
    pred_foul_z: np.ndarray | None,
    v57_target_stats: TargetStats | None,
    opts: LatentStratOptions,
) -> tuple[np.ndarray, np.ndarray]:
    if pred_foul_z is None or v57_target_stats is None:
        empty = np.full((len(table), 2), np.nan, dtype=float)
        return empty, empty.copy()
    if "red_total_score" not in table.columns or "blue_total_score" not in table.columns:
        empty = np.full((len(table), 2), np.nan, dtype=float)
        return empty, empty.copy()
    foul_names = [
        f"{color}_{target}" for color in ("red", "blue") for target in opts.foul_targets
    ]
    if pred_foul_z.shape[1] < len(foul_names):
        empty = np.full((len(table), 2), np.nan, dtype=float)
        return empty, empty.copy()
    if "red_committed_foul_pts" not in foul_names or "blue_committed_foul_pts" not in foul_names:
        empty = np.full((len(table), 2), np.nan, dtype=float)
        return empty, empty.copy()
    mu, sigma = _stats_for_names(v57_target_stats, foul_names)
    predicted_fouls = pred_foul_z * sigma + mu
    red_committed_idx = foul_names.index("red_committed_foul_pts")
    blue_committed_idx = foul_names.index("blue_committed_foul_pts")
    predicted_total = np.column_stack(
        [
            pred_phase[:, 0] + predicted_fouls[:, blue_committed_idx],
            pred_phase[:, 1] + predicted_fouls[:, red_committed_idx],
        ]
    )
    actual_total = optional_target_matrix(table, ["red_total_score", "blue_total_score"])
    return predicted_total, actual_total


def common_match_metrics_from_predictions(
    table: pd.DataFrame,
    split: Split,
    target_stats: TargetStats,
    opts: LatentStratOptions | None,
    pred_cont_z: np.ndarray,
    pred_bin_logits: np.ndarray,
    *,
    v57_target_stats: TargetStats | None = None,
    pred_foul_z: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute community-facing match metrics from model predictions.

    Win probability is red-alliance oriented because `red_win` is the canonical
    binary target. Blue win probability is `1 - p_red_win`.
    """

    opts = opts or default_options()
    pred_phase, actual_phase = _phase_score_predictions(table, pred_cont_z, target_stats)
    pred_total, actual_total = _total_score_predictions(
        table, pred_phase, pred_foul_z, v57_target_stats, opts
    )
    binary_names = list(opts.binary_targets)
    actual_bin = optional_target_matrix(table, binary_names)
    red_win_idx = binary_names.index("red_win") if "red_win" in binary_names else None
    red_win_prob = (
        sigmoid(pred_bin_logits[:, red_win_idx])
        if red_win_idx is not None and pred_bin_logits.shape[1] > red_win_idx
        else np.full(len(table), np.nan, dtype=float)
    )
    red_win_actual = (
        actual_bin[:, red_win_idx]
        if red_win_idx is not None
        else np.full(len(table), np.nan, dtype=float)
    )
    rows = []
    for split_name, mask in (("train", split.train_mask), ("validation", split.validation_mask)):
        phase_sse, phase_count = _sse_and_count(pred_phase[mask], actual_phase[mask])
        total_sse, total_count = _sse_and_count(pred_total[mask], actual_total[mask])
        win_valid = mask & np.isfinite(red_win_prob) & np.isfinite(red_win_actual)
        if np.any(win_valid):
            probs = np.clip(red_win_prob[win_valid], np.finfo(float).eps, 1 - np.finfo(float).eps)
            actual = red_win_actual[win_valid].astype(float)
            brier_values = (probs - actual) ** 2
            log_loss_values = -(actual * np.log(probs) + (1 - actual) * np.log(1 - probs))
            correct = (probs >= 0.5) == (actual >= 0.5)
            win_brier_sum = float(np.sum(brier_values))
            win_log_loss_sum = float(np.sum(log_loss_values))
            win_correct_count = int(np.sum(correct))
            win_count = int(np.sum(win_valid))
        else:
            win_brier_sum = 0.0
            win_log_loss_sum = 0.0
            win_correct_count = 0
            win_count = 0
        rows.append(
            {
                "split": split_name,
                "next_match_phase_score_mse": _safe_metric(phase_sse, phase_count),
                "next_match_total_score_mse": _safe_metric(total_sse, total_count),
                "match_accuracy": _safe_rate(win_correct_count, win_count),
                "win_brier": _safe_rate(win_brier_sum, win_count),
                "win_log_loss": _safe_rate(win_log_loss_sum, win_count),
                "match_count": win_count,
                "alliance_score_count": phase_count,
                "phase_score_sse": phase_sse,
                "phase_score_count": phase_count,
                "total_score_sse": total_sse,
                "total_score_count": total_count,
                "win_brier_sum": win_brier_sum,
                "win_log_loss_sum": win_log_loss_sum,
                "win_correct_count": win_correct_count,
                "win_count": win_count,
            }
        )
    return pd.DataFrame(rows)


def calibration_table(
    actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray, names: list[str]
) -> pd.DataFrame:
    rows = []
    if not names or not np.any(mask):
        return pd.DataFrame(
            columns=["target", "bin_low", "bin_high", "count", "mean_probability", "observed_rate"]
        )
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
                    "mean_probability": float(np.nanmean(probs[bin_mask]))
                    if np.any(bin_mask)
                    else np.nan,
                    "observed_rate": float(np.nanmean(outcomes[bin_mask]))
                    if np.any(bin_mask)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def ordinal_predicted_class(logits: np.ndarray) -> np.ndarray:
    return np.sum(sigmoid(logits) >= 0.5, axis=-1)


def ordinal_expected_value(logits: np.ndarray) -> np.ndarray:
    return np.sum(sigmoid(logits), axis=-1)


def endgame_metrics(
    actual: np.ndarray,
    logits: np.ndarray,
    missing_mask: np.ndarray,
    split: Split,
) -> pd.DataFrame:
    rows = []
    pred_class = ordinal_predicted_class(logits)
    pred_expected = ordinal_expected_value(logits)
    split_masks = (("train", split.train_mask), ("validation", split.validation_mask))
    for split_name, row_mask in split_masks:
        slot_mask = row_mask[:, None] & ~missing_mask
        if not np.any(slot_mask):
            rows.append(
                {
                    "split": split_name,
                    "accuracy": np.nan,
                    "mean_abs_class_error": np.nan,
                    "expected_level_mae": np.nan,
                    "count": 0,
                }
            )
            continue
        actual_values = actual[slot_mask]
        mean_abs_class_error = float(np.mean(np.abs(pred_class[slot_mask] - actual_values)))
        expected_level_mae = float(np.mean(np.abs(pred_expected[slot_mask] - actual_values)))
        rows.append(
            {
                "split": split_name,
                "accuracy": float(np.mean(pred_class[slot_mask] == actual_values)),
                "mean_abs_class_error": mean_abs_class_error,
                "expected_level_mae": expected_level_mae,
                "count": int(np.sum(slot_mask)),
            }
        )
    return pd.DataFrame(rows)


def award_metrics(
    actual: np.ndarray,
    logits: np.ndarray,
    missing_mask: np.ndarray,
    split: Split,
    names: list[str],
) -> pd.DataFrame:
    rows = []
    probs = sigmoid(logits)
    split_masks = (("train", split.train_mask), ("validation", split.validation_mask))
    for split_name, row_mask in split_masks:
        valid = np.isfinite(actual) & row_mask[:, None, None] & ~missing_mask[:, :, None]
        for axis_idx, name in enumerate(names):
            axis_valid = valid[:, :, axis_idx]
            if not np.any(axis_valid):
                rows.append(
                    {
                        "split": split_name,
                        "target": name,
                        "positive_count": 0,
                        "bce": np.nan,
                        "average_precision": np.nan,
                    }
                )
                continue
            y_true = actual[:, :, axis_idx][axis_valid]
            y_prob = probs[:, :, axis_idx][axis_valid]
            bce = F.binary_cross_entropy(
                torch.as_tensor(y_prob, dtype=torch.float32),
                torch.as_tensor(y_true, dtype=torch.float32),
            ).item()
            average_precision = np.nan
            if np.unique(y_true).size > 1:
                from sklearn.metrics import average_precision_score

                average_precision = float(average_precision_score(y_true, y_prob))
            rows.append(
                {
                    "split": split_name,
                    "target": name,
                    "positive_count": int(np.sum(y_true == 1)),
                    "bce": bce,
                    "average_precision": average_precision,
                }
            )
    return pd.DataFrame(rows)


def _team_idx_matrix(table: pd.DataFrame) -> np.ndarray:
    red, blue = match_team_matrices(table)
    return np.concatenate([red, blue], axis=1)


def _row_training_match_counts(table: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
    team_idx = _team_idx_matrix(table)
    positive = team_idx[team_idx > 0]
    minlength = int(positive.max()) + 1 if positive.size else 1
    counts = np.bincount(team_idx[train_mask].reshape(-1).clip(min=0), minlength=minlength)
    clipped = team_idx.clip(min=0)
    if clipped.max() >= len(counts):
        counts = np.pad(counts, (0, clipped.max() - len(counts) + 1))
    return counts[clipped]


def _row_has_first_event_team(table: pd.DataFrame) -> np.ndarray:
    team_idx = _team_idx_matrix(table)
    max_team = int(team_idx.max()) if team_idx.size else 0
    first_event = np.array([""] * (max_team + 1), dtype=object)
    order = np.argsort(table["sort_ordinal"].to_numpy())
    events = table["event_key"].astype(str).to_numpy()
    for row in order:
        for team in team_idx[row]:
            if team > 0 and first_event[team] == "":
                first_event[team] = events[row]
    values = []
    for row in range(len(table)):
        row_teams = team_idx[row][team_idx[row] > 0]
        values.append(np.any(first_event[row_teams] == events[row]))
    return np.array(values)


def slice_metrics(
    table: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    split: Split,
    target_names: list[str],
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


def set_attention_table(
    table: pd.DataFrame,
    split: Split,
    red_weights: np.ndarray,
    blue_weights: np.ndarray,
    red_missing: np.ndarray | None = None,
    blue_missing: np.ndarray | None = None,
) -> pd.DataFrame:
    rows = []
    split_names = np.array(["other"] * len(table), dtype=object)
    split_names[split.train_mask] = "train"
    split_names[split.validation_mask] = "validation"
    split_names[split.test_mask] = "test"
    if red_missing is None:
        red_missing = np.zeros((len(table), 3), dtype=bool)
    if blue_missing is None:
        blue_missing = np.zeros((len(table), 3), dtype=bool)
    for color, weights, prefix, missing in (
        ("red", red_weights, "red", red_missing),
        ("blue", blue_weights, "blue", blue_missing),
    ):
        team_cols = [f"{prefix}_team_1_key", f"{prefix}_team_2_key", f"{prefix}_team_3_key"]
        keys = table[team_cols].astype(str).to_numpy()
        entropy = -np.sum(weights * np.log(np.maximum(weights, np.finfo(float).eps)), axis=1)
        max_slot = np.argmax(weights, axis=1)
        for idx in range(len(table)):
            all_missing = bool(np.all(missing[idx]))
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
                    "team_1_missing": bool(missing[idx, 0]),
                    "team_2_missing": bool(missing[idx, 1]),
                    "team_3_missing": bool(missing[idx, 2]),
                    "team_1_attention": weights[idx, 0],
                    "team_2_attention": weights[idx, 1],
                    "team_3_attention": weights[idx, 2],
                    "max_attention_slot": np.nan if all_missing else int(max_slot[idx] + 1),
                    "max_attention_team_key": "" if all_missing else keys[idx, max_slot[idx]],
                    "max_attention_weight": np.nan if all_missing else weights[idx, max_slot[idx]],
                    "attention_entropy": entropy[idx],
                }
            )
    return pd.DataFrame(rows)


def _predict_batched(
    model: SetTransformerModel,
    red: np.ndarray,
    blue: np.ndarray,
    opts: LatentStratOptions | None = None,
    *,
    red_event: np.ndarray | None = None,
    blue_event: np.ndarray | None = None,
    red_missing: np.ndarray | None = None,
    blue_missing: np.ndarray | None = None,
    return_aux: bool = False,
    **forward_kwargs,
):
    opts = opts or default_options()
    device = next(model.parameters()).device
    red_event = np.zeros_like(red) if red_event is None else red_event
    blue_event = np.zeros_like(blue) if blue_event is None else blue_event
    red_missing = red == 0 if red_missing is None else red_missing
    blue_missing = blue == 0 if blue_missing is None else blue_missing
    dataset = TensorDataset(
        torch.as_tensor(red, dtype=torch.long),
        torch.as_tensor(blue, dtype=torch.long),
        torch.as_tensor(red_event, dtype=torch.long),
        torch.as_tensor(blue_event, dtype=torch.long),
        torch.as_tensor(red_missing, dtype=torch.bool),
        torch.as_tensor(blue_missing, dtype=torch.bool),
    )
    loader = DataLoader(
        dataset,
        batch_size=max(int(opts.eval_batch_size), 1),
        shuffle=False,
        num_workers=opts.dataloader_num_workers,
        pin_memory=device.type == "cuda",
    )
    was_training = model.training
    model.eval()
    chunks = {
        "cont": [],
        "bin": [],
        "atomic": [],
        "foul": [],
        "bonus": [],
        "special": [],
        "endgame": [],
        "award": [],
        "red_pma": [],
        "blue_pma": [],
        "red_missing": [],
        "blue_missing": [],
    }
    non_blocking = device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            (
                batch_red,
                batch_blue,
                batch_red_event,
                batch_blue_event,
                batch_red_missing,
                batch_blue_missing,
            ) = batch
            pred = model(
                batch_red.to(device, non_blocking=non_blocking),
                batch_blue.to(device, non_blocking=non_blocking),
                red_event_idx=batch_red_event.to(device, non_blocking=non_blocking),
                blue_event_idx=batch_blue_event.to(device, non_blocking=non_blocking),
                red_missing_mask=batch_red_missing.to(device, non_blocking=non_blocking),
                blue_missing_mask=batch_blue_missing.to(device, non_blocking=non_blocking),
                **forward_kwargs,
            )
            predictions = pred.predictions
            representations = pred.representations
            masks = pred.masks
            chunks["cont"].append(predictions["continuous"].detach().cpu().numpy())
            chunks["bin"].append(predictions["win"].detach().cpu().numpy())
            chunks["atomic"].append(predictions["atomic"].detach().cpu().numpy())
            chunks["foul"].append(predictions["foul"].detach().cpu().numpy())
            chunks["bonus"].append(predictions["bonus"].detach().cpu().numpy())
            chunks["special"].append(predictions["special"].detach().cpu().numpy())
            chunks["endgame"].append(predictions["endgame"].detach().cpu().numpy())
            chunks["award"].append(predictions["award"].detach().cpu().numpy())
            chunks["red_pma"].append(representations.red_pma_weights.detach().cpu().numpy())
            chunks["blue_pma"].append(representations.blue_pma_weights.detach().cpu().numpy())
            chunks["red_missing"].append(masks.red_missing_mask.detach().cpu().numpy())
            chunks["blue_missing"].append(masks.blue_missing_mask.detach().cpu().numpy())
    if was_training:
        model.train()
    result = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    if return_aux:
        return result
    return result["cont"], result["bin"], result["red_pma"], result["blue_pma"]


def _predict_cont(
    model: SetTransformerModel,
    red: np.ndarray,
    blue: np.ndarray,
    target_stats: TargetStats,
    opts: LatentStratOptions | None = None,
    *,
    red_event: np.ndarray | None = None,
    blue_event: np.ndarray | None = None,
    red_missing: np.ndarray | None = None,
    blue_missing: np.ndarray | None = None,
    pma_mode: str = "learned",
    red_zero_slot: int = 0,
    blue_zero_slot: int = 0,
) -> np.ndarray:
    pred = _predict_batched(
        model,
        red,
        blue,
        opts,
        red_event=red_event,
        blue_event=blue_event,
        red_missing=red_missing,
        blue_missing=blue_missing,
        pma_mode=pma_mode,
        red_zero_slot=red_zero_slot,
        blue_zero_slot=blue_zero_slot,
        return_aux=True,
    )
    return pred["cont"] * target_stats.sigma + target_stats.mu


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
    red, blue, red_event, blue_event, red_missing, blue_missing = match_v5_matrices(table)
    actual = target_matrix(table, [mapping.target_name for mapping in opts.target_map])
    baseline = _predict_cont(
        model, red, blue, target_stats, opts, red_event=red_event, blue_event=blue_event,
        red_missing=red_missing, blue_missing=blue_missing
    )
    baseline_rmse = np.sqrt(
        np.nanmean((baseline[split.validation_mask] - actual[split.validation_mask]) ** 2, axis=0)
    )
    scenarios = (
        [("uniform_pma", "uniform", 0, 0)]
        + [(f"zero_red_slot_{slot}", "learned", slot, 0) for slot in (1, 2, 3)]
        + [(f"zero_blue_slot_{slot}", "learned", 0, slot) for slot in (1, 2, 3)]
    )
    rows = []
    for scenario, pma_mode, red_slot, blue_slot in scenarios:
        predicted = _predict_cont(
            model,
            red,
            blue,
            target_stats,
            opts,
            red_event=red_event,
            blue_event=blue_event,
            red_missing=red_missing,
            blue_missing=blue_missing,
            pma_mode=pma_mode,
            red_zero_slot=red_slot,
            blue_zero_slot=blue_slot,
        )
        rmse = np.sqrt(
            np.nanmean(
                (predicted[split.validation_mask] - actual[split.validation_mask]) ** 2, axis=0
            )
        )
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


def _team_zero_out_columns() -> list[str]:
    return [
        "team_key",
        "team_base_idx",
        "target",
        "appearance_count",
        "event_count",
        "baseline_rmse",
        "zeroed_rmse",
        "delta_rmse",
    ]


def _phase_target_indices(target_names: list[str], color: str) -> list[int]:
    wanted = [f"{color}_auto_pts", f"{color}_teleop_pts"]
    return [target_names.index(name) for name in wanted if name in target_names]


def team_zero_out_sensitivity(
    model: SetTransformerModel,
    table: pd.DataFrame,
    split: Split,
    target_stats: TargetStats,
    opts: LatentStratOptions,
) -> pd.DataFrame:
    """Aggregate validation sensitivity when each visible team slot is zeroed."""

    columns = _team_zero_out_columns()
    if not np.any(split.validation_mask):
        return pd.DataFrame(columns=columns)
    red, blue, red_event, blue_event, red_missing, blue_missing = match_v5_matrices(table)
    target_names = list(target_stats.target_names)
    actual = target_matrix(table, target_names)
    baseline = _predict_cont(
        model,
        red,
        blue,
        target_stats,
        opts,
        red_event=red_event,
        blue_event=blue_event,
        red_missing=red_missing,
        blue_missing=blue_missing,
    )
    rows = []
    for color, team_idx, missing, zero_kwargs in (
        ("red", red, red_missing, {"red_zero_slot": 0, "blue_zero_slot": 0}),
        ("blue", blue, blue_missing, {"red_zero_slot": 0, "blue_zero_slot": 0}),
    ):
        color_target_indices = [
            idx for idx, name in enumerate(target_names) if name.startswith(f"{color}_")
        ]
        phase_indices = _phase_target_indices(target_names, color)
        for slot in (1, 2, 3):
            kwargs = dict(zero_kwargs)
            kwargs[f"{color}_zero_slot"] = slot
            scenario = _predict_cont(
                model,
                red,
                blue,
                target_stats,
                opts,
                red_event=red_event,
                blue_event=blue_event,
                red_missing=red_missing,
                blue_missing=blue_missing,
                **kwargs,
            )
            slot_idx = slot - 1
            valid_rows = np.flatnonzero(
                split.validation_mask & ~missing[:, slot_idx] & (team_idx[:, slot_idx] > 0)
            )
            if not len(valid_rows):
                continue
            team_keys = table[f"{color}_team_{slot}_key"].astype(str).to_numpy()
            for row_idx in valid_rows:
                base_team_idx = int(team_idx[row_idx, slot_idx])
                for target_idx in color_target_indices:
                    actual_value = actual[row_idx, target_idx]
                    if not np.isfinite(actual_value):
                        continue
                    rows.append(
                        {
                            "team_key": team_keys[row_idx],
                            "team_base_idx": base_team_idx,
                            "event_key": str(table["event_key"].iloc[row_idx]),
                            "target": target_names[target_idx],
                            "baseline_squared_error": float(
                                (baseline[row_idx, target_idx] - actual_value) ** 2
                            ),
                            "zeroed_squared_error": float(
                                (scenario[row_idx, target_idx] - actual_value) ** 2
                            ),
                        }
                    )
                if len(phase_indices) >= 2:
                    actual_phase = float(np.sum(actual[row_idx, phase_indices]))
                    if np.isfinite(actual_phase):
                        rows.append(
                            {
                                "team_key": team_keys[row_idx],
                                "team_base_idx": base_team_idx,
                                "event_key": str(table["event_key"].iloc[row_idx]),
                                "target": "alliance_phase_score",
                                "baseline_squared_error": float(
                                    (np.sum(baseline[row_idx, phase_indices]) - actual_phase) ** 2
                                ),
                                "zeroed_squared_error": float(
                                    (np.sum(scenario[row_idx, phase_indices]) - actual_phase) ** 2
                                ),
                            }
                        )
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["team_key", "team_base_idx", "target"], as_index=False)
        .agg(
            appearance_count=("baseline_squared_error", "size"),
            event_count=("event_key", "nunique"),
            baseline_mse=("baseline_squared_error", "mean"),
            zeroed_mse=("zeroed_squared_error", "mean"),
        )
    )
    grouped["baseline_rmse"] = np.sqrt(grouped.pop("baseline_mse"))
    grouped["zeroed_rmse"] = np.sqrt(grouped.pop("zeroed_mse"))
    grouped["delta_rmse"] = grouped["zeroed_rmse"] - grouped["baseline_rmse"]
    grouped = grouped.sort_values(["target", "delta_rmse"], ascending=[True, False])
    return grouped[columns]


def evaluate_model(
    model: SetTransformerModel,
    table: pd.DataFrame,
    split: Split,
    target_stats: TargetStats,
    opts: LatentStratOptions | None = None,
    baselines: Baselines | None = None,
    v57_target_stats: TargetStats | None = None,
) -> EvaluationReport:
    opts = opts or default_options()
    red, blue, red_event, blue_event, red_missing, blue_missing = match_v5_matrices(table)
    pred = _predict_batched(
        model,
        red,
        blue,
        opts,
        red_event=red_event,
        blue_event=blue_event,
        red_missing=red_missing,
        blue_missing=blue_missing,
        return_aux=True,
    )
    pred_cont = pred["cont"] * target_stats.sigma + target_stats.mu
    pred_bin = sigmoid(pred["bin"])
    actual_cont = target_matrix(table, [mapping.target_name for mapping in opts.target_map])
    actual_bin = target_matrix(table, list(opts.binary_targets))
    actual_endgame = endgame_target_matrix(table, opts)
    actual_awards = award_target_tensor(table, opts)
    parameter_count = model.parameter_count()
    rows_per_parameter = float(np.sum(split.train_mask) / parameter_count)
    red_weights = pred["red_pma"].reshape(len(table), 3)
    blue_weights = pred["blue_pma"].reshape(len(table), 3)
    effective_missing = np.concatenate([pred["red_missing"], pred["blue_missing"]], axis=1)
    return EvaluationReport(
        parameter_count=parameter_count,
        rows_per_parameter=rows_per_parameter,
        continuous_metrics=continuous_metrics(
            actual_cont, pred_cont, split, target_stats.target_names
        ),
        binary_metrics=binary_metrics(actual_bin, pred_bin, split, list(opts.binary_targets)),
        common_metrics=common_match_metrics_from_predictions(
            table,
            split,
            target_stats,
            opts,
            pred["cont"],
            pred["bin"],
            v57_target_stats=v57_target_stats,
            pred_foul_z=pred["foul"],
        ),
        endgame_metrics=endgame_metrics(actual_endgame, pred["endgame"], effective_missing, split),
        award_metrics=award_metrics(
            actual_awards, pred["award"], effective_missing, split, list(opts.award_targets)
        ),
        calibration=calibration_table(
            actual_bin, pred_bin, split.validation_mask, list(opts.binary_targets)
        ),
        slices=slice_metrics(table, actual_cont, pred_cont, split, target_stats.target_names),
        set_attention=set_attention_table(
            table, split, red_weights, blue_weights, pred["red_missing"], pred["blue_missing"]
        ),
        zero_out_diagnostics=zero_out_diagnostics(model, table, split, target_stats, opts),
        team_zero_out_sensitivity=team_zero_out_sensitivity(
            model, table, split, target_stats, opts
        ),
        baselines=baselines,
    )
