"""Walk-forward temporal validation for LatentStrat V5.8."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from latentstrat.config import LatentStratOptions, default_options
from latentstrat.data import Split, TargetStats, optional_target_matrix
from latentstrat.evaluation import _predict_batched
from latentstrat.features import (
    FeatureTrainingResult,
    enrich_sidecars_with_event_weeks,
    event_week_table_from_feature_table,
    index_sidecar_tables,
    read_feature_table,
    train_feature_table,
)
from latentstrat.training import (
    AlliancePairDataset,
    RankPairDataset,
    SelectionTripletDataset,
    create_tensorboard_writer,
    match_v5_matrices,
)


@dataclass
class WalkForwardResult:
    output_dir: Path
    metrics: pd.DataFrame
    history: pd.DataFrame
    tensorboard_logdir: Path | None = None


def _usable_week_series(table: pd.DataFrame) -> pd.Series:
    if "event_week" not in table.columns:
        raise ValueError("missing event_week; rebuild prior features with V5.8 week metadata.")
    weeks = pd.to_numeric(table["event_week"], errors="coerce")
    if not np.isfinite(weeks.to_numpy(dtype=float)).any():
        raise ValueError("missing event_week; rebuild prior features with V5.8 week metadata.")
    return weeks


def make_walk_forward_split(
    table: pd.DataFrame, *, train_max_week: int | float, val_exact_week: int | float
) -> Split:
    weeks = _usable_week_series(table)
    train_mask = (weeks <= float(train_max_week)).fillna(False).to_numpy(dtype=bool)
    validation_mask = (weeks == float(val_exact_week)).fillna(False).to_numpy(dtype=bool)
    if np.any(train_mask & validation_mask):
        raise ValueError("walk-forward train and validation masks overlap.")
    return Split(
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=~(train_mask | validation_mask),
        policy="walk-forward",
    )


def _sidecars_with_weeks(
    sidecar_tables: dict[str, pd.DataFrame] | None, feature_table: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    if not sidecar_tables:
        return {}
    enriched = enrich_sidecars_with_event_weeks(
        {name: table.copy() for name, table in sidecar_tables.items()},
        event_week_table_from_feature_table(feature_table),
    )
    for name, table in enriched.items():
        if not table.empty and "event_week" not in table.columns:
            raise ValueError(f"{name} sidecar missing event_week; rebuild V5.8 sidecars.")
    return enriched


def _filter_sidecars(
    sidecars: dict[str, pd.DataFrame],
    *,
    max_week: float | None = None,
    exact_week: float | None = None,
) -> dict[str, pd.DataFrame]:
    filtered = {}
    for name, table in sidecars.items():
        if table.empty:
            filtered[name] = table.copy()
            continue
        weeks = pd.to_numeric(table["event_week"], errors="coerce")
        if max_week is not None:
            mask = weeks <= float(max_week)
        elif exact_week is not None:
            mask = weeks == float(exact_week)
        else:
            mask = pd.Series(True, index=table.index)
        filtered[name] = table.loc[mask.fillna(False)].reset_index(drop=True)
    return filtered


def _mean_validation_mse(metrics: pd.DataFrame) -> float:
    validation = metrics[metrics["split"] == "validation"]
    if validation.empty or "rmse" not in validation.columns:
        return float("nan")
    values = pd.to_numeric(validation["rmse"], errors="coerce").to_numpy(dtype=float)
    return float(np.nanmean(values**2)) if np.isfinite(values).any() else float("nan")


def _validation_brier(metrics: pd.DataFrame) -> float:
    validation = metrics[metrics["split"] == "validation"]
    if validation.empty or "brier" not in validation.columns:
        return float("nan")
    values = pd.to_numeric(validation["brier"], errors="coerce").to_numpy(dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def _validation_common_metrics(metrics: pd.DataFrame) -> dict[str, float | int]:
    columns = [
        "next_match_phase_score_mse",
        "next_match_total_score_mse",
        "match_accuracy",
        "win_brier",
        "win_log_loss",
        "match_count",
        "alliance_score_count",
        "phase_score_sse",
        "phase_score_count",
        "total_score_sse",
        "total_score_count",
        "win_brier_sum",
        "win_log_loss_sum",
        "win_correct_count",
        "win_count",
    ]
    validation = metrics[metrics["split"] == "validation"]
    if validation.empty:
        return {f"val_{column}": float("nan") for column in columns}
    row = validation.iloc[0]
    return {
        f"val_{column}": row[column] if column in validation.columns else float("nan")
        for column in columns
    }


def _foul_target_names(opts: LatentStratOptions) -> list[str]:
    return [f"{color}_{target}" for color in ("red", "blue") for target in opts.foul_targets]


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


def _validation_foul_mse(
    result: FeatureTrainingResult, split: Split, opts: LatentStratOptions
) -> float:
    names = _foul_target_names(opts)
    if not names or not np.any(split.validation_mask):
        return float("nan")
    red, blue, red_event, blue_event, red_missing, blue_missing = match_v5_matrices(
        result.prepared
    )
    pred = _predict_batched(
        result.model,
        red,
        blue,
        opts,
        red_event=red_event,
        blue_event=blue_event,
        red_missing=red_missing,
        blue_missing=blue_missing,
        return_aux=True,
    )
    mu, sigma = _stats_for_names(result.v57_target_stats, names)
    predicted = pred["foul"] * sigma + mu
    actual = optional_target_matrix(result.prepared, names)
    valid = split.validation_mask[:, None] & np.isfinite(actual)
    if not valid.any():
        return float("nan")
    return float(np.mean((predicted[valid] - actual[valid]) ** 2))


def _sidecar_rank_loss(model, table: pd.DataFrame, opts: LatentStratOptions) -> float:
    dataset = RankPairDataset.from_table(table)
    if len(dataset) == 0:
        return float("nan")
    device = next(model.parameters()).device
    with torch.inference_mode():
        higher = dataset.higher_base_idx.to(device)
        lower = dataset.lower_base_idx.to(device)
        higher_event = dataset.higher_event_idx.to(device)
        lower_event = dataset.lower_event_idx.to(device)
        higher_score = model.team_value(higher, higher_event)
        lower_score = model.team_value(lower, lower_event)
        loss = F.margin_ranking_loss(
            higher_score,
            lower_score,
            torch.ones_like(higher_score),
            margin=float(opts.rank_margin),
        )
    return float(loss.detach().cpu())


def _sidecar_playoff_loss(model, table: pd.DataFrame, opts: LatentStratOptions) -> float:
    dataset = AlliancePairDataset.from_table(table)
    if len(dataset) == 0:
        return float("nan")
    device = next(model.parameters()).device
    with torch.inference_mode():
        better = dataset.better_base_idx.to(device)
        worse = dataset.worse_base_idx.to(device)
        better_event = dataset.better_event_idx.to(device)
        worse_event = dataset.worse_event_idx.to(device)
        better_score = model.alliance_value(better, better_event)
        worse_score = model.alliance_value(worse, worse_event)
        loss = F.margin_ranking_loss(
            better_score,
            worse_score,
            torch.ones_like(better_score),
            margin=float(opts.playoff_margin),
        )
    return float(loss.detach().cpu())


def _sidecar_selection_loss(model, table: pd.DataFrame, opts: LatentStratOptions) -> float:
    dataset = SelectionTripletDataset.from_table(table)
    if len(dataset) == 0:
        return float("nan")
    device = next(model.parameters()).device
    with torch.inference_mode():
        anchor = model.team_latent(
            dataset.captain_base_idx.to(device), dataset.captain_event_idx.to(device)
        )
        positive = model.team_latent(
            dataset.pick_base_idx.to(device), dataset.pick_event_idx.to(device)
        )
        negative = model.team_latent(
            dataset.passed_base_idx.to(device), dataset.passed_event_idx.to(device)
        )
        loss = F.triplet_margin_loss(
            anchor, positive, negative, margin=float(opts.selection_triplet_margin)
        )
    return float(loss.detach().cpu())


def _sidecar_validation_metrics(
    result: FeatureTrainingResult,
    validation_sidecars: dict[str, pd.DataFrame],
    opts: LatentStratOptions,
) -> dict[str, float]:
    indexed = index_sidecar_tables(
        validation_sidecars, result.team_index_map, result.team_event_index_map
    ) or {}
    model = result.model
    model.eval()
    return {
        "val_rank_margin_loss": _sidecar_rank_loss(
            model, indexed.get("rankings", pd.DataFrame()), opts
        ),
        "val_playoff_margin_loss": _sidecar_playoff_loss(
            model, indexed.get("playoffs", pd.DataFrame()), opts
        ),
        "val_selection_triplet_loss": _sidecar_selection_loss(
            model, indexed.get("selections", pd.DataFrame()), opts
        ),
    }


def _write_walk_forward_tensorboard_fold(writer: Any, row: dict[str, Any]) -> None:
    step = int(row["fold_number"])
    scalar_map = {
        "WalkForward/Validation_Loss": row.get("best_validation_loss"),
        "WalkForward/Next_Match_Phase_Score_MSE": row.get(
            "val_next_match_phase_score_mse"
        ),
        "WalkForward/Next_Match_Total_Score_MSE": row.get(
            "val_next_match_total_score_mse"
        ),
        "WalkForward/Match_Accuracy": row.get("val_match_accuracy"),
        "WalkForward/Win_Brier": row.get("val_win_brier"),
        "WalkForward/Win_Log_Loss": row.get("val_win_log_loss"),
        "WalkForward/Foul_MSE": row.get("val_foul_mse"),
        "WalkForward/Rank_Margin_Loss": row.get("val_rank_margin_loss"),
        "WalkForward/Playoff_Margin_Loss": row.get("val_playoff_margin_loss"),
        "WalkForward/Selection_Triplet_Loss": row.get("val_selection_triplet_loss"),
        "WalkForward/Train_Rows": row.get("train_rows"),
        "WalkForward/Validation_Rows": row.get("validation_rows"),
    }
    for tag, value in scalar_map.items():
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(scalar):
            writer.add_scalar(tag, scalar, step)


def _weighted_metric(row: dict[str, Any], numerator: str, denominator: str) -> float:
    count = row.get(denominator, np.nan)
    if not np.isfinite(count) or float(count) <= 0:
        return float("nan")
    return float(row.get(numerator, np.nan) / count)


def _average_row(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = metrics.select_dtypes(include=[np.number]).columns
    row: dict[str, Any] = {column: np.nan for column in metrics.columns}
    row["fold_number"] = "AVERAGE"
    row["train_weeks"] = ""
    row["val_week"] = ""
    for column in numeric:
        if column != "fold_number":
            row[column] = float(metrics[column].mean(skipna=True))
    sum_columns = (
        "val_match_count",
        "val_alliance_score_count",
        "val_phase_score_sse",
        "val_phase_score_count",
        "val_total_score_sse",
        "val_total_score_count",
        "val_win_brier_sum",
        "val_win_log_loss_sum",
        "val_win_correct_count",
        "val_win_count",
    )
    for column in sum_columns:
        if column in metrics.columns:
            row[column] = float(pd.to_numeric(metrics[column], errors="coerce").sum(skipna=True))
    if "val_next_match_phase_score_mse" in metrics.columns:
        row["val_next_match_phase_score_mse"] = _weighted_metric(
            row, "val_phase_score_sse", "val_phase_score_count"
        )
    if "val_next_match_total_score_mse" in metrics.columns:
        row["val_next_match_total_score_mse"] = _weighted_metric(
            row, "val_total_score_sse", "val_total_score_count"
        )
    if "val_match_accuracy" in metrics.columns:
        row["val_match_accuracy"] = _weighted_metric(
            row, "val_win_correct_count", "val_win_count"
        )
    if "val_win_brier" in metrics.columns:
        row["val_win_brier"] = _weighted_metric(row, "val_win_brier_sum", "val_win_count")
    if "val_win_log_loss" in metrics.columns:
        row["val_win_log_loss"] = _weighted_metric(row, "val_win_log_loss_sum", "val_win_count")
    return pd.DataFrame([row], columns=metrics.columns)


def run_walk_forward_validation(
    features_path: str | Path,
    output_dir: str | Path,
    *,
    prior_checkpoint: str | Path,
    opts: LatentStratOptions | None = None,
    sidecar_tables: dict[str, pd.DataFrame] | None = None,
    min_train_week: int = 1,
    max_validation_week: int | None = None,
    save_fold_checkpoints: bool = False,
    tensorboard_logdir: str | Path | None = None,
    tensorboard_run_name: str | None = None,
    verbose: bool = True,
) -> WalkForwardResult:
    opts = opts or default_options()
    feature_table = read_feature_table(features_path)
    weeks = _usable_week_series(feature_table)
    finite_weeks = sorted(int(week) for week in pd.Series(weeks).dropna().unique())
    if len(finite_weeks) < 2:
        raise ValueError("walk-forward validation requires at least two event weeks.")
    max_val = max_validation_week if max_validation_week is not None else max(finite_weeks)
    sidecars = _sidecars_with_weeks(sidecar_tables, feature_table)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out / "fold_checkpoints"
    if save_fold_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_logdir: Path | None = None
    summary_writer = None
    if tensorboard_logdir is not None:
        run_name = tensorboard_run_name or f"v58_walk_forward_{opts.season}_{int(time.time())}"
        run_logdir = Path(tensorboard_logdir) / run_name
        summary_writer = create_tensorboard_writer(str(run_logdir / "summary"))

    metric_rows = []
    history_rows = []
    fold_number = 0
    try:
        for train_week in finite_weeks:
            val_week = train_week + 1
            if train_week < min_train_week or val_week > max_val or val_week not in finite_weeks:
                continue
            split = make_walk_forward_split(
                feature_table, train_max_week=train_week, val_exact_week=val_week
            )
            if not np.any(split.train_mask) or not np.any(split.validation_mask):
                continue
            fold_number += 1
            fold_opts = opts.model_copy(
                update={
                    "use_early_stopping": False,
                    "restore_best_validation_model": True,
                }
            )
            train_sidecars = _filter_sidecars(sidecars, max_week=train_week)
            validation_sidecars = _filter_sidecars(sidecars, exact_week=val_week)
            if verbose:
                print(
                    f"Fold {fold_number}: train weeks <= {train_week}, "
                    f"validate week {val_week}"
                )
            fold_writer = (
                create_tensorboard_writer(
                    str(run_logdir / f"fold_{fold_number:02d}_val_week_{val_week}")
                )
                if run_logdir is not None
                else None
            )
            try:
                result = train_feature_table(
                    feature_table,
                    fold_opts,
                    verbose=verbose,
                    prior_checkpoint=prior_checkpoint,
                    sidecar_tables=train_sidecars,
                    split_override=split,
                    tensorboard_writer=fold_writer,
                    tensorboard_logdir=(
                        run_logdir / f"fold_{fold_number:02d}_val_week_{val_week}"
                    )
                    if run_logdir is not None
                    else None,
                )
            finally:
                if fold_writer is not None:
                    fold_writer.close()
            if save_fold_checkpoints:
                torch.save(
                    {
                        "model_state_dict": result.model.state_dict(),
                        "options": result.model.opts.model_dump(mode="json"),
                        "team_base_index_map": result.team_index_map,
                        "team_event_index_map": result.team_event_index_map,
                        "prior_checkpoint": str(prior_checkpoint),
                        "fold_number": fold_number,
                        "train_max_week": train_week,
                        "val_week": val_week,
                    },
                    checkpoint_dir / f"fold_{fold_number:02d}_val_week_{val_week}.pt",
                )
            history = result.history.copy()
            history.insert(0, "fold_number", fold_number)
            history.insert(1, "train_max_week", train_week)
            history.insert(2, "val_week", val_week)
            history_rows.append(history)
            sidecar_metrics = _sidecar_validation_metrics(result, validation_sidecars, fold_opts)
            common_metrics = _validation_common_metrics(result.report.common_metrics)
            row = {
                "fold_number": fold_number,
                "train_weeks": f"1-{train_week}",
                "val_week": val_week,
                "train_rows": int(np.sum(split.train_mask)),
                "validation_rows": int(np.sum(split.validation_mask)),
                "val_match_mse": _mean_validation_mse(result.report.continuous_metrics),
                "val_win_brier": _validation_brier(result.report.binary_metrics),
                "val_foul_mse": _validation_foul_mse(result, split, fold_opts),
                **common_metrics,
                **sidecar_metrics,
                "best_epoch": result.diagnostics.best_epoch,
                "best_validation_loss": result.diagnostics.best_validation_loss,
                "restored_best_validation_model": result.diagnostics.restored_best_validation_model,
            }
            metric_rows.append(row)
            if summary_writer is not None:
                _write_walk_forward_tensorboard_fold(summary_writer, row)
    finally:
        if summary_writer is not None:
            summary_writer.close()
    if not metric_rows:
        raise ValueError("No valid walk-forward folds were produced.")
    metrics = pd.DataFrame(metric_rows)
    metrics = pd.concat([metrics, _average_row(metrics)], ignore_index=True)
    history = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    metrics.to_csv(out / "walk_forward_metrics.csv", index=False)
    history.to_csv(out / "walk_forward_history.csv", index=False)
    return WalkForwardResult(
        output_dir=out, metrics=metrics, history=history, tensorboard_logdir=run_logdir
    )
